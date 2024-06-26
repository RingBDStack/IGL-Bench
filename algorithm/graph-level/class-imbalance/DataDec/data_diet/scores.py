from .gradients import compute_mean_logit_gradients, flatten_jacobian, get_mean_logit_gradients_fn
from .metrics import cross_entropy_loss
import flax.linen as nn
from jax import jacrev, jit, vmap
import jax.numpy as jnp
import numpy as np
import torch


def get_lord_error_fn(logits, Y, ord):
  errors = torch.nn.functional.softmax(logits, dim=1) - Y
  scores = np.linalg.norm(errors.detach().cpu().numpy(), ord=ord, axis=-1)
  return scores

def get_grad_norm_fn(loss):
  loss_grads = loss.grad
  scores = np.linalg.norm(loss_grads, axis=-1)
  return scores

def get_margin_error(fn, params, state, score_type):
  fn_jit = jit(lambda X: fn(params, state, X))

  def margin_error(X, Y):
    batch_sz = X.shape[0]
    P = np.array(nn.softmax(fn_jit(X)))
    correct_logits = Y.astype(bool)
    margins = P[~correct_logits].reshape(batch_sz, -1) - P[correct_logits].reshape(batch_sz, 1)
    if score_type == 'max':
      scores = np.max(margins, -1)
    elif score_type == 'sum':
      scores = np.sum(margins, -1)
    return scores

  return margin_error


def _get_grad_norm_fn(fn, params, state):

  @jit
  def score_fn(X, Y):
    per_sample_loss_fn = lambda p, x, y: vmap(cross_entropy_loss)(fn(p, state, x), y)
    loss_grads = flatten_jacobian(jacrev(per_sample_loss_fn)(params, X, Y))
    scores = jnp.linalg.norm(loss_grads, axis=-1)
    return scores

  return lambda X, Y: np.array(score_fn(X, Y))


def get_score_fn(fn, params, state, score_type):
  if score_type == 'l2_error':
    print(f'compute {score_type}...')
    score_fn = get_lord_error_fn(fn, params, state, 2)
  elif score_type == 'l1_error':
    print(f'compute {score_type}...')
    score_fn = get_lord_error_fn(fn, params, state, 1)
  elif score_type == 'max_margin':
    print(f'compute {score_type}...')
    score_fn = get_margin_error(fn, params, state, 'max')
  elif score_type == 'sum_margin':
    print(f'compute {score_type}...')
    score_fn = get_margin_error(fn, params, state, 'sum')
  elif score_type == 'grad_norm':
    print(f'compute {score_type}...')
    score_fn = get_grad_norm_fn(fn, params, state)
  else:
    raise NotImplementedError
  return score_fn


def compute_scores(fn, params, state, X, Y, batch_sz, score_type):
  n_batches = X.shape[0] // batch_sz
  Xs, Ys = np.split(X, n_batches), np.split(Y, n_batches)
  score_fn = get_score_fn(fn, params, state, score_type)
  scores = []
  for i, (X, Y) in enumerate(zip(Xs, Ys)):
    print(f'score batch {i+1} of {n_batches}')
    scores.append(score_fn(X, Y))
  scores = np.concatenate(scores)
  return scores


def compute_unclog_scores(fn, params, state, X, Y, cls_smpl_sz, seed, batch_sz_mlgs):
  n_batches = X.shape[0]
  Xs = np.split(X, n_batches)
  X_mlgs, _ = get_class_balanced_random_subset(X, Y, cls_smpl_sz, seed)
  mlgs = compute_mean_logit_gradients(fn, params, state, X_mlgs, batch_sz_mlgs)
  logit_grads_fn = get_mean_logit_gradients_fn(fn, params, state)
  score_fn = jit(lambda X: jnp.linalg.norm((logit_grads_fn(X) - mlgs).sum(0)))
  scores = []
  for i, X in enumerate(Xs):
    if i % 500 == 0: print(f'images {i} of {n_batches}')
    scores.append(score_fn(X).item())
  scores = np.array(scores)
  return scores

def get_class_balanced_random_subset(X, Y, cls_smpl_sz, seed):
  """get_class_balanced_random_subset: X, Y, cls_smpl_sz, seed -> X, Y
  In:
    X          : nparr(N, img): all images, ASSUME sorted by class
    Y          : nparr(N, C)  : corresponding labels, ASSUME equal number of examples per class
    cls_smpl_sz: int          : number of examples per class in subset
    seed       : int          : random seed
  Out:
    X: nparr(C * cls_smpl_sz, img): subsampled images, cls_smpl_sz examples per class, sorted by class
    Y: nparr(C * cls_smpl_sz, C)  : corresponding labels
  """
   
  n_cls = Y.shape[1]
  X_c, Y_c = np.stack(np.split(X, n_cls)), np.stack(np.split(Y, n_cls))
   
  rng = np.random.RandomState(seed)
  idxs = [rng.choice(X_c.shape[1], cls_smpl_sz, replace=False) for _ in range(n_cls)]
  X = np.concatenate([X_c[c, idxs[c]] for c in range(n_cls)])
  Y = np.concatenate([Y_c[c, idxs[c]] for c in range(n_cls)])
  return X, Y
