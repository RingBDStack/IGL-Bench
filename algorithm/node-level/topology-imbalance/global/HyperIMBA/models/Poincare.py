import time
import networkx as nx
import tqdm
import numpy as np
from multiprocessing import Lock, Manager, Pool
from numpy.linalg import norm

def norm(x, axis=None):
    return np.linalg.norm(x, axis=axis)

def poincare_dist(u, v, eps=1e-5):
    d = 1 + 2 * norm(u-v)**2 / ((1 - norm(u)**2) * (1 - norm(v)**2) + eps)
    return np.arccosh(d)

class PoincareModel():
    
    def __init__(self, relations, node_weights, node_labels, n_components=2, eta=0.01, n_negative=10,
                 eps=1e-5, burn_in=10, burn_in_eta=0.01, init_lower=-0.001,
                 init_upper=0.001, dtype=np.float64, seed=0, name="", device='cuda', batch_size=None):
        self.relations = relations
        self.n_components = n_components
        self.eta = eta  # Learning rate for training
        self.burn_in_eta = burn_in_eta  # Learning rate for burn-in
        self.n_negative = n_negative
        self.eps = eps
        self.burn_in = burn_in
        self.dtype = dtype
        self.init_lower = init_lower
        self.init_upper = init_upper
        self.node_weights = node_weights
        self.node_labels = node_labels
        self.network = nx.Graph()
        self.name = name
        self.device = device
        self.batch_size = batch_size
        self.manager = Manager()
        self.lock = self.manager.Lock()
       
    def init_embeddings(self):
        unique_nodes = np.unique([item for sublist in self.relations for item in sublist])
        theta_init = np.random.uniform(self.init_lower, self.init_upper, 
                                       size=(len(unique_nodes), self.n_components))
        embedding_dict = dict(zip(unique_nodes, theta_init))
        self.nodes = unique_nodes
        self.embeddings = theta_init
        self.emb_dict = embedding_dict

        
    def negative_sample(self, u):
        positives = [x[1] for x in self.relations if x[0] == u]
        negatives = np.array([x for x in self.nodes if x not in positives])
        random_ix = np.random.permutation(len(negatives))[:self.n_negative]
        neg_samples = [[u, x] for x in negatives[random_ix]]
        neg_samples.append([u,u])
        return neg_samples
    
    def partial_d(self, theta, x):
        alpha = 1 - norm(theta)**2
        beta = 1 - norm(x)**2
        gamma = 1 + 2/(alpha*beta + self.eps) * norm(theta-x)**2
        lhs = 4 / (beta*np.sqrt(gamma**2 - 1) + self.eps)
        rhs = 1/(alpha**2 + self.eps) * (norm(x)**2 - 2*np.inner(theta,x) + 1) * theta - x/(alpha + self.eps)
        return lhs*rhs
        
    def proj(self, theta):
        if norm(theta) >= 1:
            theta = theta/norm(theta) - self.eps
        return theta
    
    def update(self, u, grad):
        with self.lock:  # 使用锁来同步更新操作
            theta = self.emb_dict[u]
            step = 1/4 * self.eta * (1 - norm(theta)**2)**2 * grad
            self.emb_dict[u] = self.proj(theta - step)
    
    def train(self, num_epochs=10, edge_index=None):
        node_rank = np.array([1 / self.node_labels[v] for v in self.node_labels])

        if edge_index is not None:
            self.relations = edge_index

        for _ in range(num_epochs):
            losses = 0

            for relation in tqdm.tqdm(self.relations):
                u, v = relation[0], relation[1]
                if u == v:
                    continue

                theta, x = self.emb_dict[u], self.emb_dict[v]
                neg_relations = [x[1] for x in self.negative_sample(u)]
                neg_embed = np.array([self.emb_dict[x] for x in neg_relations])

                # Vectorized computation for partial derivatives
                rank_comparison = node_rank[u] > node_rank[v]
                dd_theta = self.partial_d(theta, x) * rank_comparison
                dd_x = self.partial_d(x, theta) * (~rank_comparison)

                if np.isnan(dd_theta).any() or np.isinf(dd_theta).any() or np.isnan(dd_x).any() or np.isinf(dd_x).any():
                    return

                # Loss gradients
                grad_theta = -dd_theta
                grad_x = -dd_x

                self.update(u, grad_theta)
                self.update(v, grad_x)

                # Vectorized gradient computation for negative samples
                neg_dists = np.array([np.exp(-poincare_dist(theta, self.emb_dict[vprime])) for vprime in neg_relations])
                Z = np.sum(neg_dists)
                losses_for_neg = neg_dists / Z

                for idx, vprime in enumerate(neg_relations):
                    if node_rank[u] < node_rank[vprime]:
                        dd_u = self.partial_d(theta, self.emb_dict[vprime])
                        grad_u = dd_u * (-losses_for_neg[idx])
                        self.update(u, grad_u)
                    else:
                        dd_vprime = self.partial_d(self.emb_dict[vprime], theta)
                        grad_vprime = dd_vprime * (-losses_for_neg[idx])
                        self.update(vprime, grad_vprime)

                    losses += losses_for_neg[idx] * poincare_dist(theta, self.emb_dict[vprime])

                pos_loss = np.exp(-poincare_dist(theta, x))
                losses += pos_loss
'''    
    def train(self, num_epochs=10,edge_index=None):
        node_rank = {}
        for v in self.node_labels:
            node_rank[v] = 1/self.node_labels[v]
        if edge_index is not None:
            self.relations = edge_index
        for i in range(num_epochs):
            losses=0
            start = time.time()
            
            for relation in tqdm.tqdm(self.relations):
                u, v = relation[0], relation[1]
                if u == v:
                    continue
                # embedding vectors (theta, x) for relation (u, v)
                theta, x = self.emb_dict[u], self.emb_dict[v]
                # embedding vectors v' in sample negative relations (u, v')
                neg_relations = [x[1] for x in self.negative_sample(u)]
                neg_embed = np.array([self.emb_dict[x] for x in neg_relations])
                # find partial derivatives of poincare distance
                dd_theta = np.zeros(self.n_components)
                dd_x = np.zeros(self.n_components)
                if node_rank[u] > node_rank[v]:
                    dd_theta = self.partial_d(theta, x) 
                else:
                    dd_x = self.partial_d(x, theta)

                if np.isnan(dd_theta.any()) or np.isinf(dd_theta.any()) or np.isnan(dd_x.any()) or np.isinf(dd_x.any()):
                    return
                # find partial derivatives of loss function
                dloss_theta = -1
                dloss_x = -1
                if node_rank[u] < node_rank[v]:
                    grad_theta = dloss_theta * dd_theta
                    self.update(u, grad_theta)
                else:
                    grad_x = dloss_x * dd_x
                    self.update(v, grad_x)
                # find gradients for negative samples
                neg_loss = 0
                neg_exp_dist = np.array([np.exp(-poincare_dist(theta, v_prime)) for v_prime in neg_embed])
                Z = neg_exp_dist.sum(axis=0)
                for vprime in neg_relations:
                    dloss_u = np.zeros(self.n_components)
                    if node_rank[u] < node_rank[vprime]:
                        dd_u = self.partial_d(theta, self.emb_dict[vprime])
                        dloss_u = -np.exp(-poincare_dist(theta, self.emb_dict[vprime])) / Z
                        grad_u = dd_u * dloss_u
                        self.update(u, grad_u)
                        loss = dloss_u
                    else:
                        dd_vprime = self.partial_d(self.emb_dict[vprime], theta)
                        dloss_vprime = -np.exp(-poincare_dist(self.emb_dict[vprime], theta)) / Z
                        grad_vprime = dd_vprime * dloss_vprime
                        self.update(vprime, grad_vprime)
                        loss = dloss_vprime
                    neg_loss += loss
                pos_loss = np.exp(-poincare_dist(theta, x)) 
                losses = -(losses)+(pos_loss+neg_loss)
'''           



