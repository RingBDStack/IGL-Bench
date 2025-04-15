import torch
import dgl
import torch.nn.functional as F
import tensorflow as tf


def inject_nodes(batched_masked_graphs, generated_neighbors, masked_offset, device, mask=None):
    assert len(masked_offset) == len(generated_neighbors)
    batched_masked_graphs_ = dgl.add_nodes(batched_masked_graphs, len(masked_offset), {'feat':generated_neighbors})
    temp = torch.arange(batched_masked_graphs_.number_of_nodes() - len(masked_offset), batched_masked_graphs_.number_of_nodes()).to(device)
    masked_offset = masked_offset.to(device)
    # src = torch.cat([temp, masked_offset])
    # dst = torch.cat([masked_offset, temp])
    src = temp[mask] if mask != None else temp
    dst = masked_offset[mask] if mask != None else masked_offset
    batched_masked_graphs_.add_edges(src, dst)
    return batched_masked_graphs_


def kl_div(x, y):
    x = F.log_softmax(x, dim=1)
    y = F.softmax(y, dim=1)
    return F.kl_div(x, y, reduction='batchmean')


def construct_placeholder(num_nodes, fea_size, num_classes):
    with tf.name_scope('input'):
        placeholders = {
            'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
            'features': tf.compat.v1.placeholder(tf.float32, shape=(num_nodes, fea_size), name='features'),
            'dropout': tf.compat.v1.placeholder_with_default(0., shape=(), name='dropout'),
            'masks': tf.compat.v1.placeholder(dtype=tf.int32, shape=(num_nodes,), name='masks'),
        }
        return placeholders
