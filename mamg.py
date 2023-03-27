import torch 
from functools import reduce
from operator import mul
import random
from copy import deepcopy



def flatten_and_get_dims(grad_task):
    """
    :param grad_task:  the gradient of single task
    :return: flatten gradient and its shape
    """
    grad_values = []
    grad_dims = []
    for grad in grad_task:
        grad_dims.append(tuple(grad.shape))
        grad_values.append(torch.flatten(grad))
    return torch.cat(grad_values), grad_dims


def unflatten_grad(flatten_grad, grad_dim):
    """
    :param flatten_grad: the flatten gradient of single task
    :param grad_dim:  the shape of flatten gradient
    :return: the unflatten gradient (with the same shape of origin gradient)
    """
    chunk_sizes = [reduce(mul, dims, 1) for dims in grad_dim]
    grad_chunk = torch.split(flatten_grad, split_size_or_sections=chunk_sizes)
    unflatten_grads = []
    for index, grad in enumerate(grad_chunk):  # TODO(speedup): convert to map since they are faster
        grad = torch.reshape(grad, grad_dim[index])
        unflatten_grads.append(grad)

    return unflatten_grads

def solve_conflicting(grad_tasks):
    mg_grad = deepcopy(grad_tasks)
    for g_i in mg_grad:
        random.shuffle(grad_tasks)
        for g_j in grad_tasks:
            if g_j is not g_i:
                g_i_g_j = torch.dot(g_i, g_j)
                g_ij = g_i - g_j
                g_ji = g_j - g_i
                plane_length = torch.norm(g_ij)
                cos_i = torch.dot(g_i, g_ij) / (torch.norm(g_i) * plane_length)
                cos_j = torch.dot(g_j, g_ji) / (torch.norm(g_j) * plane_length)
                delt_grad = torch.norm(g_i) * cos_i - torch.norm(g_j) * cos_j
                if g_i_g_j < 0:
                    # exists conflict gradient and modulate the gradient
                    g_i -= min(abs(delt_grad),0) * (g_ji / plane_length)
    return mg_grad


def get_mod_grads(original_grads):
    flatten_grads, flatten_grad_dims = [], []
    unflatten_grad_tasks = []
    for task, grad in original_grads.items():
        grad, dim = flatten_and_get_dims(grad)
        flatten_grads.append(grad)
        flatten_grad_dims.append(dim)
    modulated_grad_tasks = solve_conflicting(flatten_grads)
    for g, d in zip(modulated_grad_tasks, flatten_grad_dims):
        unflatten_grad_task = unflatten_grad(g, d)
        unflatten_grad_tasks.append(unflatten_grad_task)

    return unflatten_grad_tasks

