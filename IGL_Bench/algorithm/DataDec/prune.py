import json
import numpy as np
import torch
from collections import OrderedDict

from pdb import set_trace


class Mask(object):
    def __init__(self, model, no_reset=False):
        super(Mask, self).__init__()
        self.model = model
        if not no_reset:
            self.reset()

    @property
    def sparsity(self):
        """Return the percent of weights that have been pruned as a decimal."""
        prunableTensors = []
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                prunableTensors.append(module.prune_mask.detach())

        unpruned = torch.sum(torch.tensor([torch.sum(v) for v in prunableTensors]))
        total = torch.sum(torch.tensor([torch.sum(torch.ones_like(v)) for v in prunableTensors]))
        return 1 - unpruned.float() / total.float()

    @property
    def density(self):
        return 1 - self.sparsity

    def magnitudePruning(self, magnitudePruneFraction, randomPruneFraction=0):
        weights = []
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                weights.append(module.weight.clone().cpu().detach().numpy())

         
        self.reset()
        prunableTensors = []
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                prunableTensors.append(module.prune_mask.detach())

        number_of_remaining_weights = torch.sum(torch.tensor([torch.sum(v) for v in prunableTensors])).cpu().numpy()
        number_of_weights_to_prune_magnitude = np.ceil(magnitudePruneFraction * number_of_remaining_weights).astype(int)
        number_of_weights_to_prune_random = np.ceil(randomPruneFraction * number_of_remaining_weights).astype(int)
        random_prune_prob = number_of_weights_to_prune_random / (number_of_remaining_weights - number_of_weights_to_prune_magnitude)

         
        weight_vector = np.concatenate([v.flatten() for v in weights])
        threshold = np.sort(np.abs(weight_vector))[min(number_of_weights_to_prune_magnitude, len(weight_vector) - 1)]

         
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                module.prune_mask = (torch.abs(module.weight) >= threshold).float()
                 
                module.prune_mask[torch.rand_like(module.prune_mask) < random_prune_prob] = 0

    def reset(self):
        for name, module in self.model.named_modules():
            if hasattr(module, "prune_mask"):
                module.prune_mask = torch.ones_like(module.weight)


def save_mask(epoch, model, filename):
    pruneMask = OrderedDict()

    for name, module in model.named_modules():
        if hasattr(module, "prune_mask"):
            pruneMask[name] = module.prune_mask.cpu().type(torch.bool)

    torch.save({"epoch": epoch, "pruneMask": pruneMask}, filename)


def load_mask(model, state_dict, device):
     
    for name, module in model.named_modules():
        if hasattr(module, "prune_mask"):
            module.prune_mask.data = state_dict[name].to(device).float()

    return model

