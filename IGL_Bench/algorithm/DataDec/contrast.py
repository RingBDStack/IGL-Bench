from GCL.models import DualBranchContrast
from GCL.models.contrast_model import add_extra_mask
from GCL.losses import Loss
import torch
import numpy as np

class DualBranchContrast_diet(DualBranchContrast):
    def __init__(self, loss: Loss, mode: str, intraview_negs: bool = False, use_grad_norm: bool = False, ord: int = 1 ):
        super(DualBranchContrast_diet, self).__init__(loss=loss, mode=mode, intraview_negs=intraview_negs)
        self.use_grad_norm = use_grad_norm
        self.ord = ord


    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=h1, sample=h2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=h2, sample=h1)
        elif self.mode == 'G2G':
            assert g1 is not None and g2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=g2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=g1)
        else:   
            if batch is None or batch.max().item() + 1 <= 1:   
                assert all(v is not None for v in [h1, h2, g1, g2, h3, h4])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, neg_sample=h4)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, neg_sample=h3)
            else:   
                assert all(v is not None for v in [h1, h2, g1, g2, batch])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=batch)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=batch)

        pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        sample1.retain_grad()
        sample2.retain_grad()
        scores1 = get_lord_error_fn(anchor1, sample1, self.ord)
        scores2 = get_lord_error_fn(anchor2, sample2, self.ord)
        return (l1 + l2) * 0.5, scores1, scores2, sample1, sample2
    
def get_lord_error_fn(logits, Y, ord):
    errors = torch.nn.functional.softmax(logits, dim=1) - Y
    scores = np.linalg.norm(errors.detach().cpu().numpy(), ord=ord, axis=-1)
    return scores 
