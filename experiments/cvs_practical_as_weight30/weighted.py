"""Existing RDrop terms, with fixed group means; no new network calls."""

def combine(ordinary,selected,ordinary_weight=.125,selected_weight=.375):
 if (ordinary_weight,selected_weight) not in ((.125,.375),(.25,.25)):raise ValueError('Only fixed candidate or mathematical equal-weight control')
 if len(ordinary)!=3 or len(selected)!=3:raise ValueError('RD/CE/KL groups required')
 return tuple(ordinary_weight*a+selected_weight*b for a,b in zip(ordinary,selected))

def grouped(first,second,y,mask,terms,ordinary_weight=.125,selected_weight=.375):
 import torch
 if first.shape!=(128,6) or second.shape!=first.shape or y.shape!=(128,) or y.dtype!=torch.long or not bool(((y>=0)&(y<6)).all()) or mask.shape!=(128,) or mask.dtype!=torch.bool or int(mask.sum())!=64:raise ValueError('Only existing mixed128 with64+64 true labels')
 ordinary=terms(first[~mask],second[~mask],y[~mask],1.)[:3]
 selected=terms(first[mask],second[mask],y[mask],1.)[:3]
 weighted=combine(ordinary,selected,ordinary_weight,selected_weight)
 return weighted,ordinary,selected
