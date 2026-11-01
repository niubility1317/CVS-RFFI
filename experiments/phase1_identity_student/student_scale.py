"""Scale a frozen student correction while retaining the fixed 2% L2 bound."""
import torch

def scale_with_bound(parent,student,scale,epsilon=.02):
    if parent.shape!=student.shape or parent.ndim!=3 or scale<0 or not 0<epsilon<=.1:raise ValueError('Matched IQ, nonnegative scale and valid bound required')
    delta=(student-parent)*scale;norm=delta.flatten(1).norm(dim=1).clamp_min(1e-12);radius=epsilon*parent.flatten(1).norm(dim=1);factor=torch.minimum(torch.ones_like(norm),radius/norm);return parent+delta*factor[:,None,None]
