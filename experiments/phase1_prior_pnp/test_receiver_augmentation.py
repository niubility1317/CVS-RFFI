import torch
from receiver_augmentation import receiver_filter

def test_zero_filter_is_identity_and_boundary_is_not_circular():
 x=torch.zeros(1,2,5);x[0,0,0]=1;zero=torch.zeros(1,4);assert torch.equal(receiver_filter(x,zero),x);coeff=torch.tensor([[.1,0.,0.,0.]]);y=receiver_filter(x,coeff);assert y[0,0,-1]==0 and y[0,0,1]>0

def test_filter_is_complex_linear_and_record_specific():
 torch.manual_seed(72);x=torch.randn(3,2,16,dtype=torch.float64);y=torch.randn(3,2,16,dtype=torch.float64);c=torch.tensor([[.1,-.03,.02,.04],[0.,0.,0.,0.],[-.06,.03,0.,.07]],dtype=x.dtype);torch.testing.assert_close(receiver_filter(2*x-3*y,c),2*receiver_filter(x,c)-3*receiver_filter(y,c));rot=torch.stack((-x[:,1],x[:,0]),1);z=receiver_filter(x,c);torch.testing.assert_close(receiver_filter(rot,c),torch.stack((-z[:,1],z[:,0]),1));assert torch.equal(receiver_filter(x,c)[1],x[1])
