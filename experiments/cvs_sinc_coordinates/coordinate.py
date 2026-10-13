"""Raw-Hz vs centered frequency coordinates; same native Sinc synthesis."""
import torch
from torch import nn

class CoordinateSinc(nn.Module):
 def __init__(self,original,mode,scale_hz=25000.):
  super().__init__()
  if type(original).__name__!='SincConv1d' or mode not in ('RAW_HZ','CENTER_25K') or scale_hz!=25000.:raise ValueError('Unregistered Sinc coordinate scheme')
  if original.low_hz_.shape!=(24,1) or original.band_hz_.shape!=(24,1) or original.kernel_size!=79 or original.sample_rate!=25e6:raise ValueError('Unexpected native identity Sinc')
  self.mode=mode;self.scale_hz=float(scale_hz);self._native_type=type(original)
  for name in ('out_channels','kernel_size','sample_rate','dataset','input_len','pad_crop_mode','min_low_hz','min_band_hz'):setattr(self,name,getattr(original,name))
  low=original.low_hz_.detach().clone();band=original.band_hz_.detach().clone()
  self.low_coordinate=nn.Parameter(low.clone() if mode=='RAW_HZ' else torch.zeros_like(low))
  self.band_coordinate=nn.Parameter(band.clone() if mode=='RAW_HZ' else torch.zeros_like(band))
  self.register_buffer('low_origin_hz',low);self.register_buffer('band_origin_hz',band)
  self.register_buffer('t_',original.t_.detach().clone());self.register_buffer('window_',original.window_.detach().clone())
  self._filter_cache_key=None;self._filter_cache=None;self._coordinate_cache_key=None
 @property
 def low_hz_(self):return self.low_coordinate if self.mode=='RAW_HZ' else self.low_origin_hz+self.scale_hz*self.low_coordinate
 @property
 def band_hz_(self):return self.band_coordinate if self.mode=='RAW_HZ' else self.band_origin_hz+self.scale_hz*self.band_coordinate
 def _filters(self,device,dtype):
  # Computed low_hz_/band_hz_ tensors have version0. Track their actual leaves.
  version=(device.type,device.index,self.low_coordinate.dtype,self.low_coordinate._version,self.band_coordinate._version,self.low_origin_hz._version,self.band_origin_hz._version,self.t_._version,self.window_._version)
  if version!=self._coordinate_cache_key:
   self._filter_cache_key=None;self._filter_cache=None;self._coordinate_cache_key=version
  return self._native_type._filters(self,device,dtype)
 def forward(self,x):return self._native_type.forward(self,x)
 def forward_iq_pair(self,x):return self._native_type.forward_iq_pair(self,x)
