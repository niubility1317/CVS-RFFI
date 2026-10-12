def snapshot(model):
 return {kind+':'+name:value.detach().clone() for kind,iterator in (('parameter',model.named_parameters()),('buffer',model.named_buffers())) for name,value in iterator}
