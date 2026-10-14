"""Step1 displacement and step2 skew area on one continuous global clock."""
import torch
def signature(p):
 if p.ndim!=3 or p.shape[-1]!=7 or p.shape[1]<2 or not torch.isfinite(p).all():raise ValueError('Finite7D path required')
 delta=p[:,1:]-p[:,:-1];prefix=delta.cumsum(1)-delta;ij=torch.triu_indices(7,7,offset=1,device=p.device)
 area=.5*(prefix[:,:,ij[0]]*delta[:,:,ij[1]]-delta[:,:,ij[0]]*prefix[:,:,ij[1]]).sum(1)
 return torch.cat((delta.sum(1),area),-1)
def compose(a,b):
 if a.shape!=b.shape or a.ndim!=2 or a.shape[1]!=28 or not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Matched finite28D signatures required')
 ij=torch.triu_indices(7,7,offset=1,device=a.device);da,db=a[:,:7],b[:,:7]
 area=a[:,7:]+b[:,7:]+.5*(da[:,ij[0]]*db[:,ij[1]]-db[:,ij[0]]*da[:,ij[1]])
 return torch.cat((da+db,area),-1)
def fold(parts):
 if len(parts)<2:raise ValueError('Multiple ordered segments required')
 result=parts[0]
 for part in parts[1:]:result=compose(result,part)
 return result
def components(u):
 if u.ndim!=3 or u.shape[1:]!=(32,6) or not torch.isfinite(u).all():raise ValueError('Finite32x6 native path required')
 clock=torch.linspace(0,1,32,dtype=u.dtype,device=u.device)[None,:,None].expand(len(u),-1,-1);p=torch.cat((u,clock),-1)
 quarters=[signature(p[:,j*8:(j+1)*8]) for j in range(4)]
 bridges=[signature(p[:,j*8+7:j*8+9]) for j in range(3)]
 halves=[fold([quarters[0],bridges[0],quarters[1]]),fold([quarters[2],bridges[2],quarters[3]])]
 joined=fold([halves[0],bridges[1],halves[1]]);direct=signature(p)
 no_bridges=fold(quarters);within_only=torch.stack([*quarters,*bridges]).sum(0)
 reset=[]
 localclock=torch.linspace(0,1,8,dtype=u.dtype,device=u.device)[None,:,None].expand(len(u),-1,-1)
 for j in range(4):reset.append(signature(torch.cat((u[:,j*8:(j+1)*8],localclock),-1)))
 reset_joined=fold([reset[0],bridges[0],reset[1],bridges[1],reset[2],bridges[2],reset[3]])
 return dict(quarters=torch.stack(quarters,1),bridges=torch.stack(bridges,1),halves=torch.stack(halves,1),joined=joined,direct=direct,direct_halves=torch.stack([signature(p[:,:16]),signature(p[:,16:])],1),no_bridges=no_bridges,within_only=within_only,reset_clock_joined=reset_joined,legacy_local_reset=torch.stack(reset,1))
def checks():
 torch.manual_seed(51);u=torch.randint(-4,4,(100,32,6)).double()/16;c=components(u)
 err=float((c['joined']-c['direct']).abs().max());halferr=float((c['halves']-c['direct_halves']).abs().max());assert err<1e-12 and halferr<1e-12
 assert torch.allclose(c['quarters'][:,:,6],torch.full((100,4),7/31,dtype=torch.float64),atol=1e-15,rtol=0) and torch.allclose(c['bridges'][:,:,6],torch.full((100,3),1/31,dtype=torch.float64),atol=1e-15,rtol=0)
 assert not torch.allclose(c['no_bridges'],c['direct']) and not torch.allclose(c['reset_clock_joined'],c['direct'])
 moved=u.clone();moved[:,8:16,0]+=.125;changed=components(moved)
 assert torch.equal(c['legacy_local_reset'],changed['legacy_local_reset']) and not torch.equal(c['direct'],changed['direct'])
 negatives=0
 for bad in (u[:,:31],u[:,:,:5],u*float('nan')):
  try:components(bad)
  except ValueError:negatives+=1
  else:raise AssertionError('Invalid path accepted')
 f32=components(u.float());err32=float((f32['joined']-f32['direct']).abs().max());assert err32<2e-5
 return dict(status='PASS',synthetic_only=True,joined_direct_error64=err,half_direct_error64=halferr,joined_direct_error32=err32,three_bridges_included=True,one_global_time_coordinate=True,independent_local_signature_cannot_identify_global_signature=True,missing_bridges_and_reset_clock_are_not_equivalent=True,protocol_negatives=negatives,source_or_target_IQ_read=False)
if __name__=='__main__':
 import json
 print(json.dumps(checks()))
