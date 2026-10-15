from collections import Counter,defaultdict
from types import SimpleNamespace
import random,torch
from . import design as d
def select(index,arm):
 if arm not in d.ARMS or len(index)!=6300:raise ValueError('Original legal L role and registered exposure required')
 cells=defaultdict(list)
 for i,v in enumerate(index):cells[int(v.tx_i),int(v.rx_i),int(v.day_i)].append(i)
 if set(cells)!={(t,r,day) for t in range(6) for r in d.RX for day in (1,2,3)} or any(len(v)!=70 for v in cells.values()):raise ValueError('Original90 L cells of70 packets required')
 chosen=[]
 for group,key in enumerate(sorted(cells)):
  physical=lambda i:tuple(int(getattr(index[i],k)) for k in ('eq_i','rx_i','day_i','sig_i'))
  if len({physical(i) for i in cells[key]})!=70:raise ValueError('Duplicate source L physical records')
  values=sorted(cells[key],key=physical)
  if arm=='SAMPLE_80':random.Random(d.SUBSET_SEED+group).shuffle(values);chosen+=values[:56]
  elif key[1]!=d.HELD:chosen+=values
 chosen=sorted(chosen);counts=Counter((int(index[i].tx_i),int(index[i].day_i)) for i in chosen)
 if len(chosen)!=5040 or set(counts.values())!={280} or set(counts)!={(t,day) for t in range(6) for day in (1,2,3)}:raise ValueError('Matched L exposure budget differs')
 return chosen
class EmptyU:
 index=();selected=();hide=True;split_source='ssdg_unlabeled_tx_hidden_disabled'
 def __len__(self):return 0
 def __getitem__(self,i):raise AssertionError('Disabled U dataset read')
class NoU:
 dataset=EmptyU();sampler=None
 def __len__(self):return 222
 def __iter__(self):raise AssertionError('Pure CE teacher attempted U iteration')
def apply(native,ctx,arm):
 from cvsrffi.game_tracking.data import opaque_id
 if ctx['named_test_loaders'] or ctx.get('balanced_train_sampler') is not None:raise ValueError('Only original source carrier permitted')
 L=ctx['train_loader'].dataset;V=ctx['val_loader'].dataset
 chosen=select(L.index,arm)
 train_selected=[int(L.selected[i]) for i in chosen]
 if len(set(train_selected))!=5040:raise ValueError('Duplicate train physical base indices')
 fit=[int(V.selected[i]) for i,v in enumerate(V.index) if int(v.rx_i) in d.FIT]
 held=[i for i,v in enumerate(V.index) if int(v.rx_i)==d.HELD]
 if len(V)!=27000 or len(fit)!=21600 or len(held)!=5400:raise ValueError('Original source V split differs')
 # Keep only an unevaluated dataset reference until the final student freezes.
 held_dataset=torch.utils.data.Subset(V,held)
 for name,source,selected,shuffle,drop in (
  ('train_loader',L,train_selected,True,True),('probe_train_loader',L,train_selected,False,False),
  ('val_loader',V,fit,False,False),('source_calibration_loader',V,fit,False,False)):
  loader=ctx[name];ds=native.WiSigSubsetDataset(source.base,selected,split_source=source.split_source,transform=None)
  ctx[name]=native.make_loader(ds,loader.batch_size,shuffle,0,torch.device('cuda:0'),drop,2)
 ctx['unlabeled_loader']=NoU()
 ctx['split_info']=dict(ctx['split_info'],labeled_size=5040,unlabeled_size=0,source_val_size=21600,source_calibration_size=21600,source_selection_size=21600,
  teacher_exposure=dict(arm=arm,train_L=5040,U_iterator=False,EMA=False,held_RX1_V_final_only=5400,fixed_steps_per_epoch=222))
 allowed={int(L.selected[i]):(int(L.index[i].tx_i),int(L.index[i].rx_i),int(L.index[i].day_i)) for i in chosen}
 if len(allowed)!=5040:raise ValueError('Training physical metadata not one-to-one')
 records=[dict(physical_id=opaque_id(L.index[i]),base_index=int(L.selected[i]),TX=int(L.index[i].tx_i),RX=int(L.index[i].rx_i),day=int(L.index[i].day_i)) for i in chosen]
 return allowed,held_dataset,records
