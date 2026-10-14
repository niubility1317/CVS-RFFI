"""Small independent census and fixed qualification checks; no data access."""
import json
import torch
from .proxy import census,qualification

def check():
    # Six classes, two RXs, known downstream truth; class-set truth exclusion
    # must be reported even when that class is not the teacher's chosen class.
    truth=torch.tensor([0,1,2,3,4,5]);rx=torch.tensor([1,1,1,3,3,3]);day=torch.ones(6,dtype=torch.long)
    candidates=torch.eye(6,dtype=torch.bool)
    candidates[1,1]=False;candidates[1,0]=True
    candidates[2,:]=True;candidates[3,4]=True
    accepted=torch.tensor([True,False,False,False,True,False])
    masses={'S0':torch.log(torch.tensor([.99,.5,1.,.8,.99,.99])),
            'S1':torch.log(torch.tensor([.99,.6,1.,.7,.99,.8]))}
    r=census(candidates,accepted,masses,truth,rx,day)
    all_r=r['overall']['subsets']['all'];comp=r['overall']['subsets']['complement']
    assert all_r['count']==6 and all_r['true_class_excluded']==1
    assert comp['count']==4 and comp['size_histogram']=={'1':2,'2':1,'3':0,'4':0,'5':0,'6':1}
    assert comp['S0']['active']==2 and comp['S1']['active']==3
    assert comp['S0']['active_multiclass']==1 and comp['S0']['active_true_excluded']==1
    assert sum(z['count'] for k,z in r.items() if k.startswith('RX/'))==6
    assert sum(z['count'] for k,z in r.items() if k.startswith('TX/'))==6
    assert not qualification(r)['passed']
    # Truth changing only changes diagnostic attribution, never input masks.
    before=candidates.clone();census(candidates,accepted,masses,truth.roll(1),rx,day)
    assert torch.equal(before,candidates)
    return dict(status='VERIFIED',full_subset_counts=True,false_exclusion_visible=True,
                full_class_sets_inactive=True,strict_fixed_source_rule=True,source_or_target_read=False)

if __name__=='__main__':print(json.dumps(check()))
