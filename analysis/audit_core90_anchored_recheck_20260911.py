"""Read-only model audit with synthetic boundary inputs; no experiment launch."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json,sys
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.anchored_fusion import realize_actions
from cvsrffi.anchored_source import _verify_stage_input
from cvsrffi.anchored_cache import CacheIdentity
from cvsrffi.anchored_reporting import assess_source_promotion

torch.set_num_threads(2);torch.manual_seed(602)
n=10000
s0=torch.randn(n,6)*.4;s0[:,0]+=2;s0[:,1]+=.4
sg=torch.randn(n,6)*.3;sg[:,2:]-=3
p=s0.double().softmax(-1);delta=p[:,0]-p[:,1]
e=sg.double().exp();other=e[:,2:].sum(-1)
sg[:,1]=(((1+delta)*e[:,0]+delta*other)/(1-delta)).log().float()
training=realize_actions(s0.log_softmax(-1),sg.log_softmax(-1),protection_threshold=1.)
deployment=realize_actions(s0.double().log_softmax(-1),sg.double().log_softmax(-1),protection_threshold=1.)
different=(training['log_probabilities'][:,2].argmax(-1)!=deployment['log_probabilities'][:,2].argmax(-1))
ix=different.nonzero().flatten()
result={'boundary_rows':n,'training_deployment_argmax_mismatches':len(ix)}
if len(ix):
    i=int(ix[0]);result['example']=dict(s0=s0[i].tolist(),sG=sg[i].tolist(),
        train_prediction=int(training['log_probabilities'][i,2].argmax()),deploy_prediction=int(deployment['log_probabilities'][i,2].argmax()),
        train_prob=training['log_probabilities'][i,2].exp().tolist(),deploy_prob=deployment['log_probabilities'][i,2].exp().tolist())
config=json.loads((ROOT/'code/configs/core90_anchored_geometry_v1.json').read_text(encoding='utf-8'))
identity=CacheIdentity('a'*64,'b'*64,'received_iq_v1',{'version':'paired_leo_v1','seed':392005},392005,'V')
manifest=dict(stage='oof',candidate='A4',head_seed=392005,config=config)
with patch.object(Path,'read_text',return_value=json.dumps(manifest)):
    try:_verify_stage_input(SimpleNamespace(input='oof/all_source/expert.pt',stage='calibrate',candidate='A4',seed=392005),config,identity)
    except ValueError as error:result['oof_to_calibrate_error']=str(error)
records=[dict(candidate='A6',group='all',value='all',weighted_net=.01),dict(candidate='A6',group='view',value='clean',weighted_net=0.)]
records += [dict(candidate='A6',group='RX',value=str(rx),weighted_net=.01) for rx in (1,3,4,6,8)]
records += [dict(candidate='A6',group='TX',value='0',weighted_net=.01)]
result['singleton_seed_assessment']=assess_source_promotion({392005:records})
result['missing_five_TX_assessment']=assess_source_promotion({seed:records for seed in (392005,392006,392007)})
path=Path(__file__).with_suffix('.json')
with path.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
