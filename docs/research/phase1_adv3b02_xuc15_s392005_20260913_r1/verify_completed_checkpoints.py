"""Read-only validation of this run's E200 artifacts; no model training or truth."""
import json,time,sys
from pathlib import Path
import torch
torch.set_num_threads(2)
root=Path('/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_xuc15_s392005_20260913_r1')
state=json.loads((root/'pipeline_state.json').read_text())
sys.path.insert(0,str(Path(state['release'])/'code'))
contract=json.loads((root/'source_contract.json').read_text())
results={}
for rid,entry in state['rows'].items():
    if entry['status']!='TRAINING_COMPLETE':continue
    path=root/rid/'final_ssdg.pth'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    args=saved['args']
    assert saved['epoch']==200 and args['epochs']==200 and args['seed']==392005
    assert args['from_scratch'] and not args.get('baseline_ckpt') and not args.get('teacher_ckpt') and not args.get('game_resume')
    assert not saved.get('target_contact',False)
    if saved.get('source_info'):assert saved['source_info']['role_ids']==contract['role_ids']
    assert json.loads((root/rid/'source_contract.json').read_text())['role_ids']==contract['role_ids']
    assert args['num_classes']==6 and not args['amp']
    assert all(bool(torch.isfinite(value).all()) for value in saved['model'].values() if value.is_floating_point())
    results[rid]={'status':'VERIFIED_E200','checkpoint_bytes':path.stat().st_size,'step':saved.get('step'),
                  'seed':args['seed'],'source_roles':'EXACT_MATCH','scratch_only':True,'target_contact':False,'model_state_finite':True}
    del saved
print(json.dumps({'time':time.time(),'rows':results,'count':len(results)},indent=2))
