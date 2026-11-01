"""Source-only gated evaluation of fixed labeled-curriculum checkpoints."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from phase1_axial_spectral_mask import AxialSpectralMask
from phase1_blind_gate_model import BlindGate
from phase1_blind_gate_source import source_contract,epoch_data,new_splits,candidate_eval,choose
from experiments.pamf_phase1_flow import experiment as core

GATE_RUN='20261102-phase1-blind-gate-s2026092701-r01';GATE_COMMIT='c905f3ab3d7f0d121e2c44c3b0bfb26188fa5825'
PARENT='20261102-phase1-direct-curriculum-s2026092701-r01';PARENT_COMMIT='d850c782a2ca1982688b4ec07fa52a0353e864ad';EPOCHS=(1,2,4)

def fixed_gate(root,device):
    data=torch.load(root/'runs'/GATE_RUN/'source/selected.pt',map_location='cpu',weights_only=False)
    if data['commit']!=GATE_COMMIT or data['selected']['kind']!='gated_direct_iq':raise ValueError('Exact fixed source-selected blind gate required')
    gate=BlindGate().to(device);gate.load_state_dict(data['gate'],strict=True)
    return gate.eval().requires_grad_(False)

def fixed_model(root,epoch,device):
    done=core.read(root/'runs'/PARENT/'completion.json')
    if done['status']!='SOURCE_SELECTION_RETAINS_RAW' or done['target_predictions'] is not False:raise ValueError('Source-only curriculum parent required')
    state=torch.load(root/'runs'/PARENT/'source'/('checkpoint_epoch%02d.pt'%epoch),map_location='cpu',weights_only=False)
    if state['commit']!=PARENT_COMMIT or state['epoch']!=epoch:raise ValueError('Exact fixed curriculum checkpoint required')
    model=AxialSpectralMask().to(device);model.load_state_dict(state['model'],strict=True)
    return model.eval().requires_grad_(False),state['model']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--device',default='cuda:0');ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Immutable output root exists')
    args.out.mkdir(parents=True);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache,clean,rx,training,prior_valid,prior_hold=source_contract(args.root);validation,heldout=new_splits(cache,rx,training,prior_valid,prior_hold)
    received,scene=epoch_data(cache,19);gate=fixed_gate(args.root,args.device);teacher,_=core.teacher(args.root,args.device)
    if args.smoke:
        model,_=fixed_model(args.root,1,args.device);x=core.tensor(received[training[:16]],args.device)
        with torch.no_grad():
            active=gate(x).sigmoid()>.8;y=x+active[:,None,None]*.25*(model(x)-x);logits=teacher.classify_features(teacher.features(y))
        if not torch.isfinite(y).all() or not torch.isfinite(logits).all() or not torch.equal(y[~active],x[~active]):raise ValueError('Fixed source inference smoke failed')
        core.write(args.out/'result.json',dict(status='PASS',count=len(x),source_only=True,F_frozen=True,DURA_used=False,TX_labels_read=0,V_reads=0,test_reads=0))
        print('CURRICULUM_GATE_SOURCE_SMOKE PASS',flush=True);return
    results={};states={}
    for epoch in EPOCHS:
        model,state=fixed_model(args.root,epoch,args.device);states[epoch]=state
        results[epoch]=candidate_eval(gate,teacher,model,clean,received,scene,rx,validation,heldout,args.device)
        print('CURRICULUM_GATE_CANDIDATES_EVALUATED',epoch,flush=True)
    selected=choose(results)
    if selected['kind']!='raw':
        with (args.out/'selected.pt').open('xb') as f:torch.save(dict(commit=args.commit,selected=selected,model=states[selected['epoch']],gate=gate.state_dict()),f)
    result=dict(selected=selected,validation=results,train_ids=0,validation_ids=len(validation),heldout_ids=len(heldout),prior_sample_ids_excluded=4096,source_only=True,DURA_used=False,F_frozen=True,TX_labels_read=0,U_s_labels_read=0,V_reads=0,test_reads=0,source_cache_epoch=19,source_caveat='physical IDs previously used for semantic-gate validation with cache epoch20; this is a new view, not independent identities')
    with (args.out/'source_selection.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('CURRICULUM_GATE_SELECTION_FIXED',json.dumps(selected),flush=True)
if __name__=='__main__':main()
