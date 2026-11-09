"""Source-only frozen-checkpoint numerical and reader-decision comparison.

This is a 112-record no-query smoke, not a claim about all target predictions
or about trajectories after the independent full trainings.
"""
import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_innovation_probe.source_probe import source_inputs
from experiments.phase1_blind_unrolled.model import BlindUnrolledRecovery
from experiments.phase1_blind_unrolled.train import SCHEMA as NATIVE_SCHEMA
from experiments.phase1_blind_polyphase.model import BlindPolyphaseRecovery
from experiments.phase1_blind_polyphase.train import SCHEMA as FAST_SCHEMA

NATIVE='20261109-phase1-blind-unrolled-source-s2026092701-r01'
NATIVE_COMMIT='9a27afa61f60a20c66bb40b441402698baf29d82'
FAST='20261109-phase1-blind-polyphase-source-s2026092701-r01'
FAST_COMMIT='cd0687eed0fa69db343bb6000b486143698fe797'


def read_smoke(root, name, schema, commit):
    folder=root/'runs'/(name+'-training-smoke')
    summary=core.read(folder/'smoke.json')
    state=torch.load(folder/'smoke.pt',map_location='cpu',weights_only=True)
    expected=dict(schema=schema,commit=commit,epochs=1,fit_ids=32768,actual_fit_ids=64,
        actual_source_view_count=192,views_per_record=3,width=192,steps=4,taps=33,physics_weight=.02,
        inference_received_IQ_only=True,DURA_training_reads=0,TX_labels_read=0,F_weights_read=0,
        F_outputs_read=0,V_reads=0,target_reads=0)
    if summary.get('status')!='SOURCE_TRAINING_SMOKE_PASS' or any(
        obj.get(k)!=v for obj in (summary,state) for k,v in expected.items()):
        raise ValueError('Matching source-only frozen smoke checkpoint required')
    return state['model']


def positions_for_source(rx, scene):
    positions=[]
    for receivers in ((4,6),(8,)):
        for view in core.VIEWS:
            match=np.flatnonzero(np.isin(rx,receivers)&(scene==view))
            if len(match)<8:raise ValueError('Eight source observations per cell required')
            positions.extend(match[:8].tolist())
    return np.asarray(positions,dtype=np.int64)


def compare_pair(a, b, logits_a, logits_b):
    if a.shape!=b.shape or logits_a.shape!=logits_b.shape or logits_a.shape!=(len(a),6):
        raise ValueError('Matching IQ and six-class predictions required')
    if not all(torch.isfinite(x).all() for x in (a,b,logits_a,logits_b)):
        raise FloatingPointError('Finite source predictions required')
    return dict(iq_max_abs=float((a-b).abs().max()),
        logits_max_abs=float((logits_a-logits_b).abs().max()),
        decision_mismatches=int((logits_a.argmax(1)!=logits_b.argmax(1)).sum()))


@torch.no_grad()
def run(root,out,commit):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    if out.exists() or not torch.cuda.is_available():raise ValueError('Unique output and selected GPU required')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    native_state=read_smoke(root,NATIVE,NATIVE_SCHEMA,NATIVE_COMMIT)
    fast_state=read_smoke(root,FAST,FAST_SCHEMA,FAST_COMMIT)
    native=BlindUnrolledRecovery().to('cuda:0')
    native.load_state_dict(native_state,strict=True)
    native.eval().requires_grad_(False)
    same=BlindPolyphaseRecovery().to('cuda:0')
    same.load_state_dict(native_state,strict=True)
    same.eval().requires_grad_(False)
    trained=BlindPolyphaseRecovery().to('cuda:0')
    trained.load_state_dict(fast_state,strict=True)
    trained.eval().requires_grad_(False)
    reader,_=core.teacher(root,'cpu')
    reader.eval().requires_grad_(False)
    ids,rx,scene,_,received,_=source_inputs(root)
    positions=positions_for_source(rx,scene)
    print('BLIND_EQUIVALENCE_SOURCE_INPUT_READY',len(positions),flush=True)
    out.mkdir(parents=True)
    batches={key:[] for key in ('native_iq','same_iq','trained_iq','native_logits','same_logits','trained_logits')}
    elapsed={'native':0.,'same_checkpoint_polyphase':0.,'independent_smoke_polyphase':0.}
    for start in range(0,len(positions),8):
        selected=ids[positions[start:start+8]]
        x=core.tensor(received[selected],'cuda:0')
        outputs=[]
        for name,model in (('native',native),('same_checkpoint_polyphase',same),('independent_smoke_polyphase',trained)):
            torch.cuda.synchronize();t=time.monotonic();value=model(x);torch.cuda.synchronize()
            elapsed[name]+=time.monotonic()-t;outputs.append(value.cpu())
        for prefix,value in zip(('native','same','trained'),outputs):
            logits=reader.classify_features(reader.features(value))
            batches[prefix+'_iq'].append(value)
            batches[prefix+'_logits'].append(logits)
    data={key:torch.cat(parts,0) for key,parts in batches.items()}
    fixed=compare_pair(data['native_iq'],data['same_iq'],data['native_logits'],data['same_logits'])
    trained_pair=compare_pair(data['native_iq'],data['trained_iq'],data['native_logits'],data['trained_logits'])
    core.write(out/'result.json',dict(status='SOURCE_NO_QUERY_COMPARISON_COMPLETE',commit=commit,
        native_commit=NATIVE_COMMIT,fast_commit=FAST_COMMIT,records=len(positions),
        fixed_same_checkpoint=fixed,independent_six_step_checkpoints=trained_pair,
        recovery_forward_seconds=elapsed,F_run=core.F_RUN,F_row=core.F_ROW,
        F_weights_read=1,F_outputs_read=1,TX_truth_reads=0,U_s_truth_reads=0,V_reads=0,
        target_IQ_reads=0,target_truth_reads=0,model_updates=0,
        caveat='112 source records and two smoke checkpoints only; no claim of full-target decision equivalence or full-training trajectory identity.'))
    print('BLIND_EQUIVALENCE_SOURCE_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--commit',required=True)
    a=p.parse_args();run(a.root,a.out,a.commit)

if __name__=='__main__':main()
