"""Frozen source-IQ confidence-gated DURA prediction and truth-last score."""
import argparse
from functools import partial
from pathlib import Path
import json
import numpy as np
import torch

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_orthogonal import ground
from experiments.phase1_dura_orthogonal.signal_confidence_gate import SignalConfidenceGate
from experiments.phase1_dura_orthogonal.signal_balanced_mix_train import PARENT, PARENT_COMMIT, EPOCHS, SCHEMA, OOF_SIZE
from experiments.phase1_dura_orthogonal.train_source import extract_context


def validate_gate_fit(fit, parent_fit):
    fit=np.asarray(fit)
    if len(fit)!=OOF_SIZE or len(np.unique(fit))!=len(fit) or np.intersect1d(fit,np.asarray(parent_fit)).size:
        raise ValueError('Confidence gate fit must be disjoint from fixed parent fit')


def load_gate(root, source, commit):
    folder=root/'runs'/source
    summary=json.loads((folder/'source_summary.json').read_text(encoding='utf8'))
    state=torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
    expected=dict(schema=SCHEMA,commit=commit,parent=PARENT,parent_commit=PARENT_COMMIT, parent_fit_disjoint=True,
                  epochs=EPOCHS,TX_labels_read=0,F_weights_read=0,F_outputs_read=0,V_reads=0,target_reads=0,DURA_frozen=True)
    if summary.get('status') != 'SOURCE_TRAINED' or any(summary.get(k)!=v or state.get(k)!=v for k,v in expected.items()):
        raise ValueError('Matching frozen source-only confidence gate required')
    parent=torch.load(root/'runs'/PARENT/'final.pt',map_location='cpu',weights_only=False)
    validate_gate_fit(state['fit_ids'],parent['fit_ids'])
    gate=SignalConfidenceGate()
    gate.load_state_dict(state['model'],strict=True)
    return gate.eval().requires_grad_(False)


@torch.no_grad()
def decisions(x, model, dura, reader, *, gate, expected_mode, inference_mode):
    if inference_mode != expected_mode:
        raise ValueError('Matching fixed gate inference mode required')
    anchor,direction,variance=extract_context(dura,x)
    proposal=model(x,anchor,direction,variance)
    restored=gate(x,anchor,direction,proposal,variance)
    outputs=[]
    for signal in (x,anchor,restored):
        logits=reader.classify_features(reader.features(signal))
        if logits.shape!=(len(x),6) or not torch.isfinite(logits).all():
            raise ValueError('Finite frozen reader logits required')
        outputs.append(logits.argmax(1))
    return outputs


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=('smoke','predict','score'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--commit',required=True)
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    mode='signal_balanced_direct_dura_mix_gate_v1:'+args.source
    if args.mode in ('smoke','predict'):
        gate=load_gate(args.root,args.source,args.commit)
        ground.decisions=partial(decisions,gate=gate,expected_mode=mode)
        fixed=dict(source_commit=PARENT_COMMIT,inference_mode=mode)
        if args.mode=='smoke':ground.smoke(args.root,args.out,PARENT,args.commit,**fixed)
        else:ground.predict(args.root,args.out,PARENT,args.commit,**fixed)
    else:
        info=core.read(args.out/'flow/inference.json')
        if (info.get('source')!=PARENT or info.get('source_commit')!=PARENT_COMMIT
                or info.get('commit')!=args.commit or info.get('inference_mode')!=mode):
            raise ValueError('Fixed confidence gate prediction identity required for scoring')
        core.score(args.root,args.out,args.commit)
        ground.paired_score(args.root,args.out,args.commit)


if __name__=='__main__':main()
