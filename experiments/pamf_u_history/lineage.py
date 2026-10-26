"""Load this frozen source trajectory through its actual, explicitly named ancestry."""
from pathlib import Path
import torch
from experiments.cric_dura_comparison.io import read
from .source import RUN, METHODS, validate_config, load_prefix, check_state


def validate_segment(c, done, payload, *, folder, epochs, offset, parent):
    cfg=done['config'];validate_config(cfg);check_state(payload)
    if (done.get('status')!='SOURCE_FROZEN' or done.get('epoch')!=epochs or done.get('updates')!=epochs*222
            or done.get('target_access') is not False or done.get('U_truth_read') is not False or done.get('head_fit_updates')!=0
            or cfg['run_id']!=RUN or cfg['row_id']!=c['row_id'] or cfg['paper_method']!=c['paper_method']
            or cfg['output_root']!=str(folder) or cfg['parent_source']!=parent or cfg['step_offset']!=offset*222
            or read(cfg['source_contract'])!=read(c['source_contract'])
            or payload['run_id']!=RUN or payload['candidate_id']!=c['row_id'] or payload['epoch']!=epochs
            or payload['a1_ema_successful_updates']!=epochs*222 or payload['checkpoint_selection']!='final_only'
            or payload['args']['baseline_ckpt']!=parent+'/final_ssdg.pth' or payload['args']['teacher_ckpt']
            or payload['args']['use_unlabeled'] or payload['args']['use_ema_teacher']):
        raise ValueError('Own fixed source-only history segment/default-head ancestry required')


def load_reader(folder, expected_contract, device, require_no_unlabeled=False):
    if require_no_unlabeled:raise ValueError('History candidate explicitly trains on hidden-label source U')
    folder=Path(folder);done=read(folder/'complete.json');c=done['config'];validate_config(c)
    state=read(folder/'u_history_state.json')
    if (str(folder)!=c['output_root'] or c['paper_method'] not in METHODS or done.get('status')!='SOURCE_FROZEN'
            or done.get('epoch')!=50 or done.get('updates')!=11100 or done.get('target_access') is not False
            or done.get('U_truth_read') is not False or done.get('head_fit_updates')!=0
            or done.get('initialization')!='existing_same_contract_SIG1_PL_E20_plus_own10_own20'
            or read(c['source_contract'])!=expected_contract or len(done.get('segments',[]))!=3
            or state['calls']!=6660 or state['seen']!=1701000 or state['min_visits']!=30 or state['max_visits']!=30
            or state['U_truth_read'] or state['target_reads'] or state['model_EMA_updates']):
        raise ValueError('Complete own E50 history source trajectory and matching contract required')
    # Reuse the original prefix check, including its scratch provenance and actual tensors.
    prefix,_=load_prefix(c);segments=done['segments']
    if segments[0]!=prefix:raise ValueError('Declared original SIG1 E20 prefix differs')
    parent=c['shared_SIG1_E20']
    for end,epochs,offset,declared in ((30,10,20,segments[1]),(50,20,30,segments[2])):
        path=folder/('segment'+str(end));actual=read(path/'complete.json')
        if declared!=dict(folder=str(path),**actual):raise ValueError('Declared segment differs from actual source artifact')
        cp=torch.load(path/'final_ssdg.pth',map_location='cpu',weights_only=False)
        validate_segment(c,actual,cp,folder=path,epochs=epochs,offset=offset,parent=parent)
        parent=str(path)
    payload=torch.load(folder/'final_ssdg.pth',map_location='cpu',weights_only=False);check_state(payload)
    if (payload['run_id']!=RUN or payload['candidate_id']!=c['row_id'] or payload['epoch']!=20
            or payload['a1_ema_successful_updates']!=4440 or payload['checkpoint_selection']!='final_only'
            or payload['args']['baseline_ckpt']!=str(folder/'segment30/final_ssdg.pth')):
        raise ValueError('Final checkpoint must be own fixed E50 segment payload')
    from experiments.pamf_repl_completion.source import reader_structure
    from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
    with reader_structure(c):reader=reader_from_payload(c,payload,device)
    return reader,c
