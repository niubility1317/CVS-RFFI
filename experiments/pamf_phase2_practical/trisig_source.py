"""Reuse the fixed complete Tri-SIG-TIME SSL source models, without retraining."""
from pathlib import Path
import torch
from .io import read,write,require,activate
from .group1_resources import Measure

PROFILE='pamf_tri_sig_time_ssl_milkr'
SOURCE_RUN='20261027-phase1-pamf-repl-shuffled-views-m16-r02'
ORDER=[0,5,3,6,2,1,4,7]


def validate_done(done,definition,seed,expected_contract):
    c=done['config']
    require(definition['method_id']=='pamf_tri_sig_time_ssl','Wrong PAMF definition')
    require(done['status']=='SOURCE_FROZEN' and done['epoch']==50 and done['updates']==11100
        and done['target_access'] is False and done['U_truth_read'] is False and done['head_fit_updates']==0,
        'Complete source-only Tri-SIG-TIME SSL required')
    require(c['run_id']==SOURCE_RUN and c['model_seed']==seed and c['view_comparison']['view_count']==3,
        'Wrong original PAMF source run/seed/views')
    require(read(c['source_contract'])==expected_contract,'CHECKPOINT_DATA_CONTRACT_MISMATCH')
    require(c['U_start_epoch']==14 and not c['satellite_concat'] and not c['own_EMA']
        and c['repl_row']['structure']=='SHUFFLE_TIME' and c['repl_row']['L_A'] and c['repl_row']['L_B']
        and c['repl_row']['blocks']==dict(style=True,guard=True,joint=True,extrapolation=True),
        'Ordinary PL or incomplete REPL cannot be called current PAMF')
    for key,value in definition['frozen_REPL_recipe'].items():
        require(c.get(key)==value,'Tri-SIG-TIME SSL mechanism differs: '+key)
    require(done['actual_REPL']['calls']==6660 and done['actual_REPL']['seen']==1701000
        and done['actual_REPL']['student_views']==10206000,'Actual original REPL execution differs')
    return c


def load(c,seed,device):
    from .features import load_reader
    folder=Path(c['PAMF_sources'][str(seed)]);done=read(folder/'complete.json')
    cfg=validate_done(done,read(c['PAMF_definition']),seed,read(c['source_contract']))
    reader=load_reader(c,seed,device)
    pool=reader.sequence_pool
    require(len(pool.views)==3 and all(v.window_order.tolist()==ORDER for v in pool.views),
        'Actual checkpoint temporal permutation is not the fixed Tri-SIG-TIME SSL order')
    require(sum(p.numel() for p in reader.parameters())==232993,'Wrong full PAMF recognizer')
    return dict(model=reader,reader=reader,classes=tuple(range(6)),profile=PROFILE,method='pamf',
        source_path=str(folder),source_checkpoint=str(folder/'final_ssdg.pth'),
        provenance=dict(source_run=SOURCE_RUN,source_code_commit=cfg['code_commit'],
            method='pamf_tri_sig_time_ssl',source_only=True,training_reused=True,
            target_feedback_for_source_choice=False,temporal_permutation=ORDER))


def reuse(c,seed,out,device):
    from .group1_source import source_data,recipe
    from .features import branches
    activate(c);torch.set_num_threads(c.get('source_cpu_threads',1))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    with Measure(out/'A_import_resources.json',dict(profile=PROFILE,seed=seed,stage='A_CHECKPOINT_REUSE_ONLY')) as probe:
        bundle=load(c,seed,device);L=source_data(c,'L_s');reader=bundle['reader']
        with torch.inference_mode():values,logits=branches(reader,L.x[:2].to(device))
        require(logits.shape==(2,6) and bool(torch.isfinite(logits).all()),'Actual source-checkpoint no-query smoke failed')
        probe.model=reader;probe.result.update(new_optimizer_updates=0,query_reads=0,target_reads=0)
        folder=Path(bundle['source_path']);done=read(folder/'complete.json')
        summaries=[]
        for end in (20,30,50):
            path=folder/f'segment{end}'/'phase1_resource_summary.json'
            summaries.append(dict(path=str(path),values=read(path) if path.is_file() else None))
        write(out/'A_resources.json',dict(status='COMPLETE',profile=PROFILE,seed=seed,stage='A_ORIGINAL_REUSED',
            source_method='Tri-SIG-TIME SSL',training_reused=True,new_optimizer_updates=0,
            historical_optimizer_updates=11100,historical_epochs=50,
            historical_training_seconds=done['cumulative_training_seconds'],
            historical_source_end_to_end_seconds=done['seconds'],
            historical_resource_summaries=summaries,
            new_import_resources='A_import_resources.json',
            resource_rule='Original measured resources only; missing fields stay unavailable; import cost is not training cost',
            source_path=str(folder),source_code_commit=done['config']['code_commit'],
            target_contact=False,U_TX_truth_read=False))
        write(out/'group1_complete.json',dict(status='GROUP1_SOURCE_READY',profile=PROFILE,seed=seed,
            recipe=recipe(PROFILE),source_reused_from=str(folder),real_checkpoint_smoke='PASS',
            method='pamf_tri_sig_time_ssl',query_reads=0,target_reads=0,new_optimizer_updates=0,
            historical_optimizer_updates=11100,model_parameters=232993,temporal_permutation=ORDER))
