"""Fit one frozen feature bridge using only paired source-L received IQ."""
from pathlib import Path
import time
import numpy as np
from .io import read,write,require
from .milkr_screen import initialize,subset
from .milkr.source_bridge import FeatureBridge,blocks
from .milkr import nested_fisher


def pair_error(raw,restored):
    return float(np.mean([np.mean((a-b)**2) for a,b in zip(blocks(raw),blocks(restored))]))


def calibrate(rt):
    from .milkr_cv_screen import extract_for_method
    from experiments.cric_dura_comparison.f_data import SourceDataset
    started=time.perf_counter();prior=read(rt['prior_source_selection'])
    require(prior['status']=='SOURCE_FROZEN' and prior['target_reads']==0 and prior['query_reads']==0,'Original source-only rule required')
    rule=dict(prior['K1_rule'],outer_folds=3,outer_strengths=[0.,.25,.5,.75])
    reader,engine=initialize(rt);data=SourceDataset(rt['source_cache'],'L_s',read(rt['source_contract']))
    metadata=data.metadata;raw_parts=[];restored_parts=[]
    for epoch in rt['source_cache_epochs']:
        folder=Path(rt['augmentation_cache'])/'F0'/f'stage1-epoch{epoch:03d}-p80'
        require(read(folder/'complete.json')['status']=='L_AUGMENTATION_READY','Missing source augmentation')
        require(np.array_equal(np.load(folder/'ids.npy'),metadata['ids']),'Source physical alignment differs')
        bank=np.load(folder/'x.npy',mmap_mode='r')
        raw_parts.append(extract_for_method(rt,reader,np.asarray(bank),'cuda:0')[0])
        restored_parts.append(extract_for_method(rt,reader,np.asarray(bank),'cuda:0',engine)[0])
        print('BRIDGE_SOURCE_EPOCH',epoch,len(bank),flush=True)
    raw={key:np.concatenate([part[key] for part in raw_parts]) for key in raw_parts[0]}
    restored={key:np.concatenate([part[key] for part in restored_parts]) for key in restored_parts[0]}
    receivers=np.tile(metadata['rx_i'],len(rt['source_cache_epochs']));diagnostics=[]
    for rx in sorted(np.unique(receivers).tolist()):
        train=np.flatnonzero(receivers!=rx);held=np.flatnonzero(receivers==rx)
        shadow=FeatureBridge.fit(subset(raw,train),subset(restored,train))
        held_raw=subset(raw,held);held_restored=subset(restored,held)
        diagnostics.append(dict(receiver=int(rx),held_observations=len(held),
            unbridged_pair_MSE=pair_error(held_raw,held_restored),bridged_pair_MSE=pair_error(shadow.transform(held_raw),held_restored)))
    bridge=FeatureBridge.fit(raw,restored);bridge.save(Path(rt['run_root'])/'source_bridge.npz')
    loaded=FeatureBridge.load(Path(rt['run_root'])/'source_bridge.npz')
    positions=np.concatenate([np.flatnonzero(metadata['y']==c)[:2] for c in range(6)])
    small=subset(raw,positions);mapped=loaded.transform(small)
    require(all(np.array_equal(v,mapped[key]) for key,v in bridge.transform(small).items()),'Bridge checkpoint reload differs')
    state=nested_fisher.fit(mapped,np.repeat(np.arange(6),2),metadata['ids'][positions].tolist(),old_count=3,fixed=rule)
    require(state.selection['outer_support_selection']['selected_strength']==0.,'K2 outer folds must tie at K1')
    write(Path(rt['run_root'])/'source_selection.json',dict(status='SOURCE_FROZEN',K1_rule=rule,control_K1_rule=rule,
        source_numerical_smoke='PASS',source_physical_rows=len(metadata['ids']),source_paired_observations=len(raw['z_id']),
        source_bridge_checkpoint='source_bridge.npz',bridge_rank=8,ridge_fraction=.01,leave_one_source_RX_out=diagnostics,
        target_reads=0,query_reads=0,U_truth_reads=0,F_updates=0,G_updates=0,seconds=time.perf_counter()-started,
        source_contract=rt['source_contract'],F_source=rt['PAMF_sources'][str(rt['model_seeds'][0])],G_source=rt['DURA']['checkpoint'],
        scope='source-only frozen low-rank parameter checkpoint, no source examples exposed to Phase2; raw path mapped to restored canonical space'))
