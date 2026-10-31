"""One real-source no-target check; policy is fixed, not source-gain selected."""
from pathlib import Path
import time
import numpy as np
from .io import read,write,require
from .milkr_screen import initialize
from .milkr import nested_fisher


def source_iq_rows(data,positions):
    return np.asarray(data.x[np.asarray(positions,dtype=np.int64).tolist()].tolist(),np.float32)


def calibrate(rt):
    from .milkr_cv_screen import extract_for_method
    from experiments.cric_dura_comparison.f_data import SourceDataset
    started=time.perf_counter();prior=read(rt['prior_source_selection'])
    require(prior['status']=='SOURCE_FROZEN' and prior['target_reads']==0 and prior['query_reads']==0,'Source-only original kernel rule required')
    rule=dict(prior['K1_rule'],outer_folds=3,outer_strengths=[0.,.25,.5,.75])
    reader,_=initialize(rt);data=SourceDataset(rt['source_cache'],'L_s',read(rt['source_contract']))
    positions=np.concatenate([np.flatnonzero(data.metadata['y']==c)[:2] for c in range(6)])
    raw=source_iq_rows(data,positions);features,_=extract_for_method(rt,reader,raw,'cuda:0')
    y=np.repeat(np.arange(6),2);ids=data.metadata['ids'][positions].tolist()
    state=nested_fisher.fit(features,y,ids,old_count=3,fixed=rule)
    require(state.selection['outer_support_selection']['selected_strength']==0.,'K2 folds must tie at K1-uniform training')
    write(Path(rt['run_root'])/'source_selection.json',dict(status='SOURCE_FROZEN',K1_rule=rule,control_K1_rule=dict(rule,force_uniform=True),
        source_numerical_smoke='PASS',source_physical_rows=12,target_reads=0,query_reads=0,U_truth_reads=0,
        seconds=time.perf_counter()-started,source_gain_gate=False,
        scope='fixed support-only nested policy; source known-six used only for real-checkpoint execution, not a novel-class performance gate'))
