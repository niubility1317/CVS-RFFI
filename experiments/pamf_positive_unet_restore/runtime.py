"""Source-only gain checks, then the same original strict frozen-model readers."""
import math
import numpy as np
from . import design as d
from experiments.pamf_backfill_restore import runtime as shared
from experiments.pamf_restoration.runner import read_json
from experiments.pamf_uncertainty_restore.evaluation import VIEWS
from experiments.pamf_fcrossrx_readout.frozen import REFERENCE_RUN


def source_statistics(summary):
    expected=dict(status='ANALYZED',role='source',count=1800,F_run=REFERENCE_RUN,F_seed=2026092701,
        original_default_head=True,all_predictions_before_truth=True,training_TX_labels_read=0,
        training_F_access=False,head_fit_updates=0,target_access=False)
    if any(summary.get(k)!=v for k,v in expected.items()):raise ValueError('Original source-only F50 summaries required')
    rows=[r for r in summary['records'] if r['readout']=='restored' and r['dimension']=='overall']
    if len(rows)!=7 or {r['view'] for r in rows}!=set(VIEWS):raise ValueError('All seven source views required')
    values={r['view']:r['accuracy'] for r in rows}
    if any(r['count']!=1800 for r in rows) or any(not math.isfinite(v) or not 0<=v<=100 for v in values.values()):
        raise ValueError('Complete finite source accuracy required')
    return dict(mean6=sum(values[v] for v in VIEWS[1:])/6,clean=values['clean'],
        lowurban=values['practical_low_urban'],midurban=values['practical_mid_urban'])


def gain(candidate,control):
    delta=candidate['mean6']-control['mean6'];clean_drop=control['clean']-candidate['clean']
    if not delta>0 or clean_drop>.1:raise ValueError('Source mean6 positive gain and clean preservation required')
    return dict(mean6_gain_pp=delta,clean_drop_pp=clean_drop,
        lowurban_gain_pp=candidate['lowurban']-control['lowurban'],midurban_gain_pp=candidate['midurban']-control['midurban'])


def validate_source_matrix(args):
    values={};panel=None
    for spec in d.ROWS:
        root=args.runs_root/spec['run']
        if read_json(root/'launch.json')['commit']!=spec['commit']:raise ValueError('Original source code lineage differs')
        path=root/spec['source_summary'];values[spec['row_id']]=source_statistics(read_json(path))
        if any(abs(values[spec['row_id']][k]-v)>1e-9 for k,v in spec['source_statistics'].items()):
            raise ValueError('Frozen source gain inputs differ')
        ids=np.load(path.parent/'prediction/physical_ids.npy',allow_pickle=False)
        if ids.shape!=(1800,) or (panel is not None and not np.array_equal(ids,panel)):
            raise ValueError('Source comparisons require the same existing physical panel')
        panel=ids
    checks={}
    for spec in d.ROWS:
        control=spec['source_control']
        if control is not None:checks[spec['row_id']]=dict(control=control,**gain(values[spec['row_id']],values[control]))
    return dict(status='SOURCE_POSITIVE_SELECTION_FROZEN',rows=[s['row_id'] for s in d.ROWS],checks=checks,
        policy=d.SOURCE_POLICY,source_inputs=values,target_reads=0,target_feedback=False,
        historical_source_qualification_modified=False,training_TX_F_access=False)


def load(args,row_id,device):return shared.load(args,row_id,device,registry=d)


def audit_contract(args):
    shared.audit_contract(args,registry=d)
    selection=read_json(args.output/'source_positive_selection.json')
    if (selection.get('status')!='SOURCE_POSITIVE_SELECTION_FROZEN' or selection.get('rows')!=[s['row_id'] for s in d.ROWS]
            or selection.get('policy')!=d.SOURCE_POLICY or selection.get('target_reads')!=0 or selection.get('target_feedback') is not False
            or selection.get('historical_source_qualification_modified') is not False):
        raise ValueError('Fixed source-positive subset required before target')


def reference(args,role):
    audit_contract(args)
    return shared.reference(args,role,registry=d)
