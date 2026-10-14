"""Focused launch legality and unchanged native-gate negatives; no dataset."""
import copy
import json
import torch
from . import design as d
from .source_census import blind_metadata,source_gate
from experiments.cvs_crossrx_sharpness20 import design as parent
from experiments.cvs_phase1_stack.runtime import installed
from .original_gate import gate


def main():
    c=d.request();c['analysis_code_commit']='0'*40;d.validate(c)
    for key,value in [('U_truth_read',True),('V_gradient',True),('target_access',True),
        ('beta',.75),('pi',.95),('new_acceptance_gate',True),('student_mode','train'),
        ('U_sampler','PAIRS'),('counts',{'V_proxy':27000,'U_s':512}),('source_row','BASE_SSL-s2026092701')]:
        bad=copy.deepcopy(c);bad[key]=value
        try:d.validate(bad)
        except ValueError:pass
        else:raise AssertionError('Unauthorized request accepted: '+key)
    meta={k:torch.tensor(v) for k,v in dict(rx_i=[1,1,1,1],day_i=[1,1,1,1],
        eq_i=[0,0,0,0],sig_i=[0,1,7,8],base_index=[0,1,7,8]).items()}
    dirty=dict(meta,tx_i=torch.tensor([0,1,2,3]));clean=blind_metadata(dirty,4)
    if 'tx_i' in clean or 'tx_i' not in dirty:raise AssertionError('Metadata strip mutated input/keptTX')
    p=parent.config(next(r for r in parent.rows() if r['row_id']=='BASE_SSL-s2026092701'))
    args=parent.make_args(p,'cpu');domain=torch.zeros(4,dtype=torch.long)
    logits=torch.tensor([[9.,0.,0.,0.,0.,0.]]*4)
    with installed(p) as native:
        pseudo,masks=gate(native,args,logits,logits,(domain,clean),domain)
        if not masks['accepted'].all():raise AssertionError('Actual native gate path failed')
        separated=dict(clean,sig_i=torch.tensor([0,10,20,30]),base_index=torch.tensor([0,10,20,30]))
        _,no=gate(native,args,logits,logits,(domain,separated),domain)
        if no['accepted'].any() or no['structural'].any():raise AssertionError('Physical neighbor restriction lost')
        try:gate(native,args,logits,logits,(domain,dirty),domain)
        except ValueError:pass
        else:raise AssertionError('TX metadata accepted by gate')
    v=dict(summary=dict(EVIDENCE_active_count=100,EVIDENCE_active_precision=.98),
        by_RX={str(r):dict(EVIDENCE_active_count=20,EVIDENCE_active_precision=.98) for r in d.SOURCE_RX})
    u=dict(summary=dict(EVIDENCE_active_count=567,new_active_vs_SELF_count=100))
    if not source_gate(dict(V_proxy=v,U_s=u))['passed']:raise AssertionError('Fixed gate pass boundary wrong')
    v['by_RX']['1']['EVIDENCE_active_precision']=.5
    if source_gate(dict(V_proxy=v,U_s=u))['passed']:raise AssertionError('Overall precision hid active RX error')
    v['by_RX']['1']['EVIDENCE_active_precision']=.98;u['summary']['new_active_vs_SELF_count']=99
    if source_gate(dict(V_proxy=v,U_s=u))['passed']:raise AssertionError('No novel SELF opportunity was promoted')
    print(json.dumps(dict(status='PASS',request_negatives=10,original_native_gate_executed=True,
        physical_neighbors_required=True,TX_metadata_rejected=True,metadata_input_preserved=True,
        active_subset_RX_precision_required=True,full_U_and_new_vs_SELF_count_gate=True,
        dataset_checkpoint_remote_reads=0,actual_optimizer_updates=0)),flush=True)


if __name__=='__main__':main()
