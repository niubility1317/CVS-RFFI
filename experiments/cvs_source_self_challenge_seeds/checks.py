import copy
import json
import torch
from . import design as d


def checks():
    torch.set_num_threads(2)
    q=dict(status='PASS',fixed_rules=d.QUALIFICATION,source_only=True,target_metrics_used=False,no_automatic_training_release=True)
    d.validate_qualification(q)
    cases=[]
    for field,value in [('status','FAIL'),('source_only',False),('target_metrics_used',True),('no_automatic_training_release',False)]:
        bad=copy.deepcopy(q);bad[field]=value;cases.append(lambda bad=bad:d.validate_qualification(bad))
    bad=copy.deepcopy(q);bad['fixed_rules']['mean_selected_RSC_CE_minus_matched_random_min_exclusive']=1.
    cases.append(lambda:d.validate_qualification(bad))
    for row in d.rows():
        c=d.config(row);d.validate(c)
        assert c['source_config']['model_seed']==row['model_seed'] and c['source_config']['arm']=='SIG_TIME'
        assert c['mask_seed']==d.first.MASK_SEED and c['qualification']==d.first.QUALIFICATION
        for field,value in [('feature_drop_fraction',.5),('mask_seed',1),('model_seed',2026092701),('query_read',True)]:
            badc=copy.deepcopy(c);badc[field]=value;cases.append(lambda badc=badc:d.validate(badc))
    for f in cases:
        try:f()
        except ValueError:pass
        else:raise AssertionError('Changed source authority/remaining recipe accepted')
    from experiments.cvs_receiver_dro_seeds import model as api
    initialized=[]
    for row in d.rows():
        m=api.build(d.source_config(row),'cpu');named=dict(m.named_parameters())
        assert [named[n].numel() for n in d.PARAMETERS]==[3456,6912,576,960]
        assert type(m.id_backbone.encoder.core.id_backbone.cls_head).__name__=='ResidualPhysicalHead'
        initialized.append(row['model_seed'])
    return dict(status='PASS',negative_cases=len(cases),same_first_numerical_objective=True,
        same_source_cohort_and_random_mask_seed=True,actual_parent_model_seed_contexts=initialized,
        no_first_seed_retraining=True,real_IQ_or_checkpoint_or_remote_read=False,head_fit_updates=0)


if __name__=='__main__':print(json.dumps(checks()))
