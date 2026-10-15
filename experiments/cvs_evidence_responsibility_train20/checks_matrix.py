"""Fixed budget, real disabled paths and prediction matrix negatives."""
import copy,json
from . import design as d
def check():
    assert len(d.rows())==12 and len(set(r['row_id'] for r in d.rows()))==12
    assert sum(d.active(step) for step in range(1,4441))==180
    assert all(d.active(s) for s in d.PROBE_STEPS) and not d.active(2887) and not d.active(4440)
    n=0
    for r in d.rows():
        c=d.config(r);d.validate(c);a=d.make_args(c,'cpu')
        assert (a.epochs,a.label_epochs,a.pseudo_epochs)==(20,c['label_epochs'],c['pseudo_epochs'])
        assert a.run_id==d.RUN and a.candidate_id==r['row_id'] and a.output_dir==c['output_root']
        assert not a.use_aug and not a.use_concat_sat_channel_aug and not a.use_sat_consistency
        assert a.label_smoothing==0 and a.baseline_ckpt=='' and a.teacher_ckpt==''
        assert a.use_unlabeled==(r['arm']!='BASE_CE') and a.use_ema_teacher==(r['arm']!='BASE_CE')
        assert c['training_mechanism']['scheduled_calls']==d.scheduled(c)
        if r['arm']=='BASE_CE':assert a.lambda_u==0 and a.lambda_ent==0
        if r['arm'].startswith('BASE'):assert c['training_mechanism']['bank_unique_IQ_reads']==0 and c['training_mechanism']['auxiliary_IQ_forward_packets']==0
        for key,value in [('epochs',50),('separate_head_training',True),('checkpoint_sources',['unknown']),('test_views',['practical_high'])]:
            b=copy.deepcopy(c);b[key]=value
            try:d.validate(b)
            except ValueError:n+=1
            else:raise AssertionError(key)
    assert d.result_metadata()['prediction_count']==2016000 and d.result_metadata()['backbone_updates_new']==53280
    return dict(status='PASS',rows=12,new_epochs=240,real_updates=53280,prediction_count=2016000,
        schedule180_onlyepochs1to13=True,baseline_bank_forward_really_disabled_by_contract=True,
        CE_U_EMA_really_disabled_by_args=True,focused_negatives=n,data_checkpoint_reads=0,actual_updates=0)
if __name__=='__main__':print(json.dumps(check()))
