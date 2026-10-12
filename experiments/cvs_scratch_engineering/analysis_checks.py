"""Synthetic closed-log fixtures verify full parsing and rejection behavior."""
import csv,gzip,json,tempfile
from pathlib import Path
from . import design as d
from .source import expected_counts
from .analyze_source import analyze

def run():
    configs=[d.config(next(r for r in d.rows() if r['arm']==arm)) for arm in ('SSL_LR3','SSL_SAM','SSL_HIDDEN_MIX')]
    expected={c['row_id']:expected_counts(c) for c in configs}
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp);(base/'logs').mkdir()
        for c in configs:
            rid=c['row_id'];p=base/rid/'source';p.mkdir(parents=True)
            d.write(p/'completion.json',dict(status='SOURCE_TRAINED',config=c,target_access=False,optimizer_steps=4440,mechanism_execution=expected[rid],distinct_U_visited=56700,resources={}))
            d.write(p/'resolved_config.json',dict(c,execution={}));d.write(p/'calibration.json',dict(status='NOT_APPLICABLE',model_forwards=0))
            epochs=[dict(epoch=e,ce=.1,validation={} if e in (5,10,15,20) else None) for e in range(1,21)]
            (p/'epoch_metrics.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in epochs),encoding='utf-8')
            with (p/'epoch_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['epoch','ce','validation']);writer.writeheader();writer.writerows(epochs)
            with gzip.open(p/'step_metrics.jsonl.gz','wt',encoding='utf-8') as stream:
                for n in range(1,4441):
                    sam=c['engineering']['sam'];active=n>444
                    update=dict(identity_optimizer_steps=1,sam_applied=sam,temporary_parameter_restoration=True)
                    if sam:update.update(second_backward_calls=1,second_forward_calls=3 if active else 1,same_samples_and_noise=True,same_dropout_rng=True,rho=.02,**{k:.1 for k in ('actual_perturbation_l2','first_gradient_norm','second_gradient_norm','seconds','second_loss','perturbed_parameter_tensors','perturbed_parameters','missing_gradient_parameter_tensors')})
                    detail=dict(engineering_update=update,U_packets=128 if active else 0,accepted_U=128 if active else 0,teacher_forwards=0,auxiliary_forwards=0)
                    if c['engineering']['hidden_mix'] and active:detail['hidden_mix']=dict(level=c['engineering']['mix_levels'][(n-445)%2],one_layer_only=True,same_lambda_feature_target=True,different_TX_verified=True,main_rng_restored=True,source_role='L_s',U_truth=False,V_access=False,target_access=False,native_shape=[128,64,10],donors=list(range(128)),lambda_values=[.75]*128,eligible_packets=128,skipped_packets=0,soft_target_sum_error=0)
                    stream.write(json.dumps(dict(step=n,epoch=(n-1)//222+1,metrics=dict(lr=d.learning_rate_step(c,n),SAM_second_loss=.1 if sam else None),details=detail,gradient_audit={} if n%444==0 else None))+'\n')
            (base/'logs'/f'{rid}.log').write_text('RESOLVED {}\nSOURCE_TRAINED {}\nTARGET SENTINEL unread\n',encoding='utf-8')
        v=analyze(base,configs,expected);assert v['status']=='COMPLETE' and v['total_steps']==13320 and not v['errors'],v['errors']
        assert v['rows'][0]['metric_summary']['SAM_second_loss']['mean'] is None
        assert all(r['source_stdout']['source_lines']==2 for r in v['rows'])
        hidden=v['rows'][2]['hidden_mix'];assert hidden['time_down']['calls']==hidden['t2.pool']['calls']==1998
        bad=base/configs[0]['row_id']/'source/completion.json';payload=d.read(bad);payload['target_access']=True;d.write(bad,payload)
        v=analyze(base,configs,expected);assert v['status']=='PARTIAL' and len(v['errors'])==1
    return dict(status='PASS',synthetic_only=True,all_steps_parsed=13320,source_stdout_target_sentinel_unread=True,missing_measurement_null=True,invalid_target_contact_rejected=True,target_access=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(v))
