"""Synthetic complete logs verify the relation parser, without IQ or target."""
import csv,gzip,json,tempfile
from pathlib import Path
from . import design as d
from .source import expected_counts
from .analyze_source import analyze

def run():
    configs=[d.config(next(r for r in d.rows() if r['arm']==a)) for a in d.ARMS]
    expected={c['row_id']:expected_counts(c) for c in configs}
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);(root/'logs').mkdir()
        for c in configs:
            rid=c['row_id'];p=root/rid/'source';p.mkdir(parents=True)
            d.write(p/'completion.json',dict(status='SOURCE_TRAINED',config=c,target_access=False,optimizer_steps=4440,mechanism_execution=expected[rid],distinct_U_visited=56700,resources={}))
            d.write(p/'resolved_config.json',dict(c,execution={'explicit_CFO_identity_input':False}));d.write(p/'calibration.json',dict(status='NOT_APPLICABLE',model_forwards=0))
            epochs=[dict(epoch=e,ce=.1,validation={} if e in (5,10,15,20) else None) for e in range(1,21)]
            (p/'epoch_metrics.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in epochs),encoding='utf-8')
            with (p/'epoch_metrics.csv').open('w',encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['epoch','ce','validation']);writer.writeheader();writer.writerows(epochs)
            with gzip.open(p/'step_metrics.jsonl.gz','wt',encoding='utf-8') as stream:
                for n in range(1,4441):
                    active=n>444 and n%4==0;ramp=min(1,max(0,(n-444)/888))
                    rel=dict(active=active,L_packets=0,forwards=0)
                    if active:
                        rel.update(L_packets=30,forwards=1,support_query_PID_disjoint=True,support_IDs=[f'L{i}' for i in range(30)],all6_TX=True,per_TX_RX=1,source_RX=[1,3,4,6,8],support_day=[1,2,3][((n-448)//4)%3],reference_has_gradient=c['relation']['mode']=='extra_CE',prototype_grad=False,query_features_reused=True,U_labels_used=False,V_access=False,target_access=False,level_average=len(c['relation']['levels']),raw_loss=.2,weighted_loss=.2*ramp,head_L_accuracy=.9,head_L_margin=1.,levels={level:dict(all6_classes_compete=True,reference_RX_excludes_query=True,query_count=128,head_wrong_relation_correct=5,head_correct_relation_wrong=7,loss=.2,accuracy=.8,margin=.1) for level in c['relation']['levels']})
                    details=dict(engineering_update=dict(identity_optimizer_steps=1,sam_applied=False,temporary_parameter_restoration=True),U_packets=128 if n>444 else 0,accepted_U=128 if n>444 else 0,teacher_forwards=0,auxiliary_forwards=0,relation=rel,ramp=ramp)
                    stream.write(json.dumps(dict(step=n,epoch=(n-1)//222+1,metrics=dict(ce=.1,relation=.2 if active else 0,weighted_relation=.2*ramp if active else 0,lr=d.learning_rate_step(c,n),unmeasured=None),details=details,gradient_audit={} if n%444==0 else None))+'\n')
            (root/'logs'/f'{rid}.log').write_text('RESOLVED {}\nSOURCE_TRAINED {}\nTARGET SENTINEL unread\n',encoding='utf-8')
        v=analyze(root,configs,expected);assert v['status']=='COMPLETE' and v['total_steps']==22200,v['errors']
        assert all(r['metric_summary']['unmeasured']['mean'] is None and r['source_stdout']['source_lines']==2 and r['relation_summary']['all/raw_loss']['observations']==999 for r in v['rows'])
        path=root/configs[0]['row_id']/'source/completion.json';bad=d.read(path);bad['target_access']=True;d.write(path,bad)
        v=analyze(root,configs,expected);assert v['status']=='PARTIAL' and len(v['errors'])==1
    return dict(status='PASS',synthetic_only=True,all_steps_parsed=22200,relation_calls_each=999,source_stdout_target_sentinel_unread=True,missing_measurement_null=True,invalid_target_contact_rejected=True,target_access=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=run();d.write(a.output,v);print(json.dumps(v))
