"""Read every closed source step; never read prediction, score or truth files."""
import argparse,csv,gzip,json,math,statistics,time
from pathlib import Path

def finite(v):
    if isinstance(v,float):return math.isfinite(v)
    if isinstance(v,dict):return all(finite(x) for x in v.values())
    if isinstance(v,list):return all(finite(x) for x in v)
    return True

def read(p):
    v=json.loads(p.read_text(encoding='utf-8'))
    if not finite(v):raise ValueError('Nonfinite source JSON '+str(p))
    return v

def moments(values):
    return dict(observations=len(values),mean=statistics.mean(values),min=min(values),max=max(values),first=values[0],last=values[-1]) if values else dict(observations=0,mean=None,min=None,max=None,first=None,last=None)

def source_stdout(path):
    result=dict(source_lines=0,errors=[],closed_source_segment=False,target_segment_read=False)
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            result['source_lines']+=1
            if any(x in line for x in ('Traceback','CUDA out of memory','FloatingPointError','RuntimeError','ValueError')):result['errors'].append(line.rstrip())
            if line.startswith('SOURCE_TRAINED '):result['closed_source_segment']=True;break
    if not result['closed_source_segment']:raise ValueError('Source stdout not closed')
    return result

def analyze(base,configs,expected):
    base=Path(base);rows=[];pending=[];errors=[]
    for c in configs:
        rid=c['row_id'];p=base/rid/'source'
        if not (p/'completion.json').exists():pending.append(dict(row_id=rid));continue
        try:
            done=read(p/'completion.json');resolved=read(p/'resolved_config.json');cal=read(p/'calibration.json')
            assert done['config']==c and done['status']=='SOURCE_TRAINED' and not done['target_access'] and done['optimizer_steps']==4440
            assert all(resolved[k]==v for k,v in c.items())
            assert cal['status']=='NOT_APPLICABLE' and cal['model_forwards']==0
            assert all(done['mechanism_execution'][k]==v for k,v in expected[rid].items())
            epochs=[json.loads(line) for line in (p/'epoch_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(epochs)==20 and [e['epoch'] for e in epochs]==list(range(1,21))
            with (p/'epoch_metrics.csv').open(encoding='utf-8') as stream:csvrows=list(csv.DictReader(stream))
            assert len(csvrows)==20
            for e,ce in zip(epochs,csvrows):
                assert all(('' if v is None else str(v))==ce[k] for k,v in e.items() if not isinstance(v,(dict,list)))
                assert (e['validation'] is not None)==(e['epoch'] in (5,10,15,20))
            scalar={};sam={};hidden={};audits=[];relations={};n=0
            with gzip.open(p/'step_metrics.jsonl.gz','rt',encoding='utf-8') as stream:
                for line in stream:
                    v=json.loads(line);n+=1
                    assert finite(v) and v['step']==n and v['epoch']==(n-1)//222+1
                    m=v['metrics'];detail=v['details'];update=detail['engineering_update']
                    lo,hi=c['optimizer']['min_lr'],c['optimizer']['peak_lr']
                    rate=(lo+(hi-lo)*(n-1)/443 if n<=444 else lo+(hi-lo)*.5*(1+math.cos(math.pi*(n-445)/3995))) if c['engineering']['warm_lr'] else lo+(hi-lo)*.5*(1+math.cos(math.pi*((n-1)//222)/19))
                    assert math.isclose(m['lr'],rate,abs_tol=1e-14,rel_tol=1e-12)
                    for k,value in m.items():
                        values=scalar.setdefault(k,[])
                        if value is not None:values.append(value)
                    assert update['identity_optimizer_steps']==1 and update['sam_applied']==c['engineering']['sam'] and update['temporary_parameter_restoration']
                    assert detail['U_packets']==(128 if n>444 else 0) and detail['accepted_U']==detail['U_packets']
                    assert detail['teacher_forwards']==0 and detail['auxiliary_forwards']==0
                    if c['engineering']['sam']:
                        assert update['second_backward_calls']==1 and update['second_forward_calls']==(3 if n>444 else 1)
                        assert update['same_samples_and_noise'] and update['same_dropout_rng'] and update['rho']==.02
                        for k in ('actual_perturbation_l2','first_gradient_norm','second_gradient_norm','seconds','second_loss','perturbed_parameter_tensors','perturbed_parameters','missing_gradient_parameter_tensors'):sam.setdefault(k,[]).append(update[k])
                    rel=detail['relation'];active=n>444 and n%4==0
                    assert rel['active']==active
                    if active:
                        assert rel['L_packets']==30 and rel['forwards']==1 and rel['support_query_PID_disjoint'] and len(set(rel['support_IDs']))==30 and rel['all6_TX'] and rel['per_TX_RX']==1 and rel['source_RX']==[1,3,4,6,8]
                        assert rel['support_day']==[1,2,3][((n-448)//4)%3]
                        assert rel['reference_has_gradient']==(c['relation']['mode']=='extra_CE') and rel['prototype_grad'] is False and rel['query_features_reused'] and not rel['U_labels_used'] and not rel['V_access'] and not rel['target_access']
                        assert rel['level_average']==len(c['relation']['levels']) and set(rel['levels'])==set(c['relation']['levels'])
                        assert math.isclose(m['weighted_relation'],m['relation']*c['relation']['weight']*detail['ramp'],abs_tol=1e-6,rel_tol=1e-6)
                        for name in ('raw_loss','weighted_loss','head_L_accuracy','head_L_margin'):relations.setdefault('all/'+name,[]).append(rel[name])
                        for level,record in rel['levels'].items():
                            assert record['all6_classes_compete'] and record['reference_RX_excludes_query'] and record['query_count']==128
                            assert 0<=record['head_wrong_relation_correct']<=128 and 0<=record['head_correct_relation_wrong']<=128
                            for name,value in record.items():
                                if isinstance(value,(int,float)) and not isinstance(value,bool):relations.setdefault(level+'/'+name,[]).append(value)
                    else:assert m['relation']==0 and m['weighted_relation']==0
                    replay=detail.get('clean_replay')
                    assert bool(replay)==(c['engineering']['clean_replay'] and n>444)
                    if replay:
                        assert replay['identity_transform'] and replay['packets']==128 and replay['main_rng_restored'] and replay['dropout_clean_replay']
                        assert math.isclose(m['ce'],m['style_ce'],abs_tol=1e-6,rel_tol=1e-6)
                    assert not resolved['execution']['explicit_CFO_identity_input']
                    info=detail.get('hidden_mix')
                    assert bool(info)==(c['engineering']['hidden_mix'] and n>444)
                    if info:
                        level=c['engineering']['mix_levels'][(n-445)%2]
                        assert info['level']==level and info['one_layer_only'] and info['same_lambda_feature_target'] and info['different_TX_verified'] and info['main_rng_restored']
                        assert info['source_role']=='L_s' and not any(info[k] for k in ('U_truth','V_access','target_access'))
                        assert info['native_shape'][0]==128 and len(info['donors'])==len(info['lambda_values'])==128
                        assert all(.5<=x<=1 for x in info['lambda_values']) and all(0<=x<128 for x in info['donors'])
                        assert info['eligible_packets']+info['skipped_packets']==128 and info['soft_target_sum_error']<=1e-6
                        entry=hidden.setdefault(level,dict(calls=0,packets=0,eligible=0,skipped=0,lambdas=[]))
                        entry['calls']+=1;entry['packets']+=128;entry['eligible']+=info['eligible_packets'];entry['skipped']+=info['skipped_packets'];entry['lambdas'].extend(info['lambda_values'])
                    audit=v.get('gradient_audit')
                    if audit is not None:audits.append(dict(step=n,audit=audit))
            assert n==4440 and len(audits)==10 and finite(epochs)
            assert done['distinct_U_visited']==56700
            for level,entry in hidden.items():entry['lambda_summary']=moments(entry.pop('lambdas'))
            stdout=source_stdout(base/'logs'/f'{rid}.log')
            rows.append(dict(row_id=rid,arm=c['arm'],seed=c['model_seed'],steps=n,epochs=epochs,counts=done['mechanism_execution'],resources=done['resources'],source_stdout=stdout,stdout_errors=stdout['errors'],metric_summary={k:moments(v) for k,v in scalar.items()},sam_summary={k:moments(v) for k,v in sam.items()},hidden_mix=hidden,relation_summary={k:moments(v) for k,v in relations.items()},gradient_audits=audits,raw_steps_ref=str(p/'step_metrics.jsonl.gz'),raw_steps_retained=True,resolved_execution=resolved['execution']))
        except Exception as error:errors.append(dict(row_id=rid,reason=type(error).__name__+': '+str(error)))
    return dict(status='COMPLETE' if len(rows)==len(configs) and not errors else 'PARTIAL',run_id=configs[0]['run_id'],read_at=time.time(),rows=rows,pending=pending,errors=errors,expected_rows=len(configs),total_steps=sum(r['steps'] for r in rows),source_only=True,target_access=False,prediction_read=False,truth_read=False,scores_read=False)

# LOCAL CONTROL: the remote payload ends above this line.
def collect():
    from . import design as d
    from .source import expected_counts
    from experiments.cvs_phase1_stack.publish import ssh
    own=Path(__file__).read_text(encoding='utf-8').split('# LOCAL CONTROL:')[0]
    configs=[d.config(r) for r in d.rows()];expected={c['row_id']:expected_counts(c) for c in configs}
    payload=own+'\nprint(json.dumps(analyze('+repr(d.BASE.as_posix())+','+repr(configs)+','+repr(expected)+'),allow_nan=False))\n'
    compile(payload,'relation-source-analysis','exec');return json.loads(ssh(payload))

def report(v):
    lines=['# 跨 RX 关系约束完整源日志核验','',f"{v['status']}；{len(v['rows'])}/{v['expected_rows']}行，{v['total_steps']}次更新；全闭合源step/epoch/stdout，只读到SOURCE_TRAINED，不读取target输出。",'', '|row|层|关系CE均值|关系准确率均值|关系margin均值|身份头错/关系对均数|身份头对/关系错均数|','|---|---|---:|---:|---:|---:|---:|']
    for row in v['rows']:
        stats=row['relation_summary']
        for level in ('time_down','t2.pool','id_feat_joint'):
            if level+'/loss' not in stats:continue
            get=lambda k:stats[level+'/'+k]['mean']
            lines.append(f"|{row['row_id']}|{level}|{get('loss'):.6f}|{get('accuracy'):.4%}|{get('margin'):.6f}|{get('head_wrong_relation_correct'):.4f}|{get('head_correct_relation_wrong'):.4f}|")
    lines+=['','每行999次/每次30个合法L参考，全部TX×5RX均衡、循环day；与128个主L查询的PID完全不交。每query排除其RX，用其余4RX的24个参考估计中心及六TX原型。合并原型或分RX cosine/logmeanexp；tau.1。关系原型eval/no_grad，梯度只经实际query特征回身份；支持当前模型无参数更新。L30_CE对照则直接以额外L30训练CE并恢复主RNG。原身份CE/固定U软KL保留；三层组合按启用层数取mean。','关系读出的分类结果与原身份头不同，源训练读出的纠错/破坏只比较两种规则；不能冒充参数更新后的因果纠错，也不能冒充未知RX测试。日志同时保留原身份头L准确率/margin和关系CE/margin/低范数率、全部源V健康与10次梯度审计，V不选模。','支持indices/PIDs保存远端原step gzip中；完整摘要在JSON中。原型分离的是不同RX的类别对应，不声称纯TX物理分离，不删除全部交互。','',str(v['pending']),str(v['errors'])]
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    from . import design as d
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=collect()
    d.write(a.output/'evidence/source_full_analysis.json',v);(a.output/'source_trajectory.md').write_text(report(v),encoding='utf-8')
    print(json.dumps(dict(status=v['status'],rows=len(v['rows']),steps=v['total_steps'],pending=v['pending'],errors=v['errors'])))
