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
            scalar={};sam={};hidden={};audits=[];n=0
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
                    assert detail['U_packets']==(128 if c['unlabeled'] and n>444 else 0) and detail['accepted_U']==detail['U_packets']
                    assert detail['teacher_forwards']==0 and detail['auxiliary_forwards']==0
                    if c['engineering']['sam']:
                        assert update['second_backward_calls']==1 and update['second_forward_calls']==(3 if n>444 else 1)
                        assert update['same_samples_and_noise'] and update['same_dropout_rng'] and update['rho']==.02
                        for k in ('actual_perturbation_l2','first_gradient_norm','second_gradient_norm','seconds','second_loss','perturbed_parameter_tensors','perturbed_parameters','missing_gradient_parameter_tensors'):sam.setdefault(k,[]).append(update[k])
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
            assert done['distinct_U_visited']==(56700 if c['unlabeled'] else 0)
            for level,entry in hidden.items():entry['lambda_summary']=moments(entry.pop('lambdas'))
            stdout=source_stdout(base/'logs'/f'{rid}.log')
            rows.append(dict(row_id=rid,arm=c['arm'],seed=c['model_seed'],steps=n,epochs=epochs,counts=done['mechanism_execution'],resources=done['resources'],source_stdout=stdout,stdout_errors=stdout['errors'],metric_summary={k:moments(v) for k,v in scalar.items()},sam_summary={k:moments(v) for k,v in sam.items()},hidden_mix=hidden,gradient_audits=audits,raw_steps_ref=str(p/'step_metrics.jsonl.gz'),raw_steps_retained=True,resolved_execution=resolved['execution']))
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
    compile(payload,'engineering-source-analysis','exec');return json.loads(ssh(payload))

def report(v):
    lines=['# 工程方案完整源日志核验','',f"状态 {v['status']}；{len(v['rows'])}/{v['expected_rows']} 行、{v['total_steps']} 次更新。读取全部闭合 step、20 轮 JSONL/CSV 和源 stdout 至 SOURCE_TRAINED；不读取随后目标预测及评分。",'', '|方案／seed|首轮→末轮 CE|SAM 步／额外前向|hidden mix 前向|末轮源 V|训练秒|','|---|---:|---:|---:|---:|---:|']
    for r in v['rows']:
        e=r['epochs'];c=r['counts'];lines.append(f"|{r['row_id']}|{e[0]['ce']:.5f}→{e[-1]['ce']:.5f}|{c['sam_steps']}/{c['sam_second_forwards']}|{c['hidden_mix_steps']}|{e[-1]['validation']['V_accuracy']:.4%}|{r['resources']['identity_training_and_source_diagnostics_seconds']:.2f}|")
    lines+=['','SAM 每次两遍目标及反向、一次正式 AdamW 更新；全部身份参数参与，临时参数精确恢复，第一遍缓冲与随机状态恢复。额外读出和梯度范数在 JSON 中完整汇总。hidden mix 每步仅一个原生层、不同 TX 的 L 供体，特征与软标签共用 lambda。每层调用/覆盖/lambda 单列。原逐步供体及数组保留远端 gzip 原位，定位见 raw_steps_ref。','所有 20 轮损失、LR、参数位移、4 次 V 健康评估和 10 次完整梯度审计保留在分析 JSON；V 不选模。全部 negative 结果保留。','', '未完成：'+str(v['pending']), '异常：'+str(v['errors'])]
    return '\n'.join(lines)+'\n'

if __name__=='__main__':
    from . import design as d
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=collect()
    d.write(a.output/'evidence/source_full_analysis.json',v);(a.output/'source_trajectory.md').write_text(report(v),encoding='utf-8')
    print(json.dumps(dict(status=v['status'],rows=len(v['rows']),steps=v['total_steps'],pending=v['pending'],errors=v['errors'])))
