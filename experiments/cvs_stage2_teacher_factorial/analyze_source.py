"""Read complete available source logs; never read target predictions or scores."""
import argparse
import base64
import gzip
import json
from pathlib import Path
import statistics
from . import design as d

PAYLOAD = r'''
import base64,csv,gzip,io,json,re,sys,time,zlib
from pathlib import Path
base=Path(BASE);sys.path.insert(0,RELEASE)
from experiments.cvs_receiver_residual_capacity4.control import process
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def lines(blob):
 return [json.loads(x) for x in blob.splitlines(keepends=True) if x.endswith(b'\n') and x.strip()]
result=dict(run_id=RUN,read_at=time.time(),target_access=False,rows=[],queue=read(base/'queue_state.json'))
launch=read(base/'launch.json') or {'rows':[]}
for launch_row in launch['rows']:
 root=base/launch_row['row_id'];source=root/'source';p=source/'epoch_metrics.jsonl'
 # Snapshot bytes once; parse all complete records, including complete step prefix.
 epoch_blob=p.read_bytes() if p.is_file() else b'';epochs=lines(epoch_blob)
 p=source/'step_metrics.jsonl.gz';step_blob=p.read_bytes() if p.is_file() else b''
 decompressor=zlib.decompressobj(16+zlib.MAX_WBITS);raw=decompressor.decompress(step_blob) if step_blob else b''
 steps=lines(raw)
 if [r['step'] for r in steps]!=list(range(1,len(steps)+1)):raise ValueError('Noncontiguous complete step prefix')
 if [r['epoch'] for r in epochs]!=list(range(1,len(epochs)+1)):raise ValueError('Noncontiguous epoch prefix')
 logpath=Path(launch_row['log']);stdout=logpath.read_text(errors='replace') if logpath.is_file() else ''
 patterns=['Traceback','CUDA out of memory','FloatingPointError','ValueError','RuntimeError','Killed','UserWarning','FutureWarning']
 matches={pattern:[line for line in stdout.splitlines() if pattern in line] for pattern in patterns}
 csv_path=source/'epoch_metrics.csv';csv_rows=list(csv.DictReader(io.StringIO(csv_path.read_text()))) if csv_path.is_file() else []
 result['rows'].append(dict(row_id=launch_row['row_id'],process=process(launch_row['pid']),launch=launch_row,epochs=epochs,steps=steps,epoch_csv=csv_rows,
  source_completion=read(source/'completion.json'),resolved=read(source/'resolved_config.json'),initialization=read(source/'initialization.json'),
  teacher_diagnostics=read(source/'source_teacher_diagnostics.json'),U_scan=read(source/'post_readapt_U_scan.json'),
  stdout_scan=dict(total_lines=len(stdout.splitlines()),bytes=len(stdout.encode()),markers=matches),
  step_snapshot=dict(compressed_bytes=len(step_blob),complete_records=len(steps),gzip_finished=decompressor.eof,partial_last_line_ignored=bool(raw and not raw.endswith(b'\n'))),
  snapshot_note='Files sampled sequentially; CSV/epoch/step endpoints can differ while live. Every complete available record parsed.'))
blob=gzip.compress(json.dumps(result,separators=(',',':'),allow_nan=False).encode())
print(json.dumps(dict(payload=base64.b64encode(blob).decode())))
'''


def capture():
    from experiments.cvs_phase1_stack.publish import ssh
    payload=PAYLOAD.replace('BASE',repr(d.BASE.as_posix())).replace('RELEASE',repr(d.PROJECT+'/releases/'+d.RELEASE)).replace('RUN',repr(d.RUN))
    wrapper="import subprocess\nr=subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-'],input="+repr(payload)+",text=True,capture_output=True)\nprint(r.stdout)\nif r.returncode:raise RuntimeError(r.stderr)\n"
    blob=base64.b64decode(json.loads(ssh(wrapper))['payload'])
    value=json.loads(gzip.decompress(blob))
    if value['run_id']!=d.RUN or value['target_access']:raise ValueError('Wrong source snapshot')
    return blob,value


def summaries(value):
    rows=[];paired=[]
    for raw in value['rows']:
        epochs=raw['epochs'];steps=raw['steps'];last=epochs[-1] if epochs else {}
        metric_names=sorted({k for step in steps for k,v in step['metrics'].items() if isinstance(v,(int,float))})
        phases=[]
        for phase,subset in [('L_readaptation',[s for s in steps if s['step']<=888]),('LU_identity',[s for s in steps if s['step']>888])]:
            if not subset:continue
            aggregates={}
            for key in metric_names:
                values=[s['metrics'][key] for s in subset if isinstance(s['metrics'].get(key),(int,float))]
                if values:aggregates[key]=dict(count=len(values),mean=statistics.mean(values),min=min(values),max=max(values),first=values[0],last=values[-1])
            phases.append(dict(phase=phase,first_step=subset[0]['step'],last_step=subset[-1]['step'],count=len(subset),metrics=aggregates))
        audits=[]
        for s in steps:
            if s['step']<=888 or not s.get('gradient_audit'):continue
            norms=s['gradient_audit']['weighted_dense_gradient_norms']
            ratios={g:norms['soft_identity'][g]/n if n>0 else None for g,n in norms['ce'].items()}
            audits.append(dict(step=s['step'],weighted_norms=norms,U_to_CE_norm_ratio=ratios))
        audit_summary={g:dict(count=len(vals),median=statistics.median(vals),min=min(vals),max=max(vals)) for g in ('encoder','fusion_head') if (vals:=[a['U_to_CE_norm_ratio'][g] for a in audits if a['U_to_CE_norm_ratio'][g] is not None])}
        rows.append(dict(row_id=raw['row_id'],alive=bool(raw['process']),complete_epochs=len(epochs),complete_steps=len(steps),last_source_V=last.get('V_accuracy'),last_worst_RX=last.get('V_worst_RX'),last_weight_movement=last.get('weight_movement'),counts=last.get('counts'),all_step_phases=phases,all_recorded_gradient_audits=audits,U_to_CE_norm_ratios=audit_summary,source_complete=bool(raw['source_completion']),stdout_scan=raw['stdout_scan']))
    byid={r['row_id']:r for r in value['rows']}
    # Compare every shared completed epoch, not unmatched latest rows.
    for seed in d.SEEDS:
        for ref in ('O','R'):
            for a,b in [('DIRECT','EMA'),('DIRECT_HI','DIRECT'),('EMA','CE'),('DIRECT','CE')]:
                left=byid.get(f'{ref}_{a}-s{seed}');right=byid.get(f'{ref}_{b}-s{seed}')
                if not left or not right:continue
                comparisons=[]
                for x,y in zip(left['epochs'],right['epochs']):
                    assert x['epoch']==y['epoch']
                    movement={g:dict(left=x['weight_movement'][g]['relative_l2'],right=y['weight_movement'][g]['relative_l2'],ratio=x['weight_movement'][g]['relative_l2']/max(y['weight_movement'][g]['relative_l2'],1e-15)) for g in x['weight_movement']}
                    comparisons.append(dict(epoch=x['epoch'],V_difference=x['V_accuracy']-y['V_accuracy'],worst_RX_difference=x['V_worst_RX']-y['V_worst_RX'],weight_movement=movement))
                warmup=None
                if a=='DIRECT' and b=='EMA':
                    diffs=[dict(step=x['step'],loss_difference=x['metrics']['total_loss']-y['metrics']['total_loss']) for x,y in zip(left['steps'][:888],right['steps'][:888]) if x['metrics']['total_loss']!=y['metrics']['total_loss']]
                    warmup=dict(first_nonzero=diffs[0] if diffs else None,max_absolute_difference=max((abs(x['loss_difference']) for x in diffs),default=0),deterministic_CUDA=left['resolved']['backend_flags']['cudnn_deterministic'],interpretation='Same seeded recipe is not bitwise deterministic; pre-treatment numerical drift prevents attributing tiny individual pair differences entirely to EMA')
                paired.append(dict(seed=seed,left=left['row_id'],right=right['row_id'],epochs=comparisons,warmup_numerical_drift=warmup))
    complete=len(rows)==24 and all(r['source_complete'] and r['complete_steps']==8880 and r['complete_epochs']==40 for r in rows)
    return dict(run_id=d.RUN,read_at=value['read_at'],target_access=False,all_available_step_records=sum(r['complete_steps'] for r in rows),rows=rows,paired=paired,complete_source_matrix=complete,scope='Complete source trajectories; no target feedback' if complete else 'Partial live source trajectories; not final results or target feedback')


def report(summary,artifact):
    rows=summary['rows'];title='完整源域分析' if summary.get('complete_source_matrix') else '源域中期检查';text=['# 续训实际更新轨迹：'+title,'',f"读取时间（Unix）：{summary['read_at']}。完整解析当前 {len(rows)}/24 行可用日志，共 {summary['all_available_step_records']} 条完整逐步记录；尚未启动行不补值。未读取目标预测或评分。",'',
    '本报告只反映采样时刻的完整可用前缀。对照比较使用相同 seed、相同 epoch，不比较不同训练进度的最新值。参数相对 L2 位移排除原生 Hz 参数，不等于特征位移，也不证明泛化提升。','',
    '|行|完整 epoch|完整逐步记录|末次 V准确率 (%)|最差源 RX (%)|','|---|---:|---:|---:|---:|']
    for r in rows:
        v='N/A' if r['last_source_V'] is None else f"{r['last_source_V']*100:.4f}"
        w='N/A' if r['last_worst_RX'] is None else f"{r['last_worst_RX']*100:.4f}"
        text.append(f"|{r['row_id']}|{r['complete_epochs']}|{r['complete_steps']}|{v}|{w}|")
    text+=['','## 同进度参数位移比较','','下表给出每一对的最后共同 epoch；所有共同 epoch 的轨迹均在 JSON 中，全部逐步损失统计按前888步/后续步骤分开保存。','',
    '|处理/对照|seed|共同 epoch|V差（百分点）|各参数组相对位移比|','|---|---|---:|---:|---|']
    for pair in summary['paired']:
        if not pair['epochs']:continue
        e=pair['epochs'][-1];ratios=', '.join(f"{g}={v['ratio']:.3f}" for g,v in e['weight_movement'].items())
        text.append(f"|{pair['left'].split('-s')[0]}/{pair['right'].split('-s')[0]}|{pair['seed']}|{e['epoch']}|{100*e['V_difference']:+.4f}|{ratios}|")
    text+=['','## 已记录的分量梯度','','下表统计后888步阶段的全部既有梯度审计点（每888步一次），不是每一步都有分量梯度。使用已经乘上损失权重的梯度范数，不能从范数比推断方向冲突或实际AdamW更新贡献。','', '|行|审计点数|U/CE编码器范数比中位数|U/CE融合头范数比中位数|','|---|---:|---:|---:|']
    for r in rows:
        a=r['U_to_CE_norm_ratios']
        if not a:continue
        text.append(f"|{r['row_id']}|{a['encoder']['count']}|{a['encoder']['median']:.3f}|{a['fusion_head']['median']:.3f}|")
    text+=['','## 解释边界','','参数位移更大不自动意味着识别提升。若U梯度显著大于L梯度，下一步应检查U目标的身份纠错价值及其与L的方向关系；单凭范数不能认定U有害，也不应据此在线修改损失权重。', '', '本批沿用非确定性CUDA配置。全记录比较发现EMA尚未启用的前888步就有微小数值分歧：首个损失差约1e-9至1e-8，随后可能积累。因此这是同seed、同配方对照，并非逐步数值完全相同的反事实；不能把极小单seed差异全部归因于EMA。保留原配方完成三seed，后续若做严格梯度因果实验应重用同一个已冻结预热状态或确定性路径。','', 'EMA与DIRECT的效果必须结合完整40轮训练和固定测试报告。此处不选择候选、不调阈值、不干预健康任务。', '',f'全部源日志快照：`evidence/{artifact}`；完整逐步统计与每个共同epoch对照：`evidence/source_trajectory_summary.json`。stdout全文件已扫描异常标记，命中保存在统计文件；已知环境警告与训练失败应按具体文本区分。']
    return '\n'.join(text)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);parser.add_argument('--compact-snapshot',action='store_true');args=parser.parse_args()
    blob,value=capture();summary=summaries(value);folder=args.output;folder.mkdir(parents=True,exist_ok=True);(folder/'evidence').mkdir(exist_ok=True)
    if args.compact_snapshot:
        # All records were parsed and summarized before omitting large per-U arrays.
        # Original full logs remain at each registered remote source path.
        for row in value['rows']:
            for step in row['steps']:
                for key in ('U_predictions','U_accepted_mask'):step.get('details',{}).pop(key,None)
        value['snapshot_omissions']=['details.U_predictions','details.U_accepted_mask'];value['full_original_logs']=d.BASE.as_posix()+'/<row>/source/step_metrics.jsonl.gz'
        blob=gzip.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode())
    name=f"source_trajectory_{int(value['read_at'])}.json.gz";path=folder/'evidence'/name
    with path.open('xb') as f:f.write(blob)
    d.write(folder/'evidence/source_trajectory_summary.json',summary);(folder/'source_trajectory.md').write_text(report(summary,name),encoding='utf-8')
    print(json.dumps(dict(snapshot=str(path),compressed_bytes=len(blob),rows=len(summary['rows']),steps=summary['all_available_step_records'],epochs={r['row_id']:r['complete_epochs'] for r in summary['rows']})))
