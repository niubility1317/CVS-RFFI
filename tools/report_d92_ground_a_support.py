"""Render completed paired A/B/C support tables; no fitting or data selection."""
import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path

KS=(1,5,10,20)
NEWS=(0,2,5,10,20)
METRICS=('A_old_accuracy','B_old_accuracy','C_old_accuracy','C_new_accuracy','C_h',
         'adaptation_gain_B_minus_A','total_old_accuracy_drop','C_abs_new_old_gap')
METHODS={'D92-AffineJointLocalRidge-v1':'R_AFFINE_seq',
         'D92-ConditionalJointLocalRidge-v1':'R_CONDITIONAL_seq',
         'D92-MarginJointLocalRidge-v1':'R_MARGIN_seq'}
LEGACY_METHODS=frozenset(('D92-AffineJointLocalRidge-v1','D92-ConditionalJointLocalRidge-v1'))
MARGIN_METHOD='D92-MarginJointLocalRidge-v1'
SCOPE='SUPPORT_ONLY_GROUND_A_PAIRED_OLD_HELD_NOT_QUERY_EVALUATION'

def require(condition,message):
    if not condition:raise ValueError(message)

def percent(value):return 'N/A' if value is None else f'{100*value:.4f}'

def aggregate(items,dimensions):
    groups=defaultdict(list)
    for method,rows in items:
        for row in rows:
            require(row['metric'] in METRICS,'Unknown paired metric')
            group=(method,)+tuple(row[key] for key in dimensions)+(row['metric'],)
            groups[group].append(row)
    result=[]
    for key,rows in sorted(groups.items(),key=repr):
        count=sum(r['measured_parent_count'] for r in rows)
        parents=sum(r['parent_count'] for r in rows)
        total=sum(r['mean']*r['measured_parent_count'] for r in rows if r['mean'] is not None)
        result.append(dict(zip(('method',)+dimensions+('metric',),key),parent_count=parents,
            measured_parent_count=count,mean=None if count==0 else total/count))
    return result

def cells(rows,dimensions):
    groups={}
    for row in rows:
        key=tuple(row[k] for k in dimensions)
        cell=groups.setdefault(key,{})
        require(row['metric'] not in cell,'Duplicate aggregate metric')
        cell[row['metric']]=row
    result=[]
    for key,metrics in sorted(groups.items(),key=repr):
        require(set(metrics)==set(METRICS),'Missing paired report metric')
        result.append(dict(zip(dimensions,key),means={m:metrics[m]['mean'] for m in METRICS},
            parent_count=metrics['A_old_accuracy']['parent_count'],
            measured_parents=metrics['A_old_accuracy']['measured_parent_count']))
    return result

def validate_row(pairing,row):
    expected=dict(run_id=row['source_run_id'],row_id=row['source_row_id'],
        checkpoint_sha256=row['expected_checkpoint_sha256'],capsule_id=row['expected_capsule_id'],
        model_seed=row['expected_model_seed'])
    require(pairing['status']=='GROUND_A_SUPPORT_PAIRING_COMPLETE' and pairing['scope']==SCOPE
        and pairing['schema']=='d92_ground_a_support_pairing_v1','Unverified paired input')
    method=row['expected_method']
    require(method in METHODS and pairing['method']==method and pairing['binding']==expected
        and pairing['adapted_path']==METHODS[method],'Wrong method/physical binding')
    if method==MARGIN_METHOD:
        require(pairing.get('method_schema')==row.get('expected_summary_schema')=='d92_margin_joint_local_ridge_v1'
            and row.get('expected_summary_status')=='COMPLETE_MARGIN_JOINT_PROBE_VERIFIED',
            'Wrong fixed Margin method/summary identity')
    require(pairing['query_rows_used']==pairing['source_rows_used']==0
        and not any(pairing[k] for k in ('original_summary_modified','original_trace_modified',
            'training_performed','calibration_performed','model_called_during_truth_join')),'Forbidden behavior')
    require(pairing['parent_count']==40 and pairing['k1_parent_count']==10
        and len(pairing['ordered_ground_classes'])==len(set(pairing['ordered_ground_classes']))==6,
        'Incomplete physical/class coverage')
    for table,rows in pairing['statistics'].items():
        require(table in ('overall','by_k_new_count','by_receiver_scene','by_model_row'),'Unknown table')
        seen=set()
        for item in rows:
            require(item['metric'] in METRICS and item['path']==METHODS[method]
                and item['diagnostic'] in ('oof','proxy'),'Unknown metric/path')
            identity=tuple((k,str(v)) for k,v in sorted(item.items())
                if k not in ('mean','minimum','maximum','parent_count','measured_parent_count','null_parent_count'))
            require(identity not in seen,'Duplicate input metric');seen.add(identity)
            n=item['measured_parent_count'];parents=item['parent_count'];mean=item['mean']
            require(type(n) is int and type(parents) is int and 0<=n<=parents
                and (mean is None)==(n==0),'Invalid paired mean/count')
            require(mean is None or type(mean) in (int,float) and math.isfinite(mean),'Nonfinite report value')
            if method==MARGIN_METHOD and table=='by_model_row':
                require(item.get('model_seed')==row['expected_model_seed'] and item.get('row_id')==row['source_row_id'],
                    'Margin model table escaped the paired source row')
            if 'k' in item:
                require(item['k'] in KS and item['new_count'] in NEWS,'Unknown matrix cell')
                if item['k']==1:require(mean is None and n==0,'True K1 held value fabricated')
                if item['new_count']==0 and item['metric'] in ('C_new_accuracy','C_h','C_abs_new_old_gap'):
                    require(mean is None,'New-class metric fabricated for new0')
        # Every table must carry all eight metrics in each declared cell.
        dims=tuple(k for k in rows[0] if k not in ('metric','mean','minimum','maximum',
            'parent_count','measured_parent_count','null_parent_count')) if rows else ()
        require(bool(rows),'Missing paired table')
        cells(rows,dims)
    matrix=pairing['statistics']['by_k_new_count']
    expected_cells=set(itertools.product(('oof','proxy'),KS,NEWS))
    require({(r['diagnostic'],r['k'],r['new_count']) for r in matrix}==expected_cells,'Partial K/new matrix')
    for r in matrix:
        require(r['parent_count']==2,'Wrong row parent count')
        missing=r['k']==1 or r['new_count']==0 and r['metric'] in ('C_new_accuracy','C_h','C_abs_new_old_gap')
        require(r['measured_parent_count']==(0 if missing else 2),'Incomplete row measurement')

def assemble(spec,complete,pairings,*,source):
    rows=spec['rows'];methods={r['expected_method'] for r in rows}
    legacy=len(rows)==8 and methods==LEGACY_METHODS
    margin=len(rows)==4 and methods=={MARGIN_METHOD}
    require(legacy or margin,'Only legacy eight-row or Margin-only four-row reporting is supported')
    require(len({r['row_id'] for r in rows})==len(rows),'Wrong declared rows')
    groups=defaultdict(list)
    for row in rows:groups[row['source_run_id']].append(row)
    require(len(groups)==(2 if legacy else 1) and all(len(g)==4 and len({r['expected_method'] for r in g})==1 for g in groups.values()),
        'Each method must have one complete four-row source run')
    require(len({(r['source_run_id'],r['source_row_id']) for r in rows})==len(rows),'Duplicate original source row')
    if margin:
        seeds={r['expected_model_seed'] for r in rows};capsules={r['expected_capsule_id'] for r in rows}
        require(len(seeds)==len(capsules)==2 and {(r['expected_model_seed'],r['expected_capsule_id']) for r in rows}
            ==set(itertools.product(seeds,capsules)),'Margin report requires the exact two-model/two-capsule crossing')
        require(len({r['expected_checkpoint_sha256'] for r in rows})==2 and
            all(len({r['expected_checkpoint_sha256'] for r in rows if r['expected_model_seed']==seed})==1 for seed in seeds),
            'Margin model/checkpoint mapping changed')
    require(complete['schema']=='d92_ground_a_support_supervisor_v1'
        and complete['status']=='GROUND_A_SUPPORT_SUPERVISOR_COMPLETE'
        and complete['run_id']==spec['run_id'] and complete['rows']==complete['completed_rows']==len(rows)
        and complete['parent_count']==40*len(rows),'Incomplete supervisor')
    require(complete['query_rows_used']==complete['source_rows_used']==0
        and not complete['training_performed'] and not complete['original_inputs_modified'],'Forbidden supervisor input')
    require(set(pairings)=={r['row_id'] for r in rows}==set(complete['row_outputs']),'Missing/extra row output')
    items={name:[] for name in ('overall','by_k_new_count','by_receiver_scene','by_model_row')}
    resources=[]
    for row in rows:
        pairing=pairings[row['row_id']];validate_row(pairing,row)
        for table in items:items[table].append((pairing['method'],pairing['statistics'][table]))
        resources.append(dict(method=pairing['method'],row_id=row['row_id'],measurements=pairing['resources']))
    require(all(sum(r['expected_method']==m for r in rows)==4 for m in methods),'Unbalanced declared method rows')
    dimensions=dict(overall=('diagnostic','path','population'),by_k_new_count=('diagnostic','path','k','new_count'),
        by_receiver_scene=('diagnostic','path','receiver','scenario','k','new_count'),
        by_model_row=('diagnostic','path','model_seed','row_id','k','new_count'))
    tables={name:cells(aggregate(values,dimensions[name]),('method',)+dimensions[name]) for name,values in items.items()}
    overall=[r for r in tables['overall'] if r['diagnostic']=='oof' and r['population']=='new_present']
    require(len(tables['by_k_new_count'])==40*len(methods),'Incomplete method K/new matrix')
    require(len(overall)==len(methods) and all(r['measured_parents']==96 for r in overall),'Incomplete overall paired population')
    # Keep parent-first H/abs-gap supplied by the independent scorer.
    lines=['# 原地面A与联合LocalRidge方法：完整三阶段配对','',
        f'全部{len(rows)}行已独立读回完成，共{40*len(rows)}个parent（每种方法160个）。旧类数6，新增类数0、2、5、10、20；K为1、5、10、20。',
        'A使用原地面6类float32分类头；A/B/C旧类准确率来自同一source row、split与物理旧held support。B/C为原方法已固定预测，没有重拟合。',
        '本报告是合法support持出诊断，不是query准确率或新增独立验证。A预测先固定，随后连接truth；结果不回流选模、参数或重跑。',
        'H、注册下降、绝对新旧差先逐parent计算，再等parent平均；不以汇总准确率重新计算H或绝对差。true K1无独立held，记N/A。',
        '理想目标：B−A≥10个百分点、B−C旧≤1个百分点、平均绝对新旧差≤3个百分点；当前为逐步改善方向，未增加硬门槛。','']
    def table(title,values,dims):
        lines.extend([title,'','| '+' | '.join(dims)+' | 旧类数 | 总注册类数 | A旧 | B旧 | C旧 | C新 | H | B−A | 注册下降 | 绝对差 |',
            '|'+'---|'*(len(dims)+2)+'---:|'*8])
        for row in values:
            total='6+new' if 'new_count' not in row else str(6+row['new_count'])
            lines.append('| '+' | '.join([str(row[k]) for k in dims]+['6',total]+[percent(row['means'][m]) for m in METRICS])+' |')
        lines.append('')
    table('整体（OOF，K5/10/20、新增2/5/10/20，每方法96个可测parent）',overall,('method','measured_parents'))
    table('完整K×新增类数',tables['by_k_new_count'],('method','diagnostic','k','new_count','measured_parents'))
    for name in ('by_receiver_scene','by_model_row'):
        table(name,tables[name],('method',)+dimensions[name]+('measured_parents',))
    lines+=['资源口径','',
        f'唯一supervisor程序墙钟{complete["wall_seconds"]:.6f}s，CPU进程峰值RSS为{complete["peak_process_rss_bytes"]}B；包含配对验证与日志。单条分类头评分计时不代表完整配对耗时。',
        '原分类头及packet文件字节取下表当次pairing.resources的实测值；缺失值为N/A。文件/数组字节不等于传输字节或部署常驻内存；实际星地传输、GPU峰值、能耗未测，为N/A。',
        '| 方法 | 行 | 原始实测资源 |','|---|---|---|']
    for row in resources:lines.append('| '+row['method']+' | '+row['row_id']+' | '+json.dumps(row['measurements'],ensure_ascii=False,allow_nan=False)+' |')
    lines+=['',f'来源：`{source}`；实际runtime commit：`{complete["commit"]}`。','']
    value=dict(status='COMPLETE_GROUND_A_PAIRED_REPORT_RENDERED',run_id=spec['run_id'],scope=SCOPE,
        actual_A_scope='MATCHED_OLD_SUPPORT_HELD_ONLY_NOT_QUERY_OR_DEPLOYMENT_ACCURACY',
        automatic_promotion=False,goal_complete=False,runtime_commit=complete['commit'],
        source=source,overall=overall,tables=tables,resources=resources,
        supervisor_resources={k:complete[k] for k in ('wall_seconds','peak_process_rss_bytes','peak_gpu_memory_bytes',
            'incremental_network_transfer_bytes','energy')})
    return '\n'.join(lines),value

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('spec','raw-root','report','interpretation'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();require(not args.report.exists() and not args.interpretation.exists(),'Exclusive outputs required')
    spec=json.loads(args.spec.read_text(encoding='utf-8'));root=args.raw_root
    complete=json.loads((root/'complete.json').read_text(encoding='utf-8'))
    pairings={}
    for row in spec['rows']:
        raw=json.loads((root/row['row_id']/'pairing.json').read_text(encoding='utf-8'))
        # Full fixed parent scores remain in original artifacts; report tables
        # retain their independently computed parent-first metrics only.
        pairings[row['row_id']]={key:value for key,value in raw.items() if key!='parents'}
        del raw
    text,value=assemble(spec,complete,pairings,source=str(root))
    for target in (args.report,args.interpretation):target.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x',encoding='utf-8') as stream:stream.write(text)
    with args.interpretation.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')

if __name__=='__main__':main()
