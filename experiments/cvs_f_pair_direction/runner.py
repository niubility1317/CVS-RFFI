"""CPU-only reuse of original BASE3 cached sourceV features; never loads a model."""
import argparse,os,time
from pathlib import Path
import numpy as np
from . import analysis as a
ROOT=Path(__file__).resolve().parents[2]
def run(args):
 from .checks import smoke
 smoke()
 cache=Path(args.cache).resolve();out=Path(args.output).resolve()
 if os.name!='nt' or Path.cwd().resolve()!=ROOT or (ROOT/'release_commit.txt').read_text(encoding='utf8').strip()!=args.commit or out.name!=a.RUN:raise ValueError('Fixed native immutable analysis release/output only')
 prior=a.read(cache/'completion.json');execution=a.read(cache/'execution.json')
 if prior!=dict(status='ARTIFACTS_COMPLETE',run_id=a.CACHE_RUN,rows=3,source_only=True,head_fit_updates=0,optimizer_updates=0,target_reads=0) or execution.get('reference_code_commit')!=a.REFERENCE_COMMIT or execution.get('data_role')!='V' or execution.get('source_only') is not True:raise ValueError('Reuse complete original validated source cache only')
 for seed in a.SEEDS:a.scope('V',seed,cache.name);a.metadata(a.read(cache/('BASE-s'+str(seed))/'source_geometry.json'),seed)
 out.mkdir(parents=True,exist_ok=False);a.write(out/'execution.json',dict(run_id=a.RUN,commit=args.commit,pid=os.getpid(),cwd=str(ROOT),cache=str(cache),source_role='V',CPU_only=True,new_model_forwards=0,checkpoint_loads=0,target_reads=0,U_TX_truth_reads=0,head_fit_updates=0,optimizer_updates=0))
 records=[];started=time.perf_counter()
 for seed in a.SEEDS:
  tick=time.perf_counter();row='BASE-s'+str(seed)
  with np.load(cache/row/'source_features.npz',allow_pickle=False) as data:
   if set(data.files)!={'features','y','rx','day'}:raise ValueError('Only existing sourceV feature cache interface')
   mu=a.centers(data['features'],data['y'],data['rx'],data['day'])
  result=a.analyze(mu,seed);result.update(row_id=row,source_role='V',source_count=27000,seconds=time.perf_counter()-tick,source_code_commit=a.REFERENCE_COMMIT,new_model_forwards=0,target_reads=0,head_fit_updates=0)
  (out/row).mkdir();a.write(out/row/'source_direction.json',result);records.append(result);print('SOURCE_DIRECTION_COMPLETE '+row,flush=True)
 decision=a.decision(records)
 result=dict(status='ARTIFACTS_COMPLETE',run_id=a.RUN,code_commit=args.commit,reference_run=a.REFERENCE_RUN,cache_run=a.CACHE_RUN,records=records,source_decision=decision,source_role='V',unique_source_samples=27000,cached_model_feature_rows=81000,new_IQ_reads=0,new_model_forwards=0,new_checkpoint_loads=0,head_fit_updates=0,optimizer_updates=0,target_reads=0,U_TX_truth_reads=0,seconds=time.perf_counter()-started,limitation='Frozen source-only diagnostic on already exposed benchmark; all3 BASE caches fixed. Geometric premise test is not a performance result. No new target model/score or fitted classifier.')
 a.write(out/'result.json',result);a.write(out/'completion.json',dict(status='ARTIFACTS_COMPLETE',rows=3,source_only=True,target_reads=0,model_forwards=0,optimizer_updates=0))
 lines=['# 跨RX判别方向：源域三种子诊断','', '只复用原BASE三种子的sourceV缓存。每模型27000条，源RX1/3/4/6/8；不加载checkpoint、不新跑网络、不读取IQ/U真值/目标，不训练或拟合分类器。','', '对单位joint160，按TX/RX求均值，固定弱对14-7与20-19：δ_RX=μ_14-7,RX−μ_20-19,RX。比较五个RX的10对差向量余弦；同一RX的类中心公共偏移可在δ中代数抵消，不表示消除了真实非线性信道。','', '预先固定：每个seed的弱对平均方向余弦≤0.90，且比其他14类对中位数至少低0.05，全部类对均有10对有效RX方向；三seed都满足才认为该方向路线的前提有支持。这个阈值用于证伪几何前提，不是性能门槛。','', '|seed|弱对平均方向余弦|其他类对中位数|差额|cross/self|前提通过|','|---|---:|---:|---:|---:|---|']
 for r in records:
  w=r['weak_pair'];lines.append('|'+str(r['model_seed'])+'|'+format(w['mean_direction_cosine'],'.6f')+'|'+format(r['median_other_pair_direction_cosine'],'.6f')+'|'+format(r['weak_cosine_deficit_to_other_pair_median'],'.6f')+'|'+format(w['cross_to_self'],'.6f')+'|'+str(r['prospective_direction_route_qualified'])+'|')
 lines+=['','三种子前提判定：'+str(decision['PASS'])+'。','', '若FAIL，不降低阈值或把源域几何结果当作目标收益，不启动该预设方向一致性训练路线；若PASS，也只说明值得进入机制实验，仍需固定三种子训练及clean独立测试证明性能。所有类对/负结果保留。']
 with (out/'results.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
 print('SOURCE_DIRECTION_ALL_COMPLETE',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--commit',required=True);run(p.parse_args())
