"""Post-score descriptive paired contrasts, including prespecified interactions."""
import statistics
from . import design as d

INTERACTIONS={
 'SAM_by_physical':dict(SAM_PHYS=1,PHYS=-1,SAM_SSL=-1,SSL=1),
 'SAM_by_risk':dict(SAM_RANK=1,RANK=-1,SAM_PHYS=-1,PHYS=1),
}

def contrast(records,coefficients,dimension,stratum):
 lookup={}
 for r in records:
  if r['dimension']!=dimension or r['stratum']!=stratum or r['arm'] not in coefficients:continue
  key=(r['arm'],r['model_seed'])
  if key in lookup:raise ValueError('Duplicate arm/seed/stratum in paired analysis')
  if r['model_seed'] not in d.SEEDS:raise ValueError('Unexpected seed in paired analysis')
  lookup[key]=r['accuracy']
 seeds=[s for s in d.SEEDS if all((a,s) in lookup for a in coefficients)]
 values=[dict(seed=s,difference=sum(k*lookup[(a,s)] for a,k in coefficients.items())) for s in seeds]
 diffs=[x['difference'] for x in values]
 return dict(coefficients=coefficients,dimension=dimension,stratum=stratum,seed_count=len(seeds),complete_three_seeds=len(seeds)==len(d.SEEDS),missing_seeds=[s for s in d.SEEDS if s not in seeds],per_seed=values,mean_difference=statistics.mean(diffs) if diffs else None,sd_difference=statistics.stdev(diffs) if len(diffs)>1 else None,positive_seeds=sum(x>0 for x in diffs),unit='accuracy fraction; multiply by100 for percentage points',inferential_significance_claim=False)

def analyze(result):
 records=result['records']+result['day_records']+result['parent_records']
 strata=sorted({(r['dimension'],r['stratum']) for r in records})
 pairs=[dict(name=a+'-'+b,**contrast(records,{a:1,b:-1},dim,st)) for a,b in d.PAIRS for dim,st in strata]
 interactions=[dict(name=name,**contrast(records,coeff,dim,st)) for name,coeff in INTERACTIONS.items() for dim,st in strata]
 return dict(run_id=d.RUN,read_at=result['read_at'],status=result['status'],pairs=pairs,interactions=interactions,scope='prespecified paired seed accuracy contrasts; source training and selection never consume this output',missing_rows=result['missing_rows'])

def markdown(value):
 lines=['# SAM的配对增量与交互差','',f"状态 {value['status']}。只读取已完成的独立评分；每项仅使用全部组成方案均已完成的同seed配对，不混用不完整种子。差值单位为百分点。",'', '|比较|种子1|种子2|种子3|配对均值|配对SD|种子数|','|---|---:|---:|---:|---:|---:|---:|']
 for r in value['pairs']+value['interactions']:
  if r['dimension']!='overall':continue
  lookup={x['seed']:x['difference'] for x in r['per_seed']}
  values=['N/A' if s not in lookup else f'{100*lookup[s]:+.4f}' for s in d.SEEDS]
  mean='N/A' if r['mean_difference'] is None else f"{100*r['mean_difference']:+.4f}"
  sd='N/A' if r['sd_difference'] is None else f"{100*r['sd_difference']:.4f}"
  lines.append(f"|{r['name']}|"+'|'.join(values)+f"|{mean}|{sd}|{r['seed_count']}/3|")
 lines+=['', 'SAM_by_physical = (SAM_PHYS−PHYS)−(SAM_SSL−SSL)。SAM_by_risk = (SAM_RANK−RANK)−(SAM_PHYS−PHYS)。正交互只说明本批中SAM增量随该配方改变，不能单独证明整体超过基座或达到84%，也不等同统计显著。','', '完整RX、TX、日期分层配对与缺失种子保存在 evidence/paired_interactions.json。主比较SAM_RANK−RANK，所有固定配方均报告；本分析不选择性重跑，不回流设计或训练。']
 return '\n'.join(lines)+'\n'

def checks():
 # Different control seeds cannot form a spurious pair.
 rec=lambda a,s,v:dict(arm=a,model_seed=s,accuracy=v,dimension='overall',stratum='all')
 s1,s2,s3=d.SEEDS
 rows=[rec('SAM_SSL',s1,.85),rec('SSL',s2,.80)]
 assert contrast(rows,{'SAM_SSL':1,'SSL':-1},'overall','all')['seed_count']==0
 # Analytically known interaction: .04-.01=.03 across all matched seeds.
 rows=[rec(a,s,v) for s in d.SEEDS for a,v in [('SAM_PHYS',.86),('PHYS',.82),('SAM_SSL',.81),('SSL',.80)]]
 v=contrast(rows,INTERACTIONS['SAM_by_physical'],'overall','all')
 assert v['complete_three_seeds'] and abs(v['mean_difference']-.03)<1e-12 and abs(v['sd_difference'])<1e-12
 v=contrast(rows[:-1],INTERACTIONS['SAM_by_physical'],'overall','all')
 assert v['seed_count']==2 and not v['complete_three_seeds'] and v['missing_seeds']==[s3]
 try:contrast(rows+[rows[0]],INTERACTIONS['SAM_by_physical'],'overall','all')
 except ValueError:pass
 else:raise AssertionError('Duplicate was accepted')
 return dict(status='PASS',matched_seeds=True,analytic_interaction=True,missing_seed_not_imputed=True,duplicates_rejected=True,scope='synthetic post-score arithmetic only, no experiment scores')

if __name__=='__main__':
 import json
 print(json.dumps(checks()))
