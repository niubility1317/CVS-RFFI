import json,math,sys
from pathlib import Path
D=Path(sys.argv[1]);r=json.loads((D/'result.json').read_text(encoding='utf8'));rows=[];errs=[]
def independently(g,other):
 R=8
 def k(i,j):
  a,b=3*i,3*j
  return .25*(g[a][b]+g[a][b+other]+g[a+other][b]+g[a+other][b+other])
 return sum(k(i,i)+k(j,j)-2*k(i,j) for i in range(R) for j in range(i+1,R))/(R*(R-1))
for x in r['groups']:
 out=dict(scene=x['scene'],batch=x['batch'],count=x['count'])
 for block,key in [('full','full_gram'),('head','head_gram')]:
  g=x[key];assert len(g)==24 and all(len(t)==24 and all(math.isfinite(v) for v in t) for t in g)
  i,a=independently(g,1),independently(g,2);assert i>0 and a>=0
  expected=x[block];errs += [abs(i-expected['independent']),abs(a-expected['antithetic']),abs(a/i-expected['ratio'])]
  out[block]=dict(independent=i,antithetic=a,ratio=a/i)
 rows.append(out)
assert len(rows)==18 and len({(x['scene'],x['batch']) for x in rows})==18
summary={}
for scene in [*r['config']['scenes'],'all']:
 selected=[x for x in rows if scene=='all' or x['scene']==scene];summary[scene]={}
 for block in ('full','head'):
  i=sum(x[block]['independent'] for x in selected)/len(selected);a=sum(x[block]['antithetic'] for x in selected)/len(selected);ratio=a/i;summary[scene][block]=dict(independent=i,antithetic=a,ratio=ratio)
  errs += [abs(ratio-r['summary'][scene][block]['ratio'])]
c=dict(overall_full_at_most_point8=summary['all']['full']['ratio']<=.8,overall_head_at_most_point8=summary['all']['head']['ratio']<=.8,mid_full_below1=summary['practical_mid_urban']['full']['ratio']<1,low_full_below1=summary['practical_low_urban']['full']['ratio']<1)
assert all(c.values())==r['source_criteria_passed']
assert r['cost']['NN']==25922 and r['cost']['backward_calls']==432 and r['cost']['physical_channel_calls']==12960 and r['cost']['optimizer_updates']==0 and r['model_state_unchanged'] and not r['target_access']
assert max(errs)<1e-8
v=dict(status='INDEPENDENT_GRAM_RECOUNT_VERIFIED',run_id=r['config']['run_id'],source_only=True,source_criteria_passed=all(c.values()),decision=r['decision'],conditions=c,summary=summary,groups=rows,recount_max_absolute_error=max(errs),source_RX=r['config']['source_RX'],source_RX_holdout=None,parent_roles=r['source_parent_roles'],gradient_parameters=r['gradient_parameters'],all_model_parameters=r['all_model_parameters'],head_parameter=r['head_parameter'],cost=r['cost'],extra_r01_failed_cost=dict(CP=1,source_cache=1,NN=2,physical_channel_calls=90,gradients=0,updates=0),independent_analysis_cost=dict(IQ=0,CP=0,NN=0,updates=0),model_state_unchanged=True,Ag_gain_proved=False,As_identity_gain_proved=False,joint_goal_complete=False,target_access=False)
(D/'analysis.json').write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')
print(json.dumps(dict(status=v['status'],decision=v['decision'],conditions=c,summary=summary,recount_error=max(errs),cost=r['cost'])))
