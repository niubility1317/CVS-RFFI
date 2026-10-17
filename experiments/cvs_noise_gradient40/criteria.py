"""Primary classification-margin parameter gradient;LS calibration is secondary."""
def decide(cells):
 if len(cells)!=2 or sorted(c.get('draw') for c in cells)!=[0,1]:raise ValueError('Two fixed low-urban draws required')
 rows=[];checks=[]
 for c in sorted(cells,key=lambda x:x['draw']):
  if c.get('objective')!='margin' or c.get('parameter')!='head_W':raise ValueError('Primary uses actual W gradient of margin,not feature or LS gradient')
  n=c['N'];a=c['negative_full'];b=c['negative_no_noise'];v=c['valid_full'];w=c['valid_no_noise']
  if any(type(k)is not int for k in [n,a,b,v,w]) or n!=900 or not 0<=a<=v<=n or not 0<=b<=w<=n:raise ValueError('All900 legal L observations stay in denominator')
  ok=v/n>=.95 and w/n>=.95 and a/n>=.20-1e-12 and (a-b)/n>=.10-1e-12;checks.append(ok);rows.append(dict(draw=c['draw'],N=n,valid_full=v,valid_no_noise=w,negative_full=a,negative_no_noise=b,reduction_pp=100*(a-b)/n))
 return dict(status='SOURCE_GRADIENT_DECIDED',margin_noise_conflict_evidence=all(checks),draw_checks=checks,primary_cells=rows,source_only=True,V_backprop=False,target_access=False,training_updates=0,head_fit_updates=0,automatic_training=False,Ag_proved=False,goal_complete=False)
