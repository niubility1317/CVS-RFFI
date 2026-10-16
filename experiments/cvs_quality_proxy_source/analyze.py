"""Independent complete sufficient-statistic recount;stdlib,no IQ/CP/NN."""
import argparse,gzip,json,math,statistics
from pathlib import Path
from collections import Counter,defaultdict
SCENES=('practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban');PRIMARY=SCENES[-2:];FIT=(3,4,6,8);ROW='quality-proxy-fitRX3_4_6_8'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def near(a,b):
 if not math.isfinite(a+b) or not math.isclose(a,b,abs_tol=2e-10,rel_tol=2e-10):raise ValueError('Independent power/quality recount differs')
def quality(power,noise):
 if power<=0 or noise<=0 or not math.isfinite(power+noise):raise ValueError('Positive finite power required')
 q=10*math.log10(power/noise);return q,('locked' if q>=15 else 'degraded' if q>=5 else 'unlocked'),10**min(1.,max(0.,(20-q)/20))
def calc(z):
 for key in ('input_power','propagation_energy','noise_variance_reference','filtered_power_whole','filtered_power_interior'):
  if z[key]<=0 or not math.isfinite(z[key]):raise ValueError('Finite positive saved powers required')
 if (z['RX'] not in FIT or z['day'] not in (1,2,3) or z['TX'] not in range(6) or z['draw'] not in (0,1) or z['scene'] not in SCENES or z['session']!='rx%d:day%d:eq1'%(z['RX'],z['day']) or z['receiver']['session_id']!=z['session'] or z['native_process_calls']!=5 or z['interior_start']!=z['kernel_length']-1 or z['interior_count']!=256-z['interior_start'] or z['numerical_filter_latency_samples']!=24 or z['boundary']!='synthetic_reflect' or z['noise_boundary']!='synthetic_reflect' or z['true_total_snr_known'] or not z['same_H'] or not z['same_noise_draw_stream']):raise ValueError('Fixed source/native causal physics legality differs')
 near(z['noise_variance_reference'],.001)
 ps=(z['input_power']*z['propagation_energy'],z['filtered_power_whole'],z['filtered_power_interior'])
 for name,p in zip(('native','whole','interior'),ps):
  q,status,scale=quality(p,z['noise_variance_reference']);a=z[name];near(q,a['quality_snr_db']);near(scale,a['scale'])
  if a['lock']!=status or a['EQ'] or not a['IQ_compensation'] or a['phase_tracking']!=(status!='unlocked'):raise ValueError('Native quality-to-receiver map differs')
 for name in ('native','whole','interior','fixed'):
  a=z[name];near(a['signal_to_added_noise_db'],10*math.log10(a['pre_agc_signal_power']/a['pre_agc_noise_power']))
  if not all(math.isfinite(v) for v in (a['frequency_hz'],a['agc_db'],a['output_power'])):raise ValueError('Finite CFO/AGC/output statistic required')
 near(z['fixed']['quality_snr_db'],z['whole']['quality_snr_db'])
 for k in ('lock','scale','frequency_hz','phase_tracking','EQ','IQ_compensation','agc_db','agc_limited','pre_agc_signal_power','pre_agc_noise_power','output_power'):
  if z['fixed'][k]!=z['native'][k]:raise ValueError('Fixed receiver compensation changed')
 if z['proxy_replay_MSE']!=0 or z['fixed_receiver_MSE']!=0 or z['whole_total_MSE']<0 or z['interior_total_MSE']<0 or not math.isfinite(z['whole_total_MSE']+z['interior_total_MSE']):raise ValueError('Exact sanity control or finite total-effect scalar differs')
 return z['whole']['quality_snr_db']-z['native']['quality_snr_db']
def summarize(rows):
 n=len(rows);bias=[calc(z) for z in rows];flips=sum(z['whole']['lock']!=z['native']['lock'] for z in rows);robust=sum(z['whole']['lock']==z['interior']['lock'] and z['whole']['lock']!=z['native']['lock'] for z in rows)
 return dict(count=n,median_bias_db=statistics.median(bias),median_abs_bias_db=statistics.median(abs(v) for v in bias),min_bias_db=min(bias),max_bias_db=max(bias),whole_lock_flips=flips,whole_lock_flip_fraction=flips/n,interior_lock_flips=sum(z['interior']['lock']!=z['native']['lock'] for z in rows),robust_same_flip_count=robust,boundary_robust_flip_fraction=robust/flips if flips else 0.,median_whole_vs_interior_abs_db=statistics.median(abs(z['whole']['quality_snr_db']-z['interior']['quality_snr_db']) for z in rows),min_interior_count=min(z['interior_count'] for z in rows),mean_whole_total_MSE=statistics.mean(z['whole_total_MSE'] for z in rows),mean_interior_total_MSE=statistics.mean(z['interior_total_MSE'] for z in rows),max_fixed_receiver_MSE=max(z['fixed_receiver_MSE'] for z in rows),mean_abs_residual_CFO_change_hz=statistics.mean(abs(z['whole']['frequency_hz']-z['native']['frequency_hz']) for z in rows),whole_AGC_limited=sum(z['whole']['agc_limited'] for z in rows),native_AGC_limited=sum(z['native']['agc_limited'] for z in rows),white_to_whole_lock={a+'->'+b:sum(z['native']['lock']==a and z['whole']['lock']==b for z in rows) for a in ('locked','degraded','unlocked') for b in ('locked','degraded','unlocked')})
def identical(a,b):
 if isinstance(a,dict):return set(a)==set(b) and all(identical(a[k],b[k]) for k in a)
 if isinstance(a,list):return len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
 if isinstance(a,(str,bool)) or a is None:return a==b
 return math.isclose(a,b,abs_tol=2e-10,rel_tol=2e-10)
def recount(root):
 root=Path(root);out=root/ROW;s=read(out/'summary.json');c=read(out/'resolved_config.json');done=read(root/'completion.json');actual=read(out/'actual_engine.json');roles=read(out/'data_roles.json');sm=read(root/'native_noquery_smoke.json')
 if done['status']!='ARTIFACTS_COMPLETE' or (root/'failure.json').exists() or sm['status']!='PASS' or sm['native_process_calls']!=10 or any(s[k]!=0 for k in ('new_models','checkpoint_loads','model_forwards','optimizer_updates','teacher_forwards','head_fit_updates')) or s['target_access'] or s['formal_test_changed'] or s['automatic_training'] or s['goal_complete']:raise ValueError('Complete no-model source diagnostic required')
 for cfg in actual['configs'].values():
  if (cfg['fs_hz'],cfg['processing_route'],cfg['mode'],cfg['equalization_enabled'],cfg['reference_snr_db'],cfg['boundary'])!=(25000000,'residual','post_sync',False,30,'reflect') or not all(cfg[k] for k in ('quality_adaptation_enabled','phase_tracking_enabled','iq_compensation_enabled')):raise ValueError('Original physical receiver config required')
 ids={json.dumps(v,sort_keys=True) for v in roles['effective_L_IDs']}
 if len(ids)!=5040 or roles['U_values_used'] or roles['V_values_used'] or roles['held_RX1_values_used'] or roles['target_access']:raise ValueError('Exact fitL-only actual roles required')
 groups=defaultdict(list);seen=set();receivers={};cells=defaultdict(list);delta_points=[]
 with gzip.open(out/'quality_records.jsonl.gz','rt',encoding='utf8') as f:
  for line in f:
   z=json.loads(line);bias=calc(z);pid=json.dumps(z['physical_id'],sort_keys=True);key=(z['draw'],z['scene'],pid)
   if pid not in ids or key in seen:raise ValueError('Duplicate/nonfit physical source record')
   seen.add(key);groups[str(z['draw'])+'/'+z['scene']].append(z);cells[z['draw'],z['scene'],z['TX'],z['RX'],z['day']].append(z)
   if receivers.setdefault(z['session'],z['receiver'])!=z['receiver']:raise ValueError('RX hardware session changed withTX/scene/draw')
   delta_points.append(dict(draw=z['draw'],scene=z['scene'],state=z['state'],RX=z['RX'],TX=z['TX'],day=z['day'],quality_proxy_db=z['native']['quality_snr_db'],bias_db=bias,whole_vs_interior_db=z['whole']['quality_snr_db']-z['interior']['quality_snr_db'],lock_proxy=z['native']['lock'],lock_whole=z['whole']['lock'],lock_interior=z['interior']['lock'],interior_count=z['interior_count'],kernel_length=z['kernel_length'],residual_CFO_native_hz=z['native']['frequency_hz'],residual_CFO_whole_hz=z['whole']['frequency_hz'],AGC_native_db=z['native']['agc_db'],AGC_whole_db=z['whole']['agc_db'],AGC_native_limited=z['native']['agc_limited'],AGC_whole_limited=z['whole']['agc_limited'],scale_native=z['native']['scale'],scale_whole=z['whole']['scale'],signal_to_added_noise_native_db=z['native']['signal_to_added_noise_db'],signal_to_added_noise_whole_db=z['whole']['signal_to_added_noise_db']))
 if len(seen)!=60480 or len(groups)!=12 or any(len(v)!=5040 for v in groups.values()) or len(cells)!=864 or any(len(v)!=70 for v in cells.values()) or len(receivers)!=12:raise ValueError('Complete2draw6scene72cell source matrix required')
 aggregate={k:summarize(v) for k,v in groups.items()};cellrows=[dict(draw=k[0],scene=k[1],TX=k[2],RX=k[3],day=k[4],**summarize(v)) for k,v in cells.items()]
 cellkey=lambda z:(z['draw'],z['scene'],z['TX'],z['RX'],z['day'])
 if not identical(aggregate,s['groups']) or not identical(sorted(cellrows,key=cellkey),sorted(s['source_cells'],key=cellkey)) or not identical(receivers,s['receiver_sessions']):raise ValueError('Full source aggregation differs')
 minimum=min(z['interior_count'] for z in delta_points);checks={'all_interior_min128':minimum>=128}
 for dr in range(2):
  for scene in PRIMARY:
   g=aggregate[str(dr)+'/'+scene];tag=str(dr)+'_'+scene;checks.update({tag+'_magnitude':g['median_abs_bias_db']>=1.,tag+'_lock_flips':g['whole_lock_flip_fraction']>=.05,tag+'_boundary_median':g['median_whole_vs_interior_abs_db']<=.5,tag+'_robust_flips':g['boundary_robust_flip_fraction']>=.5})
 status='PASS_SOURCE_QUALITY_SWITCH_PREMISE_ONLY' if all(checks.values()) else 'FAIL_FIXED_QUALITY_PROXY_PREMISE_CLOSED'
 if checks!=s['checks'] or status!=s['status'] or s['physical_records']!=60480 or s['native_process_calls']!=302400 or s['source_physical_counts']!={'L':5040,'U':0,'V':0,'held_RX1':0,'target':0}:raise ValueError('Independent scientific decision/cost differs')
 log=(root/'dispatcher.log').read_text(encoding='utf8',errors='replace')
 if 'COMPLETE '+status not in log or any(x in log for x in ('Traceback','RuntimeError:','FloatingPointError:')) or len([z for z in log.splitlines() if z.startswith('SCENE_COMPLETE ')])!=12:raise ValueError('Full completed stdout required')
 result=dict(execution='VERIFIED',status=status,checks=checks,groups=aggregate,source_cells=cellrows,source_only=True,model_seeds=[],real_source_L=5040,physical_records=60480,native_process_calls=302400,synthetic_native_process_calls=10,new_models=0,checkpoint_loads=0,NN=0,optimizer_updates=0,teacher_forwards=0,head_fit_updates=0,target_access=False,automatic_training=False,goal_complete=False,wall_seconds=s['wall_seconds'],quality_strata={scene:{state:summarize([z for group in groups.values() for z in group if z['scene']==scene and z['state']==state]) for state in ('LOS','PARTIAL','BLOCKED') if any(z['state']==state for group in groups.values() for z in group if z['scene']==scene)} for scene in SCENES},full_stdout_lines=len(log.splitlines()),limits=['Saved sufficient statistics independently recounted;not originalIQ re-evaluation','Frozen-H signal-to-addednoise proxy,not true totalSNR or exact full route','No defaulthead accuracy/identity-protection/Ag orAs efficacy established','All256output samples retained;interior onlypower statistic;formaltest unchanged'])
 (root/'independent_analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf8');(root/'quality_plot_points.json').write_text(json.dumps(delta_points,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8');return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args();r=recount(a.input);print(json.dumps(dict(status=r['status'],checks=r['checks'],physical_records=r['physical_records'],NN=0)))
