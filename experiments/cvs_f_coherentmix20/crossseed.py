"""Two predeclared source seeds,paired metric means; no new model/IQ/target."""
import argparse,csv,json,math,statistics
from pathlib import Path
from .analyze import ARMS,VIEWS,NEW,read,dump

def summarize(first,confirm):
 first=Path(first);confirm=Path(confirm);roots=[first,confirm];reports=[read(p/'analysis_summary.json') for p in roots];assert {r['model_seed'] for r in reports}=={2026092701,2026092702}
 assert all(r['status']=='ANALYZED' and r['source_only'] and not r['target_access'] and not r['joint_goal_complete'] and r['all_L_U_sequences_identical'] and r['all_training_role_IDs_identical'] for r in reports)
 ordered=sorted(zip(roots,reports),key=lambda x:x[1]['model_seed']);reports=[r for _,r in ordered];data=[];args=[]
 for root,r in ordered:
  seed=r['model_seed'];src=root/('CROSS_RX-RX1-s'+str(seed))/'source';contract=read(src/'source_contract.json');p=read(src/'source_readouts/clean.json');channel=read(src/'actual_channel.json');args.append(read(src/'actual_native_args.json'))
  data.append(dict(roles=contract['role_ids'],ids=p['physical_ids'],labels=p['labels'],channel=channel['actual_configs'],physics={v:read(src/'source_readouts'/(v+'.json'))['physics'] for v in VIEWS[1:]}))
 assert data[0]==data[1]
 differences=sorted(k for k in set(args[0])|set(args[1]) if args[0].get(k)!=args[1].get(k));assert set(differences)<=set(('seed','run_id','candidate_id','output_dir')) and args[0]['seed']!=args[1]['seed']
 def stat(values):return dict(n=len(values),mean=statistics.fmean(values),sample_std=statistics.stdev(values),values=values)
 means={a:{v:{metric:stat([100*r['scores'][a][v][metric] for r in reports]) for metric in ('accuracy','macro_F1')} for v in VIEWS} for a in ARMS};paired={ref:{v:{metric:stat([r['paired'][ref][v][metric] for r in reports]) for metric in ('accuracy_gain_pp','macro_F1_gain_pp')} for v in VIEWS} for ref in ARMS[:2]}
 pass_all=all(r['qualification']['qualified'] for r in reports);direction=all(r['paired']['D_raw']['clean']['accuracy_gain_pp']>0 for r in reports)
 value=dict(status='ANALYZED_TWO_SOURCE_SEEDS',source_only=True,target_access=False,held_source_RX=1,held_unique_L=1260,seeds=[r['model_seed'] for r in reports],rows=[dict(model_seed=r['model_seed'],scores=r['scores'],qualification=r['qualification'],suburban_clean_gaps_pp=r['suburban_clean_gaps_pp'],execution=r['execution']) for r in reports],mean_metrics_percent=means,paired_gain_pp=paired,first_and_confirmation_pass=pass_all,clean_positive_direction_repeated=direction,fixed_operator_close=not pass_all,source_confirmation_scope='Same held sourceRX1 under2model/globalordinary seeds;not fresh independent target or allRX DG proof',same_physical_roles_and_held_packets=True,same_all_six_physics_records=True,native_argument_differences=differences,not_independent_physical_repetitions=True,uncertainty='n2 sample standarddeviation acrossmodelseeds;no pseudo2520-independent-sample CI',total_new_model_epochs=sum(r['new_epochs'] for r in reports),total_new_optimizer_updates=sum(r['new_updates'] for r in reports),total_new_NN_packets=sum(r['new_NN_packets'] for r in reports),reused_reference_epochs=sum(r['reused_reference_epochs'] for r in reports),reused_reference_updates=sum(r['reused_reference_updates'] for r in reports),reused_reference_NN=sum(r['reused_reference_NN'] for r in reports),analysis_NN=0,analysis_IQ_reads=0,head_fit_updates=0,joint_goal_complete=False,Ag_allRX_or_target_proved=False,on_fail='Close fixed coherent interpolation,no seed03/RX/fraction/window/budget rescue',on_pass='Only repeated sourceRX1 clean evidence;needs separatejoint factorial andformalclean+sixpractical',missing=['Other heldsourceRX confirmation','B00/B10/B01/B11 andinteraction','Formal fixedtarget clean+sixpractical','Jointcrossseed cleanpositive,low>60,suburbannearclean'])
 dump(confirm/'crossseed_analysis.json',value)
 rows=[]
 for r in reports:
  for a in ARMS:
   for v in VIEWS:rows.append(dict(model_seed=r['model_seed'],arm=a,view=v,accuracy_pct=100*r['scores'][a][v]['accuracy'],macroF1_pct=100*r['scores'][a][v]['macro_F1']))
 with (confirm/'crossseed_scores.csv').open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 return value
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--first',required=True);p.add_argument('--confirm',required=True);a=p.parse_args();r=summarize(a.first,a.confirm);print(json.dumps(dict(status=r['status'],bothpass=r['first_and_confirmation_pass'],direction=r['clean_positive_direction_repeated'],paired_clean_gain=r['paired_gain_pp']['D_raw']['clean'])))
