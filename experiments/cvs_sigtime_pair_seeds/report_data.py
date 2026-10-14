import argparse,json
from pathlib import Path

def combine(first,remaining):
 a,b=first['audit'],remaining['audit'];c=remaining['combined_analysis']
 if any(v.get('status')!='VERIFIED' for v in (first,remaining,a,b,c)):raise ValueError('Verified completed fixedSIG pair artifacts required')
 if a['models']!=3 or b['models']!=6 or a['epoch_records_parsed']!=150 or b['epoch_records_parsed']!=300 or a['new_step_records_parsed']!=22200 or b['new_step_records_parsed']!=44400:raise ValueError('Complete fixedSIG pair matrix not independently recounted')
 if c['model_seeds']!=[2026092701,2026092702,2026092703] or c['models']!=9 or c['newly_trained_models']!=6 or c['default_decision_count']!=1512000 or c['metric_records']!=162 or not c['truth_last']:raise ValueError('Complete fixedSIG pair three-seed summary missing')
 rows=a['rows']+b['rows'];features=first['source_features']['rows']+remaining['source_features']['rows'];pairs=first['source_features']['paired']+remaining['source_features']['paired'];obs=first['actual_training_observations']+remaining['actual_training_observations'];errors=a['error_transitions']+b['error_transitions']
 if len(rows)!=9 or len(features)!=6 or len(pairs)!=3 or len(obs)!=6 or len(errors)!=9 or len({r['row_id'] for r in rows})!=9:raise ValueError('FixedSIG pair rows missing or repeated')
 if any(r['feature_boundary']['maximum_export_difference']!=0 or not r['feature_boundary']['class_W_zeroing_leaves_features_identical'] or not r['feature_boundary']['old_class_W_removed_from_export'] or r['feature_boundary']['head_fit'] for r in features):raise ValueError('Actual fixedSIG feature boundary failed')
 if any(o['independent_layer_gradient_records']!=1998 or o['observation']['steps']!=11100 or o['observation']['sampler_packets']!=963900 or o['observation']['head_fit_updates'] or o['observation']['U_truth_read'] for o in obs):raise ValueError('Actual fixedSIG SSL evidence incomplete')
 aggregate=[]
 for arm,control in {(r['arm'],r['control']) for r in errors}:
  rr=sorted([r for r in errors if r['arm']==arm and r['control']==control],key=lambda r:r['seed'])
  if [r['seed'] for r in rr]!=[2026092701,2026092702,2026092703]:raise ValueError('Complete paired seed set required')
  corrected=sum(r['corrected_errors'] for r in rr);introduced=sum(r['introduced_errors'] for r in rr);aggregate.append(dict(arm=arm,control=control,corrected_errors=corrected,introduced_errors=introduced,net_corrected=corrected-introduced,count=504000,per_seed=rr))
 return dict(status='VERIFIED',analysis=c,source_rows=rows,source_features=features,source_feature_pairs=pairs,actual_training_observations=obs,error_transitions=sorted(aggregate,key=lambda r:(r['arm'],r['control'])),counts=dict(models=9,new_models=6,reused_models=3,new_model_epochs=300,new_optimizer_updates=66600,historical_model_epochs=150,historical_optimizer_updates=33300,source_epoch_records=450,default_decisions=1512000,metric_records=162,paired_error_records=9,actual_gradient_layer_batches=11988,saved_gradient_vectors=35964,extra_model_forwards=0),head_fit_updates=0,U_truth_read=False,no_new_target_forwards=True,goal_84=c['goal_84'],integration_not_new_core_algorithm=True,scope='Complete fixedSIG network/defaulthead three-seed clean integration reference; no newTX/Phase2/satellite claim')

def main():
 p=argparse.ArgumentParser();p.add_argument('--first',type=Path,required=True);p.add_argument('--remaining',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=combine(json.loads(a.first.read_text(encoding='utf8')),json.loads(a.remaining.read_text(encoding='utf8')));a.output.write_text(json.dumps(v,ensure_ascii=False,indent=2)+chr(10),encoding='utf8');print(json.dumps(dict(status=v['status'],counts=v['counts'],goal_84=v['goal_84'])))
if __name__=='__main__':main()
