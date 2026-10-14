"""Combine independently recounted artifacts; never used by training or selection."""
import argparse,json
from pathlib import Path

def combine(first,remaining):
 a,b=first['audit'],remaining['audit'];combined=remaining['combined_analysis']
 if any(v.get('status')!='VERIFIED' for v in (first,remaining,a,b,combined)):raise ValueError('Verified completed artifacts required')
 if a['models']!=4 or b['models']!=8 or a['epoch_records_parsed']!=200 or b['epoch_records_parsed']!=400 or a['new_step_records_parsed']!=33300 or b['new_step_records_parsed']!=66600:raise ValueError('Full fixed training matrix not independently recounted')
 if combined['model_seeds']!=[2026092701,2026092702,2026092703] or combined['models']!=12 or combined['newly_trained_models']!=9 or combined['default_decision_count']!=2016000 or combined['metric_records']!=216 or not combined['truth_last']:raise ValueError('Full predefined default-head matrix incomplete')
 rows=a['rows']+b['rows'];errors=a['error_transitions']+b['error_transitions'];features=first['source_features']['rows']+remaining['source_features']['rows'];pairs=first['source_features']['paired']+remaining['source_features']['paired'];observations=first['actual_training_observations']+remaining['actual_training_observations']
 if len(rows)!=12 or len(features)!=9 or len(pairs)!=3 or len(observations)!=9 or len(errors)!=15 or len({r['row_id'] for r in rows})!=12:raise ValueError('Matched matrix rows/features/errors missing or repeated')
 if any(r['feature_boundary']['maximum_export_difference']!=0 or not r['feature_boundary']['class_W_zeroing_leaves_features_identical'] or not r['feature_boundary']['old_class_W_removed_from_export'] or r['feature_boundary']['head_fit'] for r in features):raise ValueError('Actual default feature boundary failed')
 if any(o['independent_auxiliary_steps']!=2750 or o['independent_layer_cohort_gradients']!=660 or o['observation']['head_fit_updates'] or o['observation']['U_truth_read'] for o in observations):raise ValueError('Actual auxiliary evidence incomplete')
 transitions=[]
 for arm,control in {(r['arm'],r['control']) for r in errors}:
  rr=sorted([r for r in errors if r['arm']==arm and r['control']==control],key=lambda r:r['seed'])
  if [r['seed'] for r in rr]!=[2026092701,2026092702,2026092703]:raise ValueError('Complete error-pair seeds required')
  corrected=sum(r['corrected_errors'] for r in rr);introduced=sum(r['introduced_errors'] for r in rr);transitions.append(dict(arm=arm,control=control,corrected_errors=corrected,introduced_errors=introduced,net_corrected=corrected-introduced,count=504000,per_seed=rr))
 return dict(status='VERIFIED',analysis=combined,source_rows=rows,source_features=features,source_feature_pairs=pairs,actual_training_observations=observations,error_transitions=sorted(transitions,key=lambda r:(r['arm'],r['control'])),counts=dict(models=12,new_models=9,reused_models=3,new_model_epochs=450,new_optimizer_updates=99900,historical_model_epochs=150,historical_optimizer_updates=33300,source_epoch_records=600,default_decisions=2016000,metric_records=216,paired_error_records=15,actual_auxiliary_steps=24750,extra_source_packet_forwards=2227500,actual_gradient_layer_cohorts=5940,saved_gradient_vectors=23760),head_fit_updates=0,U_truth_read=False,no_new_target_forwards=True,goal_84=combined['goal_84'],scope='Complete fixed default-head clean three-seed research matrix; source seen RX and actual feature boundary evidence, no new-TX/Phase2/satellite claim')

def main():
 p=argparse.ArgumentParser();p.add_argument('--first',type=Path,required=True);p.add_argument('--remaining',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=combine(json.loads(a.first.read_text(encoding='utf8')),json.loads(a.remaining.read_text(encoding='utf8')));a.output.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status=v['status'],counts=v['counts'],goal_84=v['goal_84'])))
if __name__=='__main__':main()
