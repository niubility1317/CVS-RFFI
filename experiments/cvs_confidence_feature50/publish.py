import argparse,json
from pathlib import Path
from . import design as d
from experiments.cvs_report_ssl import publish as transport
from experiments.cvs_pa_cross_delay.publish import configure as pa_configure,INSPECT as pa_inspect
INSPECT=pa_inspect.replace('experiments.cvs_pa_cross_delay','experiments.cvs_confidence_feature50').replace("'LEGACY50'","'OLD_GATE_COS50'")
INSPECT=INSPECT.replace("done['models']==9","done['models']==3").replace("done['new_trained_models']==6","done['new_trained_models']==2").replace("done['prediction_count']==1512000","done['prediction_count']==504000").replace('len(workers)==9','len(workers)==3')
def configure():
 pa_configure();transport.d=d
 transport.PATHS=list(dict.fromkeys([*transport.PATHS,'experiments/cvs_sigtime_pair50','experiments/cvs_neighbor_pair50','experiments/cvs_identity_mixstyle50','experiments/cvs_signature_stats50','experiments/cvs_signature_stats_seeds','experiments/cvs_feature_mean_teacher50','experiments/cvs_confidence_feature50','experiments/cvs_source_teacher_confidence','experiments/cvs_source_feature_ce_gradient','experiments/cvs_local_order50','experiments/cvs_signature_order50','experiments/cvs_source_signature_order','experiments/cvs_source_neighbor_pairs','experiments/cvs_trusted_weight50','experiments/cvs_source_layer_order','experiments/cvs_moment_area50','experiments/cvs_source_weight','experiments/cvs_feature_sequence50','experiments/cvs_neighbor_pair50','experiments/cvs_e20_budget_fishr','experiments/cvs_source_gate_audit','experiments/cvs_time_factorial/__init__.py','experiments/cvs_time_factorial/monitor.py','experiments/cvs_feature_multiscale50','experiments/cvs_transfer_features','experiments/cvs_feature_diagnostics']))
 transport.REMOTE=transport.REMOTE.replace('experiments.cvs_pa_cross_delay.preflight','experiments.cvs_confidence_feature50.preflight').replace('experiments.cvs_pa_cross_delay.dispatch','experiments.cvs_confidence_feature50.dispatch')
 transport.INSPECT=INSPECT
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args();configure()
 v=transport.inspect(a.output) if a.inspect else transport.publish(a.output)
 print(json.dumps(dict(status=v['status'],workers=len(v['workers']),epochs=[w['epoch_count'] for w in v['workers']],completion=v['completion'],failure=v['failure'])))
