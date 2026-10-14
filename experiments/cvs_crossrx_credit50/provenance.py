"""Same-data own-scratch source freeze, original head, actual mechanism evidence."""
from . import design as d,model
from experiments.cvs_feature_sequence50 import provenance as shared

def report_validator(report,c):
 if (report['arm']!='SIG_TIME' or report['classifier_changed'] or not report['default_head_only']
  or report['separate_head_training'] or report['layer']!='t2' or report['output_embedding']!=160
  or report['extra_parameters']!=6*report['local_channels']+18400):
  raise ValueError('Actual fixed SIG_TIME/default head differs')

def source_proof(c):
 p=shared.source_proof(c,design=d,report_validator=report_validator)
 e=d.read(p/'source_training_mechanism.json')
 if (e.get('epoch')!=50 or e.get('auxiliary_calls')!=624 or e.get('head_fit_updates')!=0
  or e.get('extra_actual_optimizer_steps')!=0 or e.get('U_truth_read') or e.get('target_access')
  or not e.get('default_head_joint_training') or not e.get('original_U_sampler_unchanged')):
  raise ValueError('Actual source mechanism/budget/default-head evidence incomplete')
 if e.get('optimizer_probe_steps')!=list(d.PROBE_STEPS):raise ValueError('Actual optimization probes incomplete')
 return p

def own_model(c,device):
 return shared.own_model(c,device,design=d,proof=source_proof,builder=model.build)
