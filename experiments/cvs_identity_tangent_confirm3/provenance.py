from . import design as d,model
from experiments.cvs_feature_sequence50 import provenance as shared
def source_proof(c):
 p=shared.source_proof(c,design=d,report_validator=model.validate_report)
 if not c['baseline_reuse']:
  v=d.read(p/'tangent_execution.json')
  if v!=dict(auxiliary_calls=2498,extra_L_forward_packets=449640,teacher_intervention=False,U_intervention=False,head_fit_updates=0,source_only=True):raise ValueError('Actual tangent budget differs')
 return p
def own_model(c,device):return shared.own_model(c,device,design=d,proof=source_proof,builder=model.build)
