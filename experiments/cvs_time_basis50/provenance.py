from . import design as d,model
from experiments.cvs_feature_sequence50 import provenance as shared
def source_proof(c):return shared.source_proof(c,design=d,report_validator=model.validate_report)
def own_model(c,device):return shared.own_model(c,device,design=d,proof=source_proof,builder=model.build)
