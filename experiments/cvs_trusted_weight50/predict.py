from . import design as d,model as api
from experiments.cvs_signature_stats50 import predict as parent
def source_proof(c):return parent.source_proof(c,design=d)
def own_model(c,device):return parent.own_model(c,device,design=d,build_fn=api.build)
def baseline(c,ids):return parent.baseline(c,ids,design=d)
def predict(row):return parent.predict(row,design=d,build_fn=api.build)
def preflight(row):return parent.preflight(row,design=d,build_fn=api.build)
