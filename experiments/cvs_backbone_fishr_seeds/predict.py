from . import design as d,model as api
from experiments.cvs_signature_stats50 import predict as p
def source_proof(c):return p.source_proof(c,design=d)
def own_model(c,device):return p.own_model(c,device,design=d,build_fn=api.build)
def baseline(c,ids):return p.baseline(c,ids,design=d)
def predict(row):return p.predict(row,design=d,build_fn=api.build)
def preflight(row):return p.preflight(row,design=d,build_fn=api.build)
