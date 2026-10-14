from . import design as d
from experiments.cvs_signature_stats50 import predict as shared
def source_proof(c):return shared.source_proof(c,design=d)
def own_model(c,device):return shared.own_model(c,device,design=d)
def baseline(c,ids):return shared.baseline(c,ids,design=d)
def preflight(row):return shared.preflight(row,design=d)
def predict(row):return shared.predict(row,design=d)
