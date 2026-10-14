from . import design as d
from experiments.cvs_identity_mixstyle50 import predict as parent

def source_proof(c):return parent.source_proof(c,design=d)
def own_model(c,device):return parent.own_model(c,device,design=d)
def baseline(c,ids):return parent.baseline(c,ids,design=d)
def preflight(row):return parent.preflight(row,design=d)
def predict(row):return parent.predict(row,design=d)
