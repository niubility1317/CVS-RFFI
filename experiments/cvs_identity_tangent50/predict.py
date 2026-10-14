from . import design as d
from .provenance import source_proof,own_model
from experiments.cvs_feature_sequence50 import predict as shared
def predict(row):return shared.predict(row,design=d,proof=source_proof,load=own_model)
def preflight(row):return shared.preflight(row,design=d,proof=source_proof)
