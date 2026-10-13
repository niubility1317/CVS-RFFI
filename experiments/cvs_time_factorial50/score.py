from . import design as d
from .predict import preflight
from experiments.cvs_feature_sequence50 import score as shared
def score(row):return shared.score(row,design=d,prediction_preflight=preflight)
def finish():
 result=shared.finish(design=d)
 idx={(r['arm'],r['model_seed']):r['accuracy'] for r in result['records'] if r['dimension']=='overall'}
 result['interaction']={str(s):idx['LONG_SILU',s]-idx['COMPACT_SILU',s]-idx['LONG_LINEAR',s]+idx['COMPACT_LINEAR',s] for s in d.SEEDS}
 d.write(d.BASE/'analysis.json',result);return result
