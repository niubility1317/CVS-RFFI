from . import design as d
from .predict import preflight
from experiments.cvs_feature_sequence50 import score as shared
def score(row):return shared.score(row,design=d,prediction_preflight=preflight)
def finish():
 result=shared.finish(design=d)
 import statistics as st
 index={(r['arm'],r['model_seed']):r['accuracy'] for r in result['records'] if r['dimension']=='overall'}
 values=[(index['LONG_RESP',s]-index['LONG_ZERO',s])-(index['COMPACT_RESP',s]-index['COMPACT_ZERO',s]) for s in d.SEEDS]
 result['factorial_interaction']=dict(definition='(LONG_RESP-LONG_ZERO)-(COMPACT_RESP-COMPACT_ZERO)',seed_differences=values,mean_difference=st.mean(values),sd=st.stdev(values),positive_seeds=sum(v>0 for v in values),interpretation='Whether the frozen relation/response effect changes with the previously verified temporal structure; not independent attribution to coverage or SiLU separately.')
 d.write(d.BASE/'analysis.json',result);return result
