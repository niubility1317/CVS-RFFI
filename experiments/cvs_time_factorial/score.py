import statistics as st
from . import design as d
from .predict import preflight
from experiments.cvs_pa_cross_delay.score import score as shared_score,finish as shared_finish

def score(row):return shared_score(row,design=d,preflight_fn=preflight)
def factorial(value):
 index={(r['arm'],r['model_seed']):r['accuracy'] for r in value['records'] if r['dimension']=='overall'}
 delta=[index['LONG_SILU',s]-index['COMPACT_SILU',s]-index['LONG_LINEAR',s]+index['COMPACT_LINEAR',s] for s in d.SEEDS]
 value['factorial_interaction']=dict(definition='(LONG_SILU-COMPACT_SILU)-(LONG_LINEAR-COMPACT_LINEAR)',seeds=list(d.SEEDS),seed_differences=delta,mean_difference=st.mean(delta),sd_difference=st.stdev(delta),positive_seeds=sum(v>0 for v in delta),target_feedback=False)
def finish():return shared_finish(design=d,analysis_callback=factorial)
