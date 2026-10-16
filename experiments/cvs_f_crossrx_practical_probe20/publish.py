from . import design as d
from experiments.cvs_ordinary_fold20.publish import publish as original

def publish(output):
    return original(output,design=d,package='experiments/cvs_f_crossrx_practical_probe20',folders=[
        'experiments/cvs_ordinary_fold20','experiments/cvs_raw_anchor20','experiments/cvs_white_observation20',
        'experiments/cvs_ordinary_pa_source','experiments/cvs_urban_sigtime',
        'experiments/cvs_f_crossrx_practical20',
        'experiments/cvs_f_practical_source20/__init__.py','experiments/cvs_f_practical_source20/trace.py',
        'experiments/cvs_f_crossrx_practical_probe20'])
