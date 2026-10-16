from . import design as d
from experiments.cvs_ordinary_fold20.publish import publish as original

def publish(output):
    return original(output,design=d,package='experiments/cvs_ordinary_pa_source',folders=[
        'experiments/cvs_ordinary_fold20','experiments/cvs_ordinary_confirm20','experiments/cvs_ordinary_pa_source'])
