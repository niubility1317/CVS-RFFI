from . import design as d
from experiments.cvs_ordinary_fold20.publish import publish as original

def publish(output):
    return original(output,design=d,package='experiments/cvs_f_realpair20',folders=[
        'experiments/cvs_ordinary_fold20','experiments/cvs_raw_anchor20',
        'experiments/cvs_urban_sigtime/__init__.py','experiments/cvs_urban_sigtime/channel.py','experiments/cvs_f_crossrx_practical20','experiments/cvs_f_realpair20'])
