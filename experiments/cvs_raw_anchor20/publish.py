from . import design as d
from experiments.cvs_ordinary_fold20.publish import publish as original
def publish(output):
    return original(output,design=d,package='experiments/cvs_raw_anchor20',folders=['experiments/cvs_ordinary_fold20','experiments/cvs_raw_anchor20'])
