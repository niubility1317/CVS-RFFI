from . import design as d
from experiments.cvs_f_coherentmix20.publish import publish as original
def publish(output):return original(output,design=d,package='experiments/cvs_f_timegrid20',folders=['experiments/cvs_f_timegrid20'])
