from . import design as d
from experiments.cvs_ordinary_fold20.publish import publish as original
def publish(output):return original(output,design=d,package='experiments/cvs_source_dc_crossfit',folders=['experiments/cvs_source_dc_crossfit'])
