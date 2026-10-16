from . import design as d
from experiments.cvs_f_localatten20.publish import publish as original
FOLDERS=['experiments/cvs_f_additive_rx6']
def publish(output):return original(output,design=d,package='experiments/cvs_f_additive_rx6',folders=FOLDERS)
