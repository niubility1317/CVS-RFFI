from . import design as d
from experiments.cvs_source_spur.publish import publish as original
def publish(output):return original(output,design=d,package='experiments/cvs_rxsession_probe17',folders=['experiments/cvs_rxsession_probe17'])
