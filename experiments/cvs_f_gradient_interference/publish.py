from . import design as d
from experiments.cvs_f_localatten20.publish import publish as original
FOLDERS=['experiments/cvs_ordinary_fold20','experiments/cvs_raw_anchor20','experiments/cvs_f_crossrx_practical20','experiments/cvs_urban_sigtime/__init__.py','experiments/cvs_urban_sigtime/channel.py','experiments/cvs_f_gradient_interference']
def publish(output):return original(output,design=d,package='experiments/cvs_f_gradient_interference',folders=FOLDERS)
