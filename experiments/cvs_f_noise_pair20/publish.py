from . import design as d
from experiments.cvs_f_coherentmix20.publish import publish as original
FOLDERS=['experiments/cvs_f_noise_pair20','experiments/cvs_f_noise_pair_probe/__init__.py','experiments/cvs_f_noise_pair_probe/noise.py']
def publish(output):return original(output,design=d,package='experiments/cvs_f_noise_pair20',folders=FOLDERS)
