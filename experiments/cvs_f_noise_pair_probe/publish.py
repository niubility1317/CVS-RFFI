"""Reuse established single-archive ordinary-account publisher."""
from . import design as d
from experiments.cvs_f_coherentmix20.publish import publish as original
FOLDERS=['experiments/cvs_f_noise_pair_probe']
def publish(output):return original(output,design=d,package='experiments/cvs_f_noise_pair_probe',folders=FOLDERS)
