"""Single-archive ordinary-account publication;no extra seals or permissions."""
from . import design as d
from experiments.cvs_f_coherentmix20.publish import publish as original
FOLDERS=['experiments/cvs_f_postas_noise_probe']
def publish(output):return original(output,design=d,package='experiments/cvs_f_postas_noise_probe',folders=FOLDERS)
