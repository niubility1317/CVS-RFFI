from . import design as d
from experiments.cvs_ordinary_fold20.publish import publish as original
from experiments.cvs_f_phasecycle20.publish import FOLDERS as base
FOLDERS=[*base,'experiments/cvs_source_ellipticity','experiments/cvs_f_rxmix20']
def publish(output):return original(output,design=d,package='experiments/cvs_f_rxmix20',folders=FOLDERS)
