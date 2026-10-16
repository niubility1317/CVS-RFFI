from experiments.cvs_ordinary_fold20.publish import publish as audited_publish
from . import design as d
def publish(output):return audited_publish(output,design=d,package='experiments/cvs_source_ellipticity',folders=['experiments/cvs_source_ellipticity'])
