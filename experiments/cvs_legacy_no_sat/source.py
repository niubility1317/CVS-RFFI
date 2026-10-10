from experiments.cvs_legacy_no_sat import design as d
from experiments.cvs_phase1_stack import source
def train(c):
 original=(source.validate,source.make_args)
 source.validate=d.validate;source.make_args=d.make_args
 try:source.train(c)
 finally:source.validate,source.make_args=original
