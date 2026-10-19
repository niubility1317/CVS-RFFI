from . import design as d
from .model import own_model
from experiments.cvs_feature_adapters30.diagnose import readout
def run(row):
 c=d.config(next(r for r in d.rows() if r['row_id']==row));m=own_model(c,'cuda:0');return readout(m,d.BASE/row/'source_qualification')
