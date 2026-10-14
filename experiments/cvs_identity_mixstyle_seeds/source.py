from pathlib import Path
from . import design as d
from .model import installed
from experiments.cvs_identity_mixstyle50.source import shared_train,details,epoch_diagnostic

def train(c):
 shared_train(c,design=d,installer=installed,details=details,epoch_callback=epoch_diagnostic)
 from .feature_check import run
 proof=run(c['model_seed']);own=next((r for r in proof['rows'] if r['row_id']==c['row_id']),None)
 if own is None:raise ValueError('Actual same-row source witness missing')
 d.write(Path(c['output_root'])/'source_feature_witness.json',dict(row=own,source_only=True,query_read=False,head_fit=False,V_cache_materialized=True,V_forward=False))
