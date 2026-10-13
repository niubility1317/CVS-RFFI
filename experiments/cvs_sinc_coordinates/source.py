from pathlib import Path
from . import design as d
from .model import installed,physical_state
from experiments.cvs_pa_cross_delay.source import train as shared_train,source_validation
from experiments.cvs_default_head.source_health import preserving_rng
from experiments.cvs_feature_diagnostics.sinc_change import compare

def details(model):return dict(identity_input_policy=model._identity_input_policy,actual_sinc_coordinates=model._sinc_coordinates_contract)
def epoch_diagnostic(c,state,row):
 m=state['model'];module=m.id_backbone.encoder.core.id_backbone.sinc
 result=compare(m._sinc_initial_physical_state,physical_state(module),module._native_type)
 record=dict(epoch=int(row['epoch']),Sinc=result)
 if record['epoch'] in (33,34):record['source_V_RX_health']=preserving_rng(m,state['ctx']['val_loader'],'cuda:0',source_validation)
 state.setdefault('sinc_diagnostics',[]).append(record)
 d.write(Path(c['output_root'])/'source_Sinc_epoch_diagnostic.json',dict(records=state['sinc_diagnostics'],source_only=True,head_fit=False,query_read=False,extra_V_passes=sum('source_V_RX_health' in r for r in state['sinc_diagnostics'])))
def train(c):return shared_train(c,design=d,installer=installed,details=details,epoch_callback=epoch_diagnostic)
