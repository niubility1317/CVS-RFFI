from pathlib import Path
from . import design as d
from .model import installed
from experiments.cvs_pa_cross_delay.source import train as shared_train,source_validation
from experiments.cvs_default_head.source_health import preserving_rng
def details(model):return dict(identity_input_policy=model._identity_input_policy,actual_source_neighbor_sampling=model._source_neighbor_sampling)
def epoch_diagnostic(c,state,row):
 history=state['ctx']['unlabeled_loader'].sampler.history
 epoch=int(row['epoch']);expected=max(0,epoch-c['label_epochs'])
 if len(history)!=expected or (expected and history[-1]['delivered_count']!=56700):raise ValueError('Every U physical packet once per active pseudo epoch required')
 record=dict(epoch=epoch,sampler=dict(history[-1]) if expected else None,U_IQ_fetched=bool(expected))
 if record['epoch'] in (33,34):record['source_V_RX_health']=preserving_rng(state['model'],state['ctx']['val_loader'],'cuda:0',source_validation)
 state.setdefault('sampling_diagnostics',[]).append(record)
 d.write(Path(c['output_root'])/'source_sampling_epoch_diagnostic.json',dict(records=state['sampling_diagnostics'],source_only=True,U_truth_read=False,head_fit=False,query_read=False,extra_V_passes=sum('source_V_RX_health' in r for r in state['sampling_diagnostics'])))
def train(c):return shared_train(c,design=d,installer=installed,details=details,epoch_callback=epoch_diagnostic)
