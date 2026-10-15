import argparse
from . import design as d,model
from .observer import Observer
from experiments.cvs_feature_sequence50.source import train as shared_train
def train(c):
    d.validate(c)
    shared_train(c,design=d,model_api=model,observer=Observer(c))
    from pathlib import Path
    out=Path(c['output_root']);done=d.read(out/'completion.json');mechanism=d.read(out/'source_U_increment_mechanism.json')
    health=done['final_training_L_CE']<=d.GATE['final_training_L_CE_max']
    mechanism['source_health']=dict(final_training_L_CE=done['final_training_L_CE'],L_CE_within_fixed_max=health,
        final_source_metrics=done['final_source_metrics'])
    mechanism['gate']['full_source_training_qualification']=mechanism['gate']['full_source_training_qualification'] and health
    mechanism['final_source_completion_verified']=True
    d.write(out/'source_U_increment_mechanism.json',mechanism)
    return mechanism
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    train(d.read(parser.parse_args().config))
