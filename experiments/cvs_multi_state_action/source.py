"""Exact native baseline training with scoped multi-disentanglement hooks."""
import argparse
from contextlib import contextmanager
from pathlib import Path
import torch
from experiments.cvs_multi_state_action import design as d
from experiments.cvs_multi_state_action.runtime import installed, ARM_PATHS
from experiments.cvs_phase1_stack import source
from experiments.cvs_phase1_stack.fast_execution import incremental_native_writer as base_incremental


@contextmanager
def incremental_writer(native):
    # Production fast writer replaces the native writer; annotate outside it.
    with base_incremental(native):
        writer = native._write_ssdg_epoch_telemetry
        def telemetry(cp, jp, rows):
            native._state_action_annotate(rows)
            return writer(cp, jp, rows)
        native._write_ssdg_epoch_telemetry = telemetry
        try:
            yield native
        finally:
            native._write_ssdg_epoch_telemetry = writer


def train(c):
    originals = {name: getattr(source, name) for name in
                 ('validate', 'make_args', 'installed', 'incremental_native_writer')}
    source.validate = d.validate
    source.make_args = d.make_args
    source.installed = installed
    source.incremental_native_writer = incremental_writer
    try:
        source.train(c)
        output=Path(c['output_root'])
        state=torch.load(output/'auxiliary_final.pth',map_location='cpu',weights_only=False)
        if state['epoch']!=200 or state['config']!=c or state['target_access']:
            raise ValueError('Auxiliary final-state provenance mismatch')
        execution=state['mechanism_execution']
        missing=[]
        for kind in ARM_PATHS[c['arm']]:
            if execution.get('attempted_identity_steps',execution['identity_steps']).get(kind,0)<=0:
                missing.append(kind+'_identity')
            if c['arm']!='real_views' and execution['fit_steps'].get(kind,0)<=0:
                missing.append(kind+'_fit')
        if 'receiver' in ARM_PATHS[c['arm']] and execution['receiver_relations_total']<=0:
            missing.append('receiver_group_relations')
        result=dict(status='FAILED_METHOD_NOT_EXECUTED' if missing else 'VERIFIED',
            missing=missing,**execution)
        d.write(output/'mechanism_execution.json',result)
        completion=d.read(output/'completion.json')
        completion['mechanism_execution']=result
        if missing: completion['status']='FAILED_METHOD_NOT_EXECUTED'
        d.write(output/'completion.json',completion)
        if missing:
            raise RuntimeError('FAILED_METHOD_NOT_EXECUTED: '+','.join(missing))
    finally:
        for name, value in originals.items():
            setattr(source, name, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    train(d.read(parser.parse_args().config))
