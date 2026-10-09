"""Use the exact old pseudo-gating/entropy/cosine loop with a changed architecture."""
import argparse
from contextlib import contextmanager
from experiments.cvs_dual_evidence import design as d
from experiments.cvs_dual_evidence.runtime import installed
from experiments.cvs_phase1_stack import source
from experiments.cvs_phase1_stack.fast_execution import incremental_native_writer as base_incremental


@contextmanager
def incremental_writer(native):
    # The fast writer replaces (rather than calls) the native callback. Inject
    # diagnostics outside it so production and the synthetic check use one path.
    with base_incremental(native):
        writer=native._write_ssdg_epoch_telemetry
        def telemetry(cp,jp,rows):
            native._dual_evidence_annotate(rows)
            return writer(cp,jp,rows)
        native._write_ssdg_epoch_telemetry=telemetry
        try:yield native
        finally:native._write_ssdg_epoch_telemetry=writer


def train(c):
    source.validate=d.validate;source.make_args=d.make_args;source.installed=installed
    source.incremental_native_writer=incremental_writer
    source.train(c)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);train(d.read(p.parse_args().config))
