"""Overlay the fixed historical-momentum assay on its immutable own E20 runtime."""
import json
import sys
import types
from pathlib import Path


def main():
    request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    here=Path(__file__).resolve().parents[2];previous=Path(request['source_release'])
    sys.path[:0]=[str(previous),str(previous/'code'),str(previous/'experiments/adv3b02_xuc/code')]
    for name in ('cvs_rx_momentum20','cvs_restart_state20','cvs_crossrx_sharpness20','cvs_evidence_recomposition20','cvs_identity_difference20'):
        package=types.ModuleType('experiments.'+name)
        package.__path__=[str(here/'experiments'/name)];sys.modules[package.__name__]=package
    from experiments.cvs_rx_momentum20.source_probe import run
    try:run(request)
    except Exception as exc:
        from experiments.cvs_rx_momentum20.design import write
        output=Path(request['output_root'])
        if output.is_dir() and not (output/'summary.json').exists():
            write(output/'failure.json',dict(status='FAILED',exception=type(exc).__name__,reason=str(exc),
                preserve_outputs=True,automatic_retry=False))
        raise


if __name__=='__main__':main()
