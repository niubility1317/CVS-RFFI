"""Focused launch guards; no data or target reads."""
import json
import sys
import tempfile
import contextlib
import io
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_rc4_matched import build_args, resolved_config

def main():
    root=Path(__file__).resolve().parents[2]
    configs=sorted((root/'configs/matched_20260927').glob('cvs-*.json'))
    assert len(configs)==5
    for config in configs:
        args=build_args(config,'unused','unused')
        assert args.lr==.0002 and args.epochs==200
    doc=json.loads(configs[0].read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory() as folder:
        cfg=Path(folder)/'bad.json'
        for key,value in [('--a1_periodic_target_start','100'),('--a1_final_weak_reference','true'),
                          ('--a1_periodic_target_truth','forbidden'),('--practical_equalization','true'),
                          ('--baseline_ckpt','forbidden'),('--checkpoint_selection','best'),('--epochs','199')]:
            bad=json.loads(json.dumps(doc));bad['options'][key]=value
            cfg.write_text(json.dumps(bad),encoding='utf-8')
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    build_args(cfg,'unused','unused')
            except (ValueError,AssertionError,SystemExit):pass
            else:raise AssertionError('Guard did not reject '+key)
    try:build_args(configs[0],'unused','unused',target_inputs='forbidden')
    except ValueError:pass
    else:raise AssertionError('Target path accepted')
    print('PASS five configs, seven negative mutations, forbidden target argument')

if __name__=='__main__':main()
