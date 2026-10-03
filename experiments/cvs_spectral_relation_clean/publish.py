"""Publish the committed conditional forty-eight-row clean evaluation."""
import argparse
from pathlib import Path
from experiments.cvs_spectral_relation_clean.prepare import RUN, RELEASE
from experiments.cvs_clean_eval.publish import publish, inspect
from experiments.cvs_spectral_relation_identity.publish import preflight as source_preflight

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args()
    if a.inspect:inspect(a.output,run=RUN,release=RELEASE)
    else:publish(a.output,run=RUN,release=RELEASE,spec_ref='experiments/cvs_spectral_relation_clean/configs/launch_spec.json',matrix_rows=48,preflight=source_preflight)
