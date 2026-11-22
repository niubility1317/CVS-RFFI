"""Release the LoRa Phase1 supplement without recopying the main input data."""
import argparse
from pathlib import PurePosixPath

from experiments.cric_dura_comparison.io import read
from .phase1 import OPTIMIZATION_RUN, RUN
from .publish import release_commit, release


MAIN = '20261121-phase1-pamf-lora-rx5-main-m18-r01'


def publish(repo, output, context, main_submit, ssh_config, optimization=False):
    ctx = read(context)
    main = read(main_submit)
    project = PurePosixPath(ctx['project'])
    primary = project/'runs'/MAIN
    base = project/'releases'/f'{MAIN}-release'
    if (PurePosixPath(main['run_root']) != primary or PurePosixPath(main['cwd']) != base
            or main['archive_transfer'] != 'VERIFIED'):
        raise ValueError('LoRa supplement must reuse the original main split')
    wanted = ('project','python','external_python','frozen_sources','external_releases',
              'ibrff_model_root','sdg_model_root')
    rt = {key: ctx[key] for key in wanted}
    run_id = OPTIMIZATION_RUN if optimization else RUN
    rt.update(run_id=run_id, run_root=str(project/'runs'/run_id), main_run=str(primary),
              release_root=str(project/'releases'/f'{run_id}-release'), code_commit=release_commit(repo),
              runner='experiments.pamf_lora_main.phase1', allowed_gpus=[7] if optimization else list(range(7)),
              source_root=str(base/'data/source'), target_root=str(base/'data/target'),
              source_practical=str(primary/'source_views/L_practical.npy'),
              target_practical=str(primary/'target_views'))
    return release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('repo','output','context','main-submit','ssh-config'):
        parser.add_argument(f'--{key}', required=True)
    parser.add_argument('--optimization', action='store_true')
    publish(**vars(parser.parse_args()))
