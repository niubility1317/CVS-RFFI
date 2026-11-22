"""Release the LoRa Phase1 supplement without recopying the main input data."""
import argparse
from pathlib import PurePosixPath

from experiments.cric_dura_comparison.io import read
from .phase1 import LONG_PL_RUN, LONG_SUP_RUN, OPTIMIZATION_RUN, RUN, SOURCE_OPT_RUN
from .publish import release_commit, release


MAIN = '20261121-phase1-pamf-lora-rx5-main-m18-r01'


def publish(repo, output, context, main_submit, ssh_config, optimization=False,
            long_supervised=False, long_pl=False, source_optimized=False):
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
    if source_optimized:
        run_id, gpus = SOURCE_OPT_RUN, list(range(8))
    elif long_supervised or long_pl:
        run_id, gpus = (LONG_SUP_RUN if long_supervised else LONG_PL_RUN), [3, 7]
    elif optimization:
        run_id, gpus = OPTIMIZATION_RUN, [7]
    else:
        run_id, gpus = RUN, list(range(7))
    rt.update(run_id=run_id, run_root=str(project/'runs'/run_id), main_run=str(primary),
              release_root=str(project/'releases'/f'{run_id}-release'), code_commit=release_commit(repo),
              runner='experiments.pamf_lora_main.phase1', allowed_gpus=gpus,
              source_root=str(base/'data/source'), target_root=str(base/'data/target'),
              source_practical=str(primary/'source_views/L_practical.npy'),
              target_practical=str(primary/'target_views'))
    return release(repo, output, rt, ssh_config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('repo','output','context','main-submit','ssh-config'):
        parser.add_argument(f'--{key}', required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--optimization', action='store_true')
    modes.add_argument('--long-supervised', action='store_true')
    modes.add_argument('--long-pl', action='store_true')
    modes.add_argument('--source-optimized', action='store_true')
    publish(**vars(parser.parse_args()))
