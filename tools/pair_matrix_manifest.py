"""Freeze the 24 source-only training commands without launching processes."""
import argparse
import json
from pathlib import Path, PurePosixPath
from pair_reform_dry_run import build_commands

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/phase1_adv3b02_pair24_20260905.json'


def replace_options(argv, values):
    argv = list(argv)
    for key, value in values.items():
        flag = '--' + key
        value = str(value).lower() if isinstance(value, bool) else str(value)
        if flag in argv:
            argv[argv.index(flag)+1] = value
        else:
            argv += [flag, value]
    return argv


def build_manifest(*, code_root, output_root, python, dataset, checkpoint):
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    rows = []
    # Seed-major order produces a complete mechanism screen before repetitions.
    entries = [(name, spec, seed, 'core') for seed in config['seeds']
               for name, spec in config['core'].items()]
    entries += [(name, spec, config['sensitivity_seed'], 'sensitivity')
                for name, spec in config['sensitivity'].items()]
    for name, spec, seed, family in entries:
        row_id = f'{name}_S{seed}'
        destination = str(PurePosixPath(output_root) / row_id)
        inherited = build_commands(root=code_root, python=python, dataset=dataset,
            checkpoint=checkpoint, output_root=output_root, rows=[spec['base']])['commands'][0]
        argv = replace_options(inherited['argv'], {
            **config['common_overrides'], **spec['overrides'], 'seed': seed,
            'output_dir': destination, 'candidate_id': row_id, 'run_id': config['run_id'],
            'phase2_export_path': destination + '/phase2_zid_prototypes.pt'})
        # Build Linux paths explicitly even when the manifest is prepared on Windows.
        argv[2] = str(PurePosixPath(code_root) / 'code/SSDG/train_ssdg.py')
        rows.append(dict(assigned_gpu=(len(rows)%8+len(rows)//8)%8,row_id=row_id,candidate=name,seed=seed,family=family,epochs=config['epochs'],
            argv=argv,cwd=code_root,environment={'PYTHONPATH':code_root+'/code:'+code_root,
                'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
                'PYTHONUNBUFFERED':'1'}))
    return dict(schema=config['schema'],run_id=config['run_id'],code_root=code_root,
        output_root=output_root,python=python,checkpoint=checkpoint,dataset=dataset,
        gpus=list(range(8)),max_active=24,max_per_gpu=8,max_new_per_gpu=3,min_free_mib=8192,
        epochs=config['epochs'],selection=config['selection'],rows=rows,
        smoke_argv=[python,code_root+'/code/scripts/smoke_adv3b02_pair_reform.py',
            '--checkpoint',checkpoint,'--output-json',output_root+'/checkpoint_smoke.json','--device','cpu'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('code-root','output-root','python','dataset','checkpoint','output'):
        parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    result=build_manifest(code_root=args.code_root,output_root=args.output_root,python=args.python,
        dataset=args.dataset,checkpoint=args.checkpoint)
    path=Path(args.output)
    with path.open('x',encoding='utf-8',newline='\n') as handle:
        json.dump(result,handle,ensure_ascii=False,indent=2)
        handle.write('\n')
    print(json.dumps({'rows':len(result['rows']),'path':str(path)}))


if __name__ == '__main__':
    main()
