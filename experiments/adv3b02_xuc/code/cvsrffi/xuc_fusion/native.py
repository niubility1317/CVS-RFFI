"""Pinned native A1 controls: source-only construction and common physical roles."""
import json
from pathlib import Path
from cvsrffi.game_tracking.data import opaque_id
from cvsrffi.game_tracking.runtime import json_write


def native_argv(reference,row,dataset,output):
    options=dict(reference['rows'][row]['reference_options'])
    options.update({'--wisig_pkl':str(dataset),'--output_dir':str(output),'--candidate_id':row,
        '--seed':'392005','--epochs':'200','--amp':'false','--from_scratch':'true','--a1_scratch_only':'true',
        '--baseline_ckpt':'','--teacher_ckpt':'','--a1_source_screen_only':'true',
        '--muse_external_final_eval':'true','--phase2_export_prototypes':'false',
        '--eval_sat_channel':'false','--num_workers':'0','--device':'cuda:0'})
    argv=[]
    for k,v in options.items():
        argv.append(k)
        if v is not None:argv.append(str(v))
    return argv


def role_ids_from_native(ctx):
    result={}
    for role,name in [('L_s','train_loader'),('U_s','unlabeled_loader'),('V','val_loader')]:
        ds=ctx[name].dataset
        if not hasattr(ds,'index'):ds=ds.base
        result[role]=sorted(opaque_id(it) for it in ds.index)
    return result


def train_native(reference,row,dataset,output,source_contract,validate_only=False):
    from SSDG import train_ssdg as native
    argv=native_argv(reference,row,dataset,output)
    args=native.build_arg_parser().parse_args(argv)
    if validate_only:return vars(args)
    out=Path(output)
    if out.exists():raise FileExistsError(out)
    expected=json.loads(Path(source_contract).read_text(encoding='utf-8'))
    original=native._build_ssdg_wisig_data
    def checked(*a,**kw):
        ctx=original(*a,**kw)
        if ctx['named_test_loaders']:raise ValueError('native training constructed target loaders')
        actual=role_ids_from_native(ctx)
        if actual!=expected['role_ids']:raise ValueError('native source physical roles mismatch')
        out.mkdir(parents=True,exist_ok=True)
        json_write(out/'source_contract.json',dict(expected,native_role_comparison='EXACT_MATCH'))
        json_write(out/'initialization.json',dict(scratch_only=True,checkpoint_sources=[],seed=392005,
            target_contact=False,native_pipeline=True,source_roles='EXACT_MATCH',argv=argv))
        return ctx
    native._build_ssdg_wisig_data=checked
    code=native.train(args)
    if code:raise RuntimeError(f'native training returned {code}')
    import torch
    ckpt=torch.load(out/'final_ssdg.pth',map_location='cpu',weights_only=False)
    if ckpt['epoch']!=200 or not ckpt['args'].get('from_scratch') or ckpt['args'].get('baseline_ckpt'):raise ValueError('native final checkpoint contract')
    json_write(out/'completion.json',dict(status='TRAINING_COMPLETE',epochs=200,native=True,target_evaluated=False))
    return 0
