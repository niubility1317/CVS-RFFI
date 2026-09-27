"""Explicit 200-epoch reference clock for the R3 budget experiment only."""
from pathlib import Path
import hashlib


def reference_clock_enabled(args):
    return bool(getattr(args, 'a1_r3_budget_mode', False) or getattr(args, 'a1_extended_budget_mode', False))


def reference_epoch(args, epoch):
    if not reference_clock_enabled(args):
        return int(epoch)
    total=int(args.epochs)
    allowed = (400,600) if getattr(args, 'a1_extended_budget_mode', False) else (120,160,200)
    if total not in allowed or not 1 <= int(epoch) <= total:
        raise ValueError('Epoch outside registered training budget')
    if int(epoch)==total:return 200
    return ((int(epoch)-1)*200)//total+1


def reference_total(args):
    return 200 if reference_clock_enabled(args) else int(args.epochs or 200)


def u_satellite_scenario(args, epoch, batch_index):
    from .muse_ssdg import adv3b02_core90_u_satellite_policy
    _,scenarios=adv3b02_core90_u_satellite_policy(reference_epoch(args,epoch))
    key=f'{int(args.seed)}:{int(epoch)}:{int(batch_index)}'.encode('utf-8')
    return scenarios[int.from_bytes(hashlib.sha256(key).digest()[:8],'big')%len(scenarios)]


def scale_start(value,total):
    return int(value) if int(value)<=0 else ((int(value)-1)*int(total))//200+1


def scale_duration(value,total):
    return int(value) if int(value)<=0 else max(1,round(int(value)*int(total)/200))


def compressed_options(reference_args, base_options, total):
    if total not in (120,160,200,400,600):raise ValueError('Unregistered training budget')
    options=dict(base_options)
    for name,value in vars(reference_args).items():
        if name.startswith(('test_eval_','source_val_heavy_eval_')):continue
        if name in ('epochs','muse_final_epoch','a1_budget_snapshot_epochs'):continue
        if isinstance(value,int) and not isinstance(value,bool):
            if name.endswith('_start_epoch') or name in (
                    'muse_s2a_start','muse_s2b_start','muse_s3a_start','muse_s3b_start','muse_s3c_start',
                    'late_stable_start','mixstyle_late_start'):
                options['--'+name]=str(scale_start(value,total))
            elif name in ('stage1_epochs','stage2_epochs'):
                options['--'+name]=str(value if value<=0 else value*total//200)
            elif name.endswith('_epochs'):
                options['--'+name]=str(scale_duration(value,total))
    for name in ('rc4_calibration_update_epochs','rc4_gradient_telemetry_epochs','daot_diagnostic_epochs'):
        values=[total if int(v)==200 else scale_start(int(v),total) for v in str(getattr(reference_args,name)).split(',') if v.strip()]
        options['--'+name]=','.join(str(v) for v in sorted(set(values)))
    parts=[]
    for item in str(reference_args.sat_view_schedule).split(';'):
        start,rest=item.split('@',1);parts.append(f'{scale_start(int(start),total)}@{rest}')
    options['--sat_view_schedule']=';'.join(parts)
    points=sorted(set([total*i//4 for i in (1,2,3,4)]+list(range(total//2,total+1,10))))
    options.update({'--epochs':str(total),'--muse_final_epoch':str(total),'--a1_r3_budget_mode':'true',
        '--label_epochs':str(round(130*total/200)), '--pseudo_epochs':str(total-round(130*total/200)),
        '--a1_budget_snapshot_epochs':','.join(str(v) for v in points),
        '--a1_budget_evaluation_start_epoch':str(total//2)})
    if total > 200:
        # Extended runs use the same-PID evaluator; avoid a second writer for its snapshots.
        options.update({'--a1_r3_budget_mode':'false', '--a1_extended_budget_mode':'true',
            '--a1_budget_snapshot_epochs':'', '--a1_budget_evaluation_start_epoch':'0'})
    return options


def save_budget_snapshot(args, epoch, output_dir, payload, save_fn):
    if not bool(getattr(args,'a1_r3_budget_mode',False)):return None
    points={int(x) for x in str(args.a1_budget_snapshot_epochs).split(',') if x.strip()}
    if int(epoch) not in points:return None
    path=Path(output_dir)/f'epoch_{int(epoch):03d}_ssdg.pth'
    if path.exists():raise FileExistsError(f'Refusing to overwrite budget snapshot: {path}')
    saved=dict(payload);saved['checkpoint_role']='pre_registered_source_budget_snapshot'
    temporary=path.with_suffix(path.suffix+'.writing')
    if temporary.exists():raise FileExistsError(temporary)
    save_fn(temporary,saved)
    temporary.rename(path)
    return str(path)
