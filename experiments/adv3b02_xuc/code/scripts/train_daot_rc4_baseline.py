"""Native F0 recipe, scratch only, exact common physical source roles."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from SSDG import train_ssdg as native
from cvsrffi.xuc_fusion.native import role_ids_from_native
from cvsrffi.game_tracking.runtime import json_write

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--dataset',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-contract',type=Path,required=True);p.add_argument('--validate-only',action='store_true')
    a=p.parse_args();doc=json.loads(a.config.read_text());opts=dict(doc['options'])
    opts.update({'--wisig_pkl':a.dataset,'--output_dir':str(a.output),'--candidate_id':doc['id']})
    argv=[]
    for k,v in opts.items():
        argv.append(k)
        if v is not None:argv.append(str(v))
    args=native.build_arg_parser().parse_args(argv)
    native._validate_a1_scratch_only(args);native._validate_daot_config(args)
    assert args.from_scratch and not args.baseline_ckpt and not args.teacher_ckpt
    assert args.fasttrust_rc4 and args.use_adv3b02_daot_stn and not args.use_a1_r3
    assert args.a1_source_screen_only and args.epochs==200
    if a.validate_only:print(json.dumps(vars(args),default=str));return 0
    if a.output.exists():raise FileExistsError(a.output)
    expected=json.loads(a.source_contract.read_text());original=native._build_ssdg_wisig_data
    def checked(*values,**kw):
        ctx=original(*values,**kw)
        assert not ctx['named_test_loaders']
        if role_ids_from_native(ctx)!=expected['role_ids']:raise ValueError('native physical role mismatch')
        json_write(a.output/'source_contract.json',dict(expected,native_role_comparison='EXACT_MATCH'))
        json_write(a.output/'resolved_config.json',vars(args))
        json_write(a.output/'initialization.json',dict(scratch_only=True,checkpoint_sources=[],seed=args.seed,target_contact=False,source_roles='EXACT_MATCH'))
        return ctx
    native._build_ssdg_wisig_data=checked;start=time.time()
    code=native.train(args)
    if code:raise RuntimeError(code)
    import torch
    payload=torch.load(a.output/'final_ssdg.pth',map_location='cpu',weights_only=False)
    assert payload['epoch']==200 and payload['args']['from_scratch'] and not payload['args']['baseline_ckpt']
    json_write(a.output/'completion.json',dict(status='TRAINING_COMPLETE',epochs=200,native=True,elapsed_seconds=time.time()-start,target_evaluated=False))
    return 0
if __name__=='__main__':raise SystemExit(main())
