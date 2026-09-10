"""Frozen E200 source-V per-class counts through the original evaluators."""
import argparse
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch

RELEASE=Path('/home/szu2070436088/2510044040/CV-SincNet/releases/a1_mechanism_periodic_f58f89b2')
PROJECT=RELEASE.parents[1]
RUN='a1_mechanism_periodic_s392005_20260910_r1'
ROWS=('B0_FIXED','B1_TAIL_LR','B2_RC4_WEIGHT','D0_NO_ORBIT','D1_THREE_VIEW','D2_PHYSICAL_ORBIT','D3_TANGENT','F3_R3_SWAP')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    cli=parser.parse_args()
    if cli.output.exists():raise FileExistsError(cli.output)
    sys.path[:0]=[str(RELEASE/'code'),str(RELEASE)]
    from SSDG import train_ssdg as train
    from cvsrffi import eval as evaluation
    from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
    torch.set_num_threads(2);device=torch.device('cuda:0')
    result={'scope':'frozen_final_E200_source_V_only','optimizer_updates':0,'rows':{}}
    ctx=None;contract=None
    fields=('wisig_pkl','wisig_out_len','wisig_equalized','wisig_train_days','wisig_train_rxs','wisig_domain',
        'wisig_max_day123_per_combo','seed','split_mode','labeled_ratio','unlabeled_ratio','source_val_ratio','eval_batch_size')
    for name in ROWS:
        saved=torch.load(PROJECT/'runs'/RUN/name/'final_ssdg.pth',map_location='cpu',weights_only=False)
        args=SimpleNamespace(**saved['args']);assert args.a1_source_screen_only and saved['epoch']==200
        assert args.from_scratch and args.a1_scratch_only and not args.baseline_ckpt and not args.teacher_ckpt
        args.num_workers=0
        current={k:getattr(args,k,None) for k in fields}
        if ctx is None:
            with patch.object(train,'make_wisig_trainval_test_by_day_rx',side_effect=AssertionError('Target loader forbidden')):
                ctx=train._build_ssdg_wisig_data(args,device)
            contract=current
            assert len(ctx['val_loader'].dataset)==27000 and not ctx['named_test_loaders']
            result['source_contract']=contract;result['class_id_to_tx']=ctx['class_id_to_tx']
        else:assert current==contract
        model,audit=build_exact_ssdg_model_from_checkpoint(saved,input_len=256,device=device)
        model.eval();model.requires_grad_(False)
        original_unpack=evaluation.unpack_batch
        counts=torch.zeros(6,dtype=torch.int64);correct=counts.clone();current_y=None
        def unpack(batch):
            nonlocal current_y
            x,y,extra=original_unpack(batch);current_y=y.detach().cpu()
            return x,y,extra
        def hook(module,inputs,out):
            predictions=out['tx_logits'].argmax(dim=1).detach().cpu()
            counts.add_(torch.bincount(current_y,minlength=6))
            correct.add_(torch.bincount(current_y[predictions==current_y],minlength=6))
        handle=model.register_forward_hook(hook);observations={}
        with torch.no_grad(),patch.object(evaluation,'unpack_batch',side_effect=unpack):
            for i,scene in enumerate(('clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak')):
                counts.zero_();correct.zero_()
                if scene=='clean':
                    stats=evaluation.evaluate_loader(model,ctx['val_loader'],device,ctx['domain_label_map'],max_batches=0)
                    expected=saved['stats']['val']
                else:
                    expected=saved['stats']['source_val_sat_named'][scene]['aggregate']
                    seed=saved['stats']['source_val_sat_named'][scene]['named']['source_val']['sat_seed']
                    stats=evaluation.evaluate_loader_sat_channel(model,ctx['val_loader'],device,ctx['domain_label_map'],
                        scene,args,max_batches=0,seed=seed)
                assert stats['tx_correct']==int(correct.sum()) and stats['tx_total']==int(counts.sum())==27000
                assert stats['tx_correct']==expected['tx_correct'], (name,scene,stats,expected)
                observations[scene]={'aggregate':stats,'matches_saved_aggregate':True,
                    'per_class':{str(j):{'correct':int(correct[j]),'total':int(counts[j]),
                        'accuracy_pct':100*float(correct[j])/int(counts[j])} for j in range(6)}}
        handle.remove()
        assert all(torch.equal(v.cpu(),saved['model'][key]) for key,v in model.state_dict().items())
        result['rows'][name]={'model_unchanged':True,'scenarios':observations}
        print(json.dumps({'row':name,'status':'PASS','class1_clean':observations['clean']['per_class']['1'],
            'class3_clean':observations['clean']['per_class']['3']}),flush=True)
        del model,saved;gc.collect();torch.cuda.empty_cache()
    cli.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
