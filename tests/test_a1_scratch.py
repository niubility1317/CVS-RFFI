import sys
import math
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/scripts'))
from run_a1_scratch_no_checkpoint import scratch_matrix, scratch_command
from SSDG import train_ssdg as train


def parsed_rows():
    matrix=scratch_matrix()
    return [train.build_arg_parser().parse_args(scratch_command(matrix,project_root=Path('/p'),
                run_root=Path('/new'),row=row)[3:]) for row in matrix['rows']]


def test_scratch_pair_rejects_all_checkpoint_dependencies_and_keeps_schedule():
    rows=parsed_rows()
    for args in rows:
        train._validate_daot_config(args)
        train._validate_a1_scratch_only(args)
        assert args.from_scratch and not args.baseline_ckpt and not args.teacher_ckpt
        assert not train._frozen_teacher_requested(args)
        assert args.rc4_identity_start_epoch==21
        assert args.rc4_calibration_update_epochs=='1,21,41,91,161'
        assert args.seed==392005 and args.model_variant=='lite_d' and args.epochs==200
        assert args.use_ema_teacher and args.fasttrust_rc4 and not args.rc4_use_anchor
        args.teacher_ckpt='forbidden.pth'
        with pytest.raises(ValueError,match='forbids'): train._validate_a1_scratch_only(args)
    def equal(a,b):
        return a==b or (isinstance(a,float) and isinstance(b,float) and math.isnan(a) and math.isnan(b))
    differing={k for k in vars(rows[1]) if not equal(getattr(rows[1],k),getattr(rows[2],k))}
    assert differing <= {'output_dir','candidate_id','phase2_export_path','daot_efficiency_mode',
                         'daot_batched_scale_readback','daot_skip_mean_metadata'}
    assert rows[0].a1_r3_aux_scale==0 and rows[1].a1_r3_aux_scale==rows[2].a1_r3_aux_scale==1
    assert all(a.use_a1_r3 for a in rows)


@pytest.mark.parametrize('field,value',[('rc4_use_anchor',True),('rc4_cache_anchor_logits',True),
    ('rc4_lambda_feature_anchor',0.1),('lambda_teacher_clean_kl',0.1),('sat_anchor_ssl',True)])
def test_scratch_guard_rejects_hidden_frozen_objectives(field,value):
    args=parsed_rows()[0]
    setattr(args,field,value)
    with pytest.raises(ValueError,match='forbids'): train._validate_a1_scratch_only(args)


def test_rc4_calibration_uses_ema_without_anchor():
    args=parsed_rows()[0]
    class Ema(torch.nn.Module):
        def forward(self,x,**kwargs):
            return {'tx_logits':torch.ones(len(x),6),'z_id':torch.ones(len(x),8)}
    n=12; domains=torch.arange(n)%3
    batch=(torch.randn(n,2,256),torch.arange(n)%6,domains,{'rx_i':domains})
    sentinel=object()
    with patch.object(train,'build_rc4_calibration',return_value=sentinel) as build:
        result=train._calibrate_rc4_vcal(None,Ema(),[batch],args=args,
                        domain_label_map={0:0,1:1,2:2},device=torch.device('cpu'),num_classes=6,num_domains=3)
    assert result is sentinel
    assert torch.equal(build.call_args.args[0],build.call_args.args[1])


def test_original_rc4_still_requires_frozen_anchor_by_default():
    args=parsed_rows()[0]; args.rc4_use_anchor=True; args.a1_scratch_only=False
    assert train._frozen_teacher_requested(args)
