import math
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/scripts'))
from run_a1_fast_v2 import v2_matrix, v2_command
from SSDG import train_ssdg as train


def test_real_commands_are_scratch_source_only_and_single_factor():
    matrix=v2_matrix()
    rows=[]
    for row in matrix['rows']:
        argv=v2_command(matrix,project_root=Path('/p'),run_root=Path('/new'),row=row)
        args=train.build_arg_parser().parse_args(argv[3:])
        train._validate_daot_config(args); train._validate_a1_scratch_only(args)
        assert args.from_scratch and args.a1_scratch_only and not args.use_a1_r3
        assert not args.baseline_ckpt and not args.teacher_ckpt and not train._frozen_teacher_requested(args)
        assert args.epochs==200 and args.seed==392005 and args.a1_ema_versioned_updates
        assert args.rc4_calibration_update_epochs=='1,21,41,91,161'
        assert args.test_eval_start_epoch>args.epochs and args.test_eval_interval==0
        assert (args.labeled_ratio,args.unlabeled_ratio,args.source_val_ratio)==(.07,.63,.3)
        rows.append(args)
        for flag in ('--baseline_ckpt','--teacher_ckpt'):
            forbidden=__import__('copy').deepcopy(matrix)
            forbidden['core90_options'][flag]='old.pth'
            with pytest.raises(ValueError):
                v2_command(forbidden,project_root=Path('/p'),run_root=Path('/new'),row=row)
    def same(a,b): return a==b or (isinstance(a,float) and isinstance(b,float) and math.isnan(a) and math.isnan(b))
    for a,b,change in ((0,1,'a1_runtime_fast'),(1,2,'a1_ema_startup_average'),(1,3,'a1_logit_coverage_weighting')):
        differing={k for k in vars(rows[a]) if not same(getattr(rows[a],k),getattr(rows[b],k))}
        assert differing=={'output_dir','candidate_id','phase2_export_path',change}
    assert [r['gpu'] for r in matrix['rows']]==[4,5,6,7]
