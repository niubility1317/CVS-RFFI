import json
import math
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/scripts'))
from run_a1_ecrs_cross_rx import ecrs_matrix
from run_a1_fast_v2 import v2_command, predecessor_complete
from SSDG import train_ssdg as train


def test_eight_real_matrix_commands_scratch_and_attributable():
    matrix=ecrs_matrix()
    assert matrix['max_gpu_processes']==2 and 'after_run' not in matrix
    assert [r['gpu'] for r in matrix['rows']]==list(range(8))
    parsed=[]
    for row in matrix['rows']:
        args=train.build_arg_parser().parse_args(v2_command(matrix,project_root=Path('/p'),run_root=Path('/new'),row=row)[3:])
        train._validate_daot_config(args); train._validate_a1_scratch_only(args); train._validate_a1_ecrs_config(args)
        assert args.from_scratch and args.a1_scratch_only and not train._frozen_teacher_requested(args)
        assert not args.use_a1_r3 and args.a1_runtime_fast and args.a1_ema_versioned_updates
        assert args.epochs==200 and args.seed==392005 and args.batch_size==128 and args.muse_unlabeled_batch_size==256
        assert args.test_eval_start_epoch>args.epochs and args.test_eval_interval==0
        assert args.checkpoint_selection=='final_only'
        assert args.balanced_sampler_tx_per_batch*args.balanced_sampler_domain_per_batch*args.balanced_sampler_samples_per_cell==128
        parsed.append(args)
    def same(a,b): return a==b or (isinstance(a,float) and isinstance(b,float) and math.isnan(a) and math.isnan(b))
    def differences(a,b):
        return {k for k in vars(parsed[a]) if not same(getattr(parsed[a],k),getattr(parsed[b],k))} - {'output_dir','candidate_id','phase2_export_path'}
    assert differences(0,1)=={'a1_ecrs_cross_rx_weight'}
    assert differences(1,2)=={'a1_ecrs_cross_rx_scope'}
    for index,key in ((3,'a1_ema_startup_average'),(4,'a1_logit_coverage_weighting'),
                      (5,'use_tx_rx_balanced_sampler'),(6,'identity_domain_objective_mode')):
        assert differences(2,index)=={key}
    assert differences(2,7)=={'a1_ema_startup_average','a1_logit_coverage_weighting',
                             'use_tx_rx_balanced_sampler','identity_domain_objective_mode'}


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-.1])
def test_invalid_cross_rx_config_rejected(value):
    args=train.build_arg_parser().parse_args(['--output_dir','unused'])
    args.a1_ecrs_cross_rx_weight=value
    with pytest.raises(ValueError): train._validate_a1_ecrs_config(args)


def test_predecessor_checks_include_pending_scoring_and_partial_write(tmp_path):
    matrix={'after_run':'before'}
    path=tmp_path/'runs/before/pipeline_state.json'; path.parent.mkdir(parents=True)
    path.write_text('{',encoding='utf-8')
    assert not predecessor_complete(matrix,tmp_path)
    path.write_text(json.dumps({'rows':{'r':{'status':'RUNNING'}}}),encoding='utf-8')
    assert not predecessor_complete(matrix,tmp_path)
    path.write_text(json.dumps({'rows':{'r':{'status':'SCORED_PENDING_ANALYSIS'}}}),encoding='utf-8')
    assert predecessor_complete(matrix,tmp_path)
    assert predecessor_complete({},tmp_path)


def test_detached_dispatch_preserves_command_cwd_and_refuses_duplicate_log(tmp_path,monkeypatch):
    import run_a1_fast_v2 as runner
    release=tmp_path/'release'; release.mkdir()
    project=tmp_path/'project'
    for path in (project/'Dataset_WigSig/ManySig.pkl',
                 project/'runs'/runner.BASE_RUN/'target_inputs/manifest.json',
                 project/'runs'/runner.BASE_RUN/'target_truth/truth_sidecar.json'):
        path.parent.mkdir(parents=True,exist_ok=True); path.write_text('{}',encoding='utf-8')
    script=release/'run_a1_ecrs_cross_rx.py'
    monkeypatch.setattr(runner,'RELEASE',release)
    monkeypatch.setattr(sys,'argv',[str(script),'--project-root',str(project),'--run-id','new','--detach'])
    calls=[]
    def launch(argv,**kw):
        calls.append((argv,kw))
        return type('Process',(),{'pid':1234})()
    monkeypatch.setattr(runner.subprocess,'Popen',launch)
    runner.main(matrix_factory=ecrs_matrix)
    assert len(calls)==1 and calls[0][0]==[sys.executable,'-u',str(script),'--project-root',str(project),'--run-id','new']
    assert calls[0][1]['cwd']==release and calls[0][1]['start_new_session']
    assert calls[0][1]['stdin']==runner.subprocess.DEVNULL
    assert json.loads((release/'dispatcher_process.json').read_text())['pid']==1234
    with pytest.raises(FileExistsError): runner.main(matrix_factory=ecrs_matrix)
    assert len(calls)==1
