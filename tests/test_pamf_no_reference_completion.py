import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT=Path('E:/type10-7')
sys.path.insert(0,str(PROJECT))
from experiments.cric_dura_comparison.bootstrap import activate
roots=PROJECT/'local_artifacts/cric_dura_comparison_20261021/frozen_sources'
activate(dict(frozen_sources={k:str(roots/k) for k in ('F','AF','BASE','FOLD','G')}))
import experiments
experiments.__path__.append(str(Path(__file__).resolve().parents[1]/'experiments'))
from experiments.pamf_no_reference_completion.plan import rows,configuration,validate,SEEDS,CE_VARIANT
from experiments.pamf_no_reference_completion.model import ce_build,structure_scope
from experiments.cric_dura_comparison import supervised_physics

context=json.loads((PROJECT/'local_artifacts/pamf_paper_completion_20261027/context.json').read_text(encoding='utf-8'))
branch=json.loads((PROJECT/'local_artifacts/pamf_branch_launch_20261109/context.json').read_text(encoding='utf-8'))
context['baselines']=branch['baselines']
torch.set_num_threads(2)


def test_matrix_and_source_permissions():
    plan=rows()
    assert len(plan)==190 and sum(r['new_training'] for r in plan)==184
    assert sum(r['family']=='structure_ce' for r in plan)==12
    assert sum(r['family']=='architecture' for r in plan)==8
    assert not any(r['channel_augmentation'] for r in plan if r['family']=='architecture')
    assert len({r['row_id'] for r in plan})==190
    for row in plan:
        if not row['new_training']:continue
        try: c=configuration(context,row,'/source-only','test')
        except Exception as error: raise AssertionError(row['row_id']) from error
        assert c['paper_row']==row and c['model_seed']==row['model_seed']
        if row['family']!='architecture':
            assert c['num_classes']==(10 if row['data_setting']=='ten_class' else 6)
        with pytest.raises(ValueError): validate(dict(c,target_truth='forbidden'))


def test_no_reference_e200_default_head_and_backward():
    net=ce_build(CE_VARIANT,SEEDS[0]).eval()
    assert not hasattr(net,'response')
    x=torch.randn(2,2,256)
    result=net(x)
    assert result.shape==(2,6) and torch.isfinite(result).all()
    torch.nn.functional.cross_entropy(result,torch.tensor([0,1])).backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_structure_scope_restores_original_on_exit():
    original=supervised_physics.install
    with structure_scope(): assert supervised_physics.install is not original
    assert supervised_physics.install is original


def test_pooled_macro_f1_uses_confusions():
    from experiments.pamf_no_reference_completion.analyze import combine
    a={'count':4,'confusion':[[2,0],[1,1]]}
    b={'count':2,'confusion':[[0,1],[0,1]]}
    pooled=combine([a,b])
    assert pooled['count']==6 and pooled['accuracy']==pytest.approx(4/6)
    assert pooled['macro_f1']==pytest.approx(2/3)
    with pytest.raises(ValueError): combine([dict(a,count=3),b])


@pytest.mark.parametrize('setting,method', [('main_split','sup'),('main_split','pl_noaug'),
    ('main_split','ssl'),('ten_class','ssl'),('source_rx_1','ssl')])
def test_actual_method_graph_removes_reference_and_keeps_views(setting,method):
    from experiments.pamf_paper_completion import model
    from experiments.cric_dura_comparison import f_train,t1_model
    from experiments.cvs_phase1_stack import runtime
    from experiments.pamf_no_reference_completion.model import paper_scope,seed_scope
    row=next(r for r in rows() if r['data_setting']==setting and r['method']==method and r['new_training'])
    c=configuration(context,row,'/source-only','test')
    with seed_scope(c['model_seed']),paper_scope(),model.installed(c,{},False),f_train.class_dimension(c),runtime.installed(f_train.original_config(row['model_seed'])) as src:
        a=f_train.arguments(c,'cpu');a.input_len=256
        net=t1_model.install(src.build_baseline_model(a,'cpu'),c).eval()
    ref=net.id_backbone.encoder
    assert not hasattr(ref,'response') and len(ref.sequence_pool.views)==3
    x=torch.randn(2,2,256);out=net(x)
    assert out.shape==(2,c['num_classes']) and torch.isfinite(out).all()
    torch.nn.functional.cross_entropy(out,torch.tensor([0,1])).backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_full_frozen_analysis_preserves_three_and_eight_seed_scopes(tmp_path):
    import csv,shutil
    from experiments.pamf_no_reference_completion.analyze import run
    from experiments.pamf_no_reference_completion.plan import VIEWS
    from experiments.pamf_branch_factorial.method import rows as branch_rows
    root=tmp_path/'run';root.mkdir();project=tmp_path/'project';release=tmp_path/'release';release.mkdir()
    branch=project/'runs/20261109-phase1-pamf-branch-factorial-m36-r01';control=branch/'controls/20261109-phase1-pamf-branch-factorial-m36-r01-control-r02'
    control.mkdir(parents=True);(control/'pipeline_state.json').write_text('{"status":"ANALYZED"}',encoding='utf-8')
    shutil.copyfile(PROJECT/'exports/results/20261014-sigtime-purece-e200-eight-seed-clean-r01/comparison_per_seed.csv',release/'historical_architecture.csv')
    def savedir(seed,six):
        p=tmp_path/f'scores-{seed}-{six}';p.mkdir()
        n=6 if six else 7;data=[]
        for test,perclass in (('cross_rx',3000),('cross_both',1000)):
            for view in VIEWS:
                for route in ('raw','DURA-L','DURA-LU'):
                    cm=[[perclass if i==j else 0 for j in range(6)] for i in range(6)]
                    for k in range(n):
                        data.append(dict(test=test,view=view,readout=route,dimension='receiver',stratum='rx'+str(k),count=6*perclass,accuracy=1.,macro_f1=1.,confusion=cm))
                    overall=[[v*n for v in line] for line in cm]
                    data.append(dict(test=test,view=view,readout=route,dimension='overall',stratum='ALL',count=6*perclass*n,accuracy=1.,macro_f1=1.,confusion=overall))
        (p/'complete.json').write_text('{"status":"SCORED_COMPLETE"}',encoding='utf-8')
        (p/'scores.json').write_text(json.dumps(dict(model_seed=seed,truth_last=True,target_feedback_forbidden=True,results=data)),encoding='utf-8')
        return str(p)
    paths={(seed,six):savedir(seed,six) for seed in range(2026092701,2026092709) for six in (False,True)}
    own={r['row_id']:dict(row=r,scores=paths[(r['model_seed'],r['protocol_id']=='m1-natural')]) for r in rows()}
    extpath=Path(__file__).resolve().parents[1]/'automation_reports/CV-SincNet/20261023-phase1-practical-residual-full2x2-m600-r01/experiment.json'
    matrix=json.loads(extpath.read_text(encoding='utf-8'))['matrix']['rows']
    external=[dict(row=r,scores=paths[(r['model_seed'],r['protocol_id']=='m1-natural')]) for r in matrix if r['kind']=='external']
    first={r['row_id']:dict(row=r,scores=paths[(r['model_seed'],False)]) for r in branch_rows()}
    for p,obj in ((root/'sources.json',own),(root/'external_sources.json',external),(branch/'sources.json',first)):
        p.write_text(json.dumps(obj),encoding='utf-8')
    run(dict(run_root=str(root),project=str(project),release_root=str(release)))
    with (root/'analysis/summary.csv').open(encoding='utf-8',newline='') as f: summary=list(csv.DictReader(f))
    picked=[r for r in summary if r['setting']=='F0' and r['family']=='PAMF' and r['group']=='factorial' and
            r['method']=='ssl' and r['test']=='pooled' and r['view']=='clean' and r['readout']=='raw' and r['metric']=='accuracy']
    assert {(r['seed_scope'],r['seed_n']) for r in picked}=={('all','8'),('matched_three','3')}
    with (root/'analysis/paired_architecture.csv').open(encoding='utf-8',newline='') as f:
        assert len(list(csv.DictReader(f)))==224
