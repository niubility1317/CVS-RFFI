import importlib.util
import sys
from pathlib import Path

ROOT = Path('E:/type10-7')
sys.path.insert(0, str(ROOT))
from experiments.cric_dura_comparison.bootstrap import activate
frozen = ROOT / 'local_artifacts/cric_dura_comparison_20261021/frozen_sources'
activate(dict(frozen_sources={k: str(frozen / k) for k in ('F', 'AF', 'BASE', 'FOLD', 'G')}))
import experiments
experiments.__path__.append(str(Path(__file__).resolve().parents[1] / 'experiments'))


def test_registered_matrix_and_resource_limits():
    assert importlib.util.find_spec('experiments.pamf_split_spectrum_paper') is not None
    from experiments.pamf_split_spectrum_paper.plan import rows, limits
    items = rows()
    assert len(items) == 226
    assert len({r['row_id'] for r in items}) == 226
    assert sum(r['reuse_key'] is not None for r in items) == 15
    assert sum(r['family'] == 'branch' for r in items) == 33
    assert sum(r['family'] == 'structure_ce' for r in items) == 12
    assert sum(r['family'] == 'factorial' for r in items) == 173
    assert sum(r['family'] == 'architecture' for r in items) == 8
    assert [limits(i) for i in range(8)] == [3, 3, 3, 3, 3, 3, 3, 1]


def test_all_new_rows_have_source_only_registered_configuration():
    import json
    import pytest
    from experiments.pamf_split_spectrum_paper.plan import rows, configuration, validate
    context = json.loads((ROOT / 'local_artifacts/pamf_no_reference_completion_20261109/context.json').read_text(encoding='utf8'))
    for row in rows():
        if not row['new_training']:
            continue
        cfg = configuration(context, row, '/source-only', 'test')
        assert cfg['paper_row'] == row
        assert cfg['model_seed'] == row['model_seed']
        with pytest.raises(ValueError):
            validate(dict(cfg, target_truth='forbidden'))


def test_split_graph_reuses_original_full_computation_and_separates_inputs():
    import json
    import torch
    from experiments.cric_dura_comparison import f_train, t1_model
    from experiments.cvs_phase1_stack import runtime
    from experiments.pamf_branch_factorial.graph import install as old_install
    from experiments.cric_dura_comparison.structural import remove_branch
    from experiments.pamf_split_spectrum_paper.graph import install
    ctx = json.loads((ROOT / 'local_artifacts/pamf_branch_launch_20261109/context.json').read_text(encoding='utf8'))
    c = ctx['baselines']['2026092701']['complete']['config']
    def fresh():
        torch.manual_seed(1234)
        with f_train.class_dimension(c), runtime.installed(f_train.original_config(2026092701)) as src:
            a = f_train.arguments(c, 'cpu')
            a.input_len = 256
            return t1_model.install(src.build_baseline_model(a, 'cpu'), c).eval()
    old = old_install(fresh(), 'strict_no_frequency', 2026092701).eval()
    new = fresh()
    remove_branch(new.id_backbone.encoder, 'F10')
    install(new, 1, 1, 1, 1)
    new.load_state_dict(old.state_dict(), strict=True)
    x = torch.randn(2, 2, 256)
    assert torch.equal(old(x), new(x))
    for pa, spectrum in ((1, 0), (0, 1), (1, 1)):
        net = fresh()
        remove_branch(net.id_backbone.encoder, 'F10')
        install(net, 1, 1, pa, spectrum)
        out = net(x)
        assert out.shape == (2, 6) and torch.isfinite(out).all()
        out.sum().backward()
        core = net.id_backbone.encoder.core.id_backbone
        if spectrum:
            assert any(p.grad is not None for p in core.pa_stats_proj.parameters())
        if not pa:
            assert not any(True for p in core.pa_lift.parameters())


def test_main_and_e200_graphs_have_removed_original_frequency():
    import json
    import torch
    from experiments.cric_dura_comparison import f_train, t1_model
    from experiments.cvs_phase1_stack import runtime
    from experiments.pamf_paper_completion import model as old
    from experiments.pamf_split_spectrum_paper.plan import rows, configuration, CE_VARIANT
    from experiments.pamf_split_spectrum_paper.model import paper_scope, main_graph_scope, ce_build
    context = json.loads((ROOT / 'local_artifacts/pamf_no_reference_completion_20261109/context.json').read_text(encoding='utf8'))
    for method in ('sup', 'pl_permuted', 'ssl'):
        row = next(r for r in rows() if r['family'] == 'factorial' and r['data_setting'] == 'main_split' and r['method'] == method and r['model_seed'] == 2026092701)
        c = configuration(context, row, '/source-only', 'test')
        with paper_scope(), main_graph_scope(), old.installed(c, {}, False), f_train.class_dimension(c), runtime.installed(f_train.original_config(c['model_seed'])) as src:
            args = f_train.arguments(c, 'cpu')
            args.input_len = 256
            net = t1_model.install(src.build_baseline_model(args, 'cpu'), c).eval()
        assert not hasattr(net.id_backbone.encoder, 'response')
        assert not net.id_backbone.encoder.core.id_backbone.use_freq_path
        assert net(torch.randn(2, 2, 256)).shape == (2, 6)
    net = ce_build(CE_VARIANT, 2026092701).eval()
    assert not net.core.id_backbone.use_freq_path
    assert net(torch.randn(2, 2, 256)).shape == (2, 6)


def test_each_new_structural_graph_rebuilds_from_its_checkpoint():
    import json
    import torch
    from experiments.cric_dura_comparison import f_train, t1_model
    from experiments.cvs_phase1_stack import runtime
    from experiments.cric_dura_comparison import supervised_physics
    from experiments.pamf_split_spectrum_paper.plan import rows, configuration
    from experiments.pamf_split_spectrum_paper.model import structural_graph_scope
    context = json.loads((ROOT / 'local_artifacts/pamf_no_reference_completion_20261109/context.json').read_text(encoding='utf8'))
    for row in (r for r in rows() if r['family'] in ('branch','structure_ce') and r['new_training'] and r['model_seed']==2026092701):
        c = configuration(context,row,'/source-only','test')
        def fresh():
            with f_train.class_dimension(c), runtime.installed(f_train.original_config(c['model_seed'])) as src:
                args = f_train.arguments(c,'cpu'); args.input_len = 256
                net = t1_model.install(src.build_baseline_model(args,'cpu'),c).eval()
            with structural_graph_scope():
                return supervised_physics.install(net,'SS_'+row['structure'],c['model_seed']).eval()
        net = fresh()
        clone = fresh()
        clone.load_state_dict(net.state_dict(), strict=True)
        x = torch.randn(2,2,256)
        assert torch.equal(net(x), clone(x)), row['row_id']


def test_full_matrix_analysis_and_seed_pairing(tmp_path):
    import csv, json, shutil
    from experiments.pamf_split_spectrum_paper.analyze import run
    from experiments.pamf_split_spectrum_paper.plan import rows, VIEWS
    root=tmp_path/'run'; root.mkdir(); release=tmp_path/'release'; release.mkdir()
    shutil.copyfile(ROOT/'exports/results/20261014-sigtime-purece-e200-eight-seed-clean-r01/comparison_per_seed.csv',release/'historical_architecture.csv')
    def savedir(seed, six):
        p=tmp_path/f'scores-{seed}-{six}'; p.mkdir(); n=6 if six else 7; data=[]
        for test, perclass in (('cross_rx',3000),('cross_both',1000)):
            for view in VIEWS:
                for route in ('raw','DURA-L','DURA-LU'):
                    cm=[[perclass if i==j else 0 for j in range(6)] for i in range(6)]
                    for k in range(n):
                        data.append(dict(test=test,view=view,readout=route,dimension='receiver',stratum='rx'+str(k),count=6*perclass,accuracy=1.,macro_f1=1.,confusion=cm))
                    overall=[[v*n for v in line] for line in cm]
                    data.append(dict(test=test,view=view,readout=route,dimension='overall',stratum='ALL',count=6*perclass*n,accuracy=1.,macro_f1=1.,confusion=overall))
        (p/'complete.json').write_text('{"status":"SCORED_COMPLETE"}',encoding='utf8')
        (p/'scores.json').write_text(json.dumps(dict(model_seed=seed,truth_last=True,target_feedback_forbidden=True,results=data)),encoding='utf8')
        return str(p)
    paths={(seed,six):savedir(seed,six) for seed in range(2026092701,2026092709) for six in (False,True)}
    own={r['row_id']:dict(row=r,scores=paths[(r['model_seed'],r['protocol_id']=='m1-natural')]) for r in rows()}
    matrix=json.loads((Path(__file__).resolve().parents[1]/'automation_reports/CV-SincNet/20261023-phase1-practical-residual-full2x2-m600-r01/experiment.json').read_text(encoding='utf8'))['matrix']['rows']
    external=[dict(row=r,scores=paths[(r['model_seed'],r['protocol_id']=='m1-natural')]) for r in matrix if r['kind']=='external']
    (root/'sources.json').write_text(json.dumps(own),encoding='utf8')
    (root/'external_sources.json').write_text(json.dumps(external),encoding='utf8')
    run(dict(run_root=str(root),project=str(tmp_path),release_root=str(release)))
    with (root/'analysis/paired_architecture.csv').open(encoding='utf-8-sig',newline='') as f:
        assert len(list(csv.DictReader(f)))==224
    with (root/'analysis/paired_structural.csv').open(encoding='utf-8-sig',newline='') as f:
        assert len(list(csv.DictReader(f))) > 0
