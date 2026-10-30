"""One cached producer -> independent three-truth score, plus owned-path negatives."""
from contextlib import contextmanager
from pathlib import Path
import importlib
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_class_competition_run')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_class_density_run import fixture as parent_fixture
    with parent_fixture(root,mp) as (parent,pc,_,backend,_,_,_):
        m=api();pc['code_commit']=m.PARENT_CODE;parent.run(pc,'cpu')
        counts=dict(parent.COUNTS);mp.setattr(m,'COUNTS',counts);mp.setattr(m,'Q_COUNT',parent.Q_COUNT)
        c=dict(run_id=m.RUN_ID,code_commit='e'*40,reference_config=pc,output_root=str(root/'class_competition'),batch_size=128)
        events=[];load=m.parent._backend;rows=m._rows;target=m._target_rows
        def checkpoint(*args):events.append('checkpoint');return load(*args)
        def source_rows(path,role):
            if not Path(c['output_root'],'complete.json').exists():
                assert Path(c['output_root'],'models/complete.json').is_file()
                if role=='known':assert events==['checkpoint']
                else:assert Path(c['output_root'],'calibration/complete.json').is_file() and Path(c['output_root'],'known_predictions.npz').is_file()
                events.append(role)
            return rows(path,role)
        def query_rows(path,name):
            if not Path(c['output_root'],'complete.json').exists():
                assert Path(c['output_root'],'novel_predictions.npz').is_file();events.append(name)
            return target(path,name)
        mp.setattr(m.parent,'_backend',checkpoint);mp.setattr(m,'_rows',source_rows);mp.setattr(m,'_target_rows',query_rows)
        mp.setattr(m.parent,'_public',lambda *a:pytest.fail('No target IQ or U access; only cached rows'))
        mp.setattr(m.core.density,'fit_known',lambda *a:pytest.fail('No source Gaussian refit'))
        mp.setattr(m.core.density,'fit_background',lambda *a:pytest.fail('No OE refit'))
        yield m,c,counts,events,backend


def test_exact_configuration_rejects_private_U_K_or_overlapping_output(tmp_path,monkeypatch,subtests):
    m=api();monkeypatch.setattr(m.parent,'validate_config',lambda c:c)
    p=dict(code_commit=m.PARENT_CODE,output_root=str(tmp_path/'parent'),reference_prediction_root=str(tmp_path/'reference'),
        train_config=dict(IQ_root=str(tmp_path/'IQ'),output_root=str(tmp_path/'train')))
    c=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=p,output_root=str(tmp_path/'new'),batch_size=128)
    assert m.validate_config(c) is c
    for update in (dict(truth='private'),dict(U_public_root='forbidden'),dict(true_K=3),dict(batch_size=True),
            dict(reference_config=dict(p,code_commit='b'*40)),dict(output_root=p['train_config']['IQ_root'])):
        with subtests.test(update=update),pytest.raises(ValueError):m.validate_config(dict(c,**update))


def test_output_or_failed_checkpoint_smoke_precedes_any_source_or_query(tmp_path,monkeypatch):
    m=api();monkeypatch.setattr(m,'validate_config',lambda c:c)
    monkeypatch.setattr(m.parent,'_backend',lambda *a:pytest.fail('No checkpoint reload on existing output'))
    sentinel=tmp_path/'keep';sentinel.write_text('unchanged',encoding='utf-8')
    with pytest.raises(FileExistsError):m.run(dict(output_root=str(tmp_path)),'cpu')
    assert sentinel.read_text(encoding='utf-8')=='unchanged'
    models=[torch.nn.Linear(2,2).eval().requires_grad_(False) for _ in range(3)]
    monkeypatch.setattr(m.parent,'_backend',lambda *a:dict(reader=models[0],model=models[1],dura=(models[2],None)))
    monkeypatch.setattr(m.parent,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('checkpoint smoke failure')))
    monkeypatch.setattr(m,'_reference',lambda *a:pytest.fail('No source state before smoke'))
    monkeypatch.setattr(m,'_rows',lambda *a:pytest.fail('No source rows before smoke'))
    monkeypatch.setattr(m,'_target_rows',lambda *a:pytest.fail('No Q rows before smoke'))
    out=tmp_path/'new'
    with pytest.raises(ValueError,match='smoke failure'):m.run(dict(output_root=str(out),reference_config={}),'cpu')
    assert not out.exists()


def test_actual_producer_then_CPU_three_truths_controls_and_corruptions(tmp_path,monkeypatch,subtests):
    from test_pamf_phase3_star_class_density_score import truth_files
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_class_competition_score')
    with fixture(tmp_path,monkeypatch) as (m,c,counts,events,backend):
        result=m.run(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','known','novel','clean_DURA','star_DURA']
        assert result['neural_updates']==result['density_fit_updates']==result['new_statistical_parameters']==0
        assert result['score_policy_frozen'] and result['identity_predictions_unchanged'] and result['source_states_unchanged']
        assert not (out/'models/prior.npz').exists()
        monkeypatch.setattr(s,'COUNTS',counts);monkeypatch.setattr(s,'Q_COUNT',m.Q_COUNT)
        paths=truth_files(tmp_path,s,m,out);reads=[];source=s._source_truth;target=s._target_truth
        def source_truth(*a):reads.append('source');return source(*a)
        def target_truth(path,*a):reads.append(Path(path).name);return target(path,*a)
        monkeypatch.setattr(s,'_source_truth',source_truth);monkeypatch.setattr(s,'_target_truth',target_truth)
        monkeypatch.setattr(m.parent,'_backend',lambda *a:pytest.fail('No checkpoint load in independent CPU scorer'))
        monkeypatch.setattr(m.risk,'calibrate_known',lambda *a:pytest.fail('No re-calibration in scorer'))
        scored=s.score(out,*paths,tmp_path/'score')
        assert reads==['source','clean_DURA_truth.npz','star_DURA_truth.npz'] and scored['truth_last']
        assert scored['all_three_predictions_before_truth'] and scored['primary_method']==m.PRIMARY_METHOD
        assert scored['neural_updates']==scored['density_fit_updates']==0 and not scored['final_phase3_complete']
        with pytest.raises(FileExistsError):s.score(out,*paths,tmp_path/'score')
        files=[out/'complete.json',out/'models/density.npz',out/'thresholds.npz',out/'novel_predictions.npz',out/'targets/clean_DURA/predictions.npz',out/'targets/star_DURA/predictions.npz']
        saved={p:p.read_bytes() for p in files}
        for defect in ('partial','identity','density','control','new_score','new_tau','not_frozen','private_Q'):
            with subtests.test(defect=defect):
                for path,value in saved.items():path.write_bytes(value)
                reads.clear()
                if defect=='partial':(out/'targets/star_DURA/predictions.npz').unlink()
                elif defect in ('identity','not_frozen'):
                    path=out/'complete.json';a=m._read(path)
                    if defect=='identity':a['model_identity']['updates']+=1
                    else:a['all_thresholds_frozen_before_novel_and_Q']=False
                    path.unlink();m._write(path,a)
                else:
                    path=out/'models/density.npz' if defect=='density' else out/'thresholds.npz' if defect=='new_tau' else out/'novel_predictions.npz' if defect=='control' else out/'targets/star_DURA/predictions.npz'
                    with np.load(path,allow_pickle=False) as z:a={key:z[key].copy() for key in z.files}
                    if defect=='density':a['known_means'][0,0]+=.1
                    elif defect=='new_tau':a['conditional'][3,0]=np.nan
                    elif defect=='control':a['scores'][0,0]+=.1
                    elif defect=='new_score':a['raw_scores'][0,3]+=.1
                    else:a['y']=np.zeros(m.Q_COUNT,np.int64)
                    np.savez(path,**a)
                with pytest.raises((ValueError,FileNotFoundError)):s.score(out,*paths,tmp_path/'invalid')
                assert not reads and not (tmp_path/'invalid').exists()
