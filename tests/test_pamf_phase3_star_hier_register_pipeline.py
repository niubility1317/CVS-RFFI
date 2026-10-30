"""One cached HDBSCAN producer through the shared independent truth pipeline."""
from pathlib import Path
import importlib
import numpy as np
import pytest


def test_actual_cached_hierarchical_pipeline_and_truth_closed_negatives(tmp_path,monkeypatch,subtests):
    # Import the policy before the old fixture installs its local test spies.
    h=importlib.import_module('experiments.pamf_phase3_open_world.star_hier_register_run')
    s=importlib.import_module('experiments.pamf_phase3_open_world.star_hier_register_score')
    from test_pamf_phase3_star_signal_register_run import fixture
    with fixture(tmp_path,monkeypatch) as (base,old,counts,old_events,backend,U_public):
        old['code_commit']=h.INPUT_CODE;old['output_root']=str(tmp_path/base.RUN_ID/'predict');base.run(old,'cpu')
        original_identity=(base.RUN_ID,base.SCHEMA,base.STATUS,base.core)
        monkeypatch.setattr(h,'COUNTS',counts);monkeypatch.setattr(h,'Q_COUNT',base.Q_COUNT);monkeypatch.setattr(h,'U_COUNT',base.U_COUNT)
        c=dict(old,run_id=h.RUN_ID,code_commit='d'*40,output_root=str(tmp_path/h.RUN_ID/'predict'),input_reference_root=old['output_root'])
        out=Path(c['output_root']);cached=Path(old['output_root']);events=[];current_calls=[]
        load=base.parent._backend;current=base.parent._current;provider=h._source;U_provider=h._U
        discover=h.core.discover;calibrate=h.core.calibrate;rows=base._rows;target=base._target_rows
        def checkpoint(*a):events.append('checkpoint');return load(*a)
        def synthetic_only(*a):
            assert events==['checkpoint'] and len(a[1])==2
            current_calls.append(2);return current(*a)
        def source(*a):
            assert events==['checkpoint'] and (out/'smoke.json').is_file();events.append('source_cache')
            value=provider(*a);assert value[-1]==dict(source_L_reads=0,source_signal_rows=0,source_radius_new_fits=0);return value
        def U_cache(*a):
            assert events==['checkpoint','source_cache'] and (out/'models/complete.json').is_file();events.append('U_cache')
            value=U_provider(*a)
            with np.load(cached/'registration/U.npz',allow_pickle=False) as z:
                assert set(value[0])=={'ids','default_logits','closed','signal_features'}
                assert all(np.array_equal(v,z[k]) for k,v in value[0].items())
            return value
        def hierarchical(*a):
            assert events==['checkpoint','source_cache','U_cache'];events.append('HDBSCAN');return discover(*a)
        def known_rows(path,role):
            if not (out/'complete.json').exists():
                assert (out/('registration' if role=='known' else 'calibration')/'complete.json').is_file()
                events.append(role)
            return rows(path,role)
        def known_calibration(*a):
            assert events[-1]=='known' and (out/'registration/complete.json').is_file();events.append('calibrate');return calibrate(*a)
        def future_rows(path,name):
            if not (out/'complete.json').exists():
                assert (out/'calibration/complete.json').is_file() and (out/'novel_predictions.npz').is_file();events.append(name)
            return target(path,name)
        monkeypatch.setattr(base.parent,'_backend',checkpoint);monkeypatch.setattr(base.parent,'_current',synthetic_only)
        monkeypatch.setattr(base,'_source_L',lambda *a:pytest.fail('No new source L IQ may be read'))
        monkeypatch.setattr(base.U_reader,'_public',lambda *a:pytest.fail('No new U IQ or channel observation'))
        monkeypatch.setattr(h.core,'source_radius',lambda *a:pytest.fail('No source radius refit'))
        monkeypatch.setattr(h,'_source',source);monkeypatch.setattr(h,'_U',U_cache);monkeypatch.setattr(h.core,'discover',hierarchical)
        monkeypatch.setattr(h.core,'calibrate',known_calibration);monkeypatch.setattr(base,'_rows',known_rows);monkeypatch.setattr(base,'_target_rows',future_rows)
        # The input cache may provide source/U aggregates only. Its calibration
        # and source/novel/query prediction files are not discovery inputs.
        read=base._read
        def cache_metadata_only(path):
            p=Path(path)
            assert not (p.is_relative_to(cached) and ('calibration' in p.parts or p.name in ('known_predictions.npz','novel_predictions.npz')))
            return read(path)
        monkeypatch.setattr(h,'_read',cache_metadata_only)
        done=h.run(c,'cpu')
        assert events==['checkpoint','source_cache','U_cache','HDBSCAN','known','calibrate','novel','clean_DURA','star_DURA']
        assert current_calls==[2] and original_identity==(base.RUN_ID,base.SCHEMA,base.STATUS,base.core)
        assert done['run_id']==h.RUN_ID and done['U_features_reused'] and done['source_radius_new_fits']==done['current_source_L_reads']==done['source_L_reads']==0
        assert all(done[k]==0 for k in ('real_signal_forward_rows','real_F_forward_rows','G_forward_rows'))
        assert all(done[k]==2 for k in ('synthetic_signal_forward_rows','synthetic_F_forward_rows','G_synthetic_forward_rows'))
        assert done['target_U_reads']==base.U_COUNT and done['source_radius_fitted']
        diag=h._read(out/'registration/complete.json')['diagnostics']
        assert diag['algorithm']=='sklearn.cluster.HDBSCAN' and not diag['source_radius_used_for_clustering']
        assert diag['parameters']['min_cluster_size']==16 and diag['parameters']['min_samples']==8
        scorer=s.base;monkeypatch.setattr(scorer,'COUNTS',counts);monkeypatch.setattr(scorer,'U_COUNT',base.U_COUNT);monkeypatch.setattr(scorer,'Q_COUNT',base.Q_COUNT)
        U=scorer._load(out/'registration/U.npz');n=scorer._load(out/'novel_predictions.npz')
        paths=[tmp_path/'U_truth.npz',tmp_path/'source_truth.npz',tmp_path/'clean_truth.npz',tmp_path/'star_truth.npz']
        def private_file(path,a):np.savez(path,**{k:v[::-1] for k,v in a.items()})
        private_file(paths[0],dict(ids=U['ids'],y=np.arange(base.U_COUNT,dtype=np.int64)%8,receiver=np.full(base.U_COUNT,26,np.int64),scenario=np.asarray([scorer.common.SCENES[0]]*base.U_COUNT)))
        grid=[(tx,r,sc) for tx in scorer.common.UNKNOWN_TX for r in scorer.common.RECEIVERS for sc in scorer.common.SCENES]
        private_file(paths[1],dict(ids=n['ids'],rx_i=n['rx_i'],day_i=n['day_i'],tx=np.asarray([a[0] for a in grid]),scenario=np.asarray([a[2] for a in grid])))
        for name,path in zip(base.CONDITIONS,paths[2:]):
            a=scorer._load(out/'targets'/name/'predictions.npz');N=len(a['ids'])
            private_file(path,dict(ids=a['ids'],y=np.arange(N,dtype=np.int64)%9,receiver=np.full(N,26,np.int64),scenario=np.asarray(['clean' if name=='clean_DURA' else scorer.common.SCENES[0]]*N)))
        reads=[];truth=scorer._truth;source_truth=scorer._source_truth
        monkeypatch.setattr(scorer,'_truth',lambda path,*a:reads.append(Path(path).name) or truth(path,*a))
        monkeypatch.setattr(scorer,'_source_truth',lambda path,*a:reads.append(Path(path).name) or source_truth(path,*a))
        monkeypatch.setattr(base.parent,'_backend',lambda *a:pytest.fail('No neural load in independent CPU scorer'))
        monkeypatch.setattr(h.core,'discover',lambda *a:pytest.fail('No reclustering in scorer'))
        monkeypatch.setattr(h.core,'calibrate',lambda *a:pytest.fail('No recalibration in scorer'))
        result=s.score(out,*paths,tmp_path/'score')
        assert reads==[p.name for p in paths] and result['schema']==s.SCHEMA and result['status']==s.STATUS and result['run_id']==h.RUN_ID
        assert result['truth_last'] and result['U_mapping_frozen_before_source_and_Q_truth']
        assert result['source_radius_new_fits']==result['current_source_L_reads']==0 and result['U_features_reused']
        assert not result['final_phase3_complete'] and result['full_goal_status']=='ACTIVE'
        with pytest.raises(FileExistsError):h.run(c,'cpu')
        files=[cached/'complete.json',out/'complete.json',out/'thresholds.npz',out/'novel_predictions.npz',out/'targets/clean_DURA/predictions.npz',out/'targets/star_DURA/predictions.npz']
        saved={p:p.read_bytes() for p in files}
        for defect in ('cache_ancestor','extra_truth','partial_freeze','missing_source','missing_clean','missing_star','old_tau','registration_tau','old_control'):
            with subtests.test(defect=defect):
                for p,v in saved.items():p.write_bytes(v)
                reads.clear()
                if defect=='extra_truth':
                    with pytest.raises(ValueError):h.validate_config(dict(c,U_truth='private'))
                else:
                    if defect in ('cache_ancestor','partial_freeze'):
                        p=cached/'complete.json' if defect=='cache_ancestor' else out/'complete.json';a=read(p)
                        if defect=='cache_ancestor':a['code_commit']='f'*40
                        else:a['anonymous_frozen_before_V_and_Q']=False
                        p.unlink();h._write(p,a)
                    elif defect.startswith('missing_'):
                        p=out/'novel_predictions.npz' if defect=='missing_source' else out/'targets'/('clean_DURA' if defect=='missing_clean' else 'star_DURA')/'predictions.npz';p.unlink()
                    else:
                        p=out/'thresholds.npz' if defect.endswith('tau') else out/'novel_predictions.npz';a=scorer._load(p)
                        if defect=='old_tau':a['conditional'][0,0]+=.1
                        elif defect=='registration_tau':a['registration'][1]=np.nan
                        else:a['scores'][0,0]+=.1
                        np.savez(p,**a)
                    with pytest.raises((ValueError,FileNotFoundError)):s.score(out,*paths,tmp_path/'invalid')
                assert reads==[] and not (tmp_path/'invalid').exists()
