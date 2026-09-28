import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'code')]
import evaluate_d92_branch_support_probe as entry
from test_export_d92_branch_support_features import fixture as export_fixture


def fixture(root, ks=(1, 2, 5)):
    cache = root/'cache'; cache.mkdir(); capsule = root/'capsule'; capsule.mkdir()
    sha = 'a'*64; cid = 'synthetic-capsule'; classes = ['TX9','TX2']; old = ['TX9']; n = 2*max(ks)
    ids = np.array([f'physical-{i:03}' for i in range(n)]); indices = np.arange(n,dtype=np.int64)+10
    labels = np.array([classes[i//max(ks)] for i in range(n)])
    rng = np.random.default_rng(61)
    arrays = {key:rng.normal(size=(n,96 if key == 'fft' else 160)).astype(np.float32) for key in entry.BRANCHES}
    np.savez(cache/entry.CACHE_NAME,**arrays,ids=ids,indices=indices,labels=labels,checkpoint_sha256=np.array(sha),
        capsule_id=np.array(cid),feature_contract_json=np.array(json.dumps(entry.FEATURE_CONTRACT)))
    splits = []
    for k in ks:
        positions = list(range(k))+list(range(max(ks),max(ks)+k))
        splits.append(dict(split_id=f'split-k{k}',receiver='RX',scenario='scene',k=k,support_seed=7,
            registered_classes=classes,support_indices=indices[positions].tolist(),support_ids=ids[positions].tolist(),support_labels=[0]*k+[1]*k))
    provenance = dict(checkpoint_sha256=sha,classes=old,model_seed=3,verdict='MATCHED_SOURCE_ONLY_SCRATCH',
        source_role_comparison='EXACT_MATCH',checkpoint_epoch=200,checkpoint_inheritance=[],target_access_before_freeze=False,model_file_bytes=100)
    common = dict(schema=entry.CACHE_SCHEMA,checkpoint_sha256=sha,capsule_id=cid,model_seed=3,feature_contract=entry.FEATURE_CONTRACT,
        query_iq_access=False,query_rows_read=0,source_data_access=False,truth_read=False,adapted_state_inherited=False,
        native_eval=True,new_source_payload_bytes=0,new_ground_statistics_bytes=0,model_file_bytes=100)
    startup = dict(common,provenance=provenance,query_fit_access=False)
    marker = dict(common,status='BRANCH_SUPPORT_FEATURES_COMPLETE',classes=old,dtype='float32',view_count_per_observation=1,
        query_used_for_fitting=False,encoder_updated=False,native_parameters_unchanged=True,native_buffers_unchanged=True,
        count=n,split_count=len(splits),feature_array_bytes=sum(v.nbytes for v in arrays.values()),index_bytes=indices.nbytes,
        registry_array_bytes=ids.nbytes+labels.nbytes,feature_file_bytes=(cache/entry.CACHE_NAME).stat().st_size,
        shapes={key:list(v.shape) for key,v in arrays.items()},native_physical_forward_count=n,support_iq_rows_read=n,
        identity_reference_checks=n,identity_check_additional_encoder_forwards=0,timing={'total_seconds':.1},native_batch_calls=1)
    for name,value in [('features_complete.json',marker),('startup.json',startup),('checkpoint_provenance.json',provenance),
        ('support_splits.json',dict(schema=entry.CACHE_SCHEMA,capsule_id=cid,checkpoint_sha256=sha,splits=splits))]:
        entry.write(cache/name,value)
    entry.write(capsule/'manifest.json',dict(protocol_schema='p2_min_v1',phase2_data_status='VALIDATED_ONCE',capsule_id=cid,split_count=len(splits)))
    return dict(support_features=cache,capsule=capsule,output=root/'out',expected_capsule_id=cid,expected_checkpoint_sha256=sha,
        expected_model_seed=3,config=dict(algorithm=entry.FROZEN_CONFIG,matrix=dict(receivers=['RX'],scenarios=['scene'],ks=list(ks),new_counts=[1],support_seeds=[7])))


def mutate(path, fn):
    value = entry.read(path); fn(value); path.write_text(json.dumps(value),encoding='utf-8')


class SupportProbeEntryTests(unittest.TestCase):
    def test_complete_real_core_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp))
            original_load=np.load
            def guarded_load(path,*a,**kw):
                self.assertEqual(Path(path),args['support_features']/entry.CACHE_NAME)
                return original_load(path,*a,**kw)
            with patch.object(entry.np,'load',side_effect=guarded_load),contextlib.redirect_stdout(io.StringIO()):
                marker=entry.evaluate(**args)
            self.assertEqual(marker['episodes'],3);self.assertEqual(marker['k1_episodes'],1);self.assertEqual(marker['oof_episodes'],2)
            self.assertEqual(marker['factorization_count'],30)
            self.assertEqual(marker['query_rows_used'],0);self.assertEqual(marker['source_rows_used'],0)
            out=args['output'];trace=[json.loads(v) for v in (out/'fit_trace.jsonl').read_text().splitlines()]
            self.assertIsNone(trace[0]['oof']);self.assertEqual(trace[0]['folds'],[])
            for audit in trace[1:]:
                self.assertEqual(len(audit['oof']),6)
                for fold in audit['folds']:
                    self.assertFalse(set(fold['training_ids']) & set(fold['held_ids']))
                    self.assertEqual(len(fold['stages']),6)
            compact=[json.loads(v) for v in (out/'compact.jsonl').read_text().splitlines()]
            self.assertNotIn('rows',compact[1]['oof']['z']);self.assertIn('macro_accuracy',compact[1]['oof']['z']['metrics'])
            with (out/'compact.csv').open(newline='') as f:self.assertEqual(len(list(csv.DictReader(f))),3)
            with (out/'fit_stages.csv').open(newline='') as f:self.assertEqual(len(list(csv.DictReader(f))),30)
            self.assertFalse((out/'predictions.jsonl').exists());self.assertFalse((out/'scores.json').exists())
            with self.assertRaises(FileExistsError):entry.evaluate(**args)

    def test_each_call_contains_only_current_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp));real=entry.probe_branch_support;calls=[]
            def wrapper(**kwargs):
                calls.append((kwargs['support_ids'],len(kwargs['z_id'])))
                self.assertEqual(set(kwargs),set(entry.BRANCHES)|{'support_ids','support_labels','classes','old_classes'})
                return real(**kwargs)
            with patch.object(entry,'probe_branch_support',side_effect=wrapper),contextlib.redirect_stdout(io.StringIO()):entry.evaluate(**args)
            self.assertEqual([v[1] for v in calls],[2,4,10])

    def test_binding_and_matrix_reject_before_output(self):
        edits=[('support_splits.json',lambda d:d['splits'][0].update(query_indices=[0])),
               ('support_splits.json',lambda d:d['splits'][0]['support_indices'].__setitem__(0,999)),
               ('support_splits.json',lambda d:d['splits'][0]['support_labels'].__setitem__(0,1)),
               ('support_splits.json',lambda d:d['splits'].pop()),
               ('support_splits.json',lambda d:d['splits'][1].update(receiver='elsewhere')),
               ('features_complete.json',lambda d:d.update(checkpoint_sha256='b'*64)),
               ('features_complete.json',lambda d:d.update(source_data_access=True)),
               ('features_complete.json',lambda d:d.update(native_buffers_unchanged=False)),
               ('features_complete.json',lambda d:d.update(feature_array_bytes=0)),
               ('startup.json',lambda d:d.update(model_seed=4)),
               ('checkpoint_provenance.json',lambda d:d.update(target_access_before_freeze=True))]
        for name,change in edits:
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                args=fixture(Path(tmp));mutate(args['support_features']/name,change)
                with self.assertRaises(ValueError):entry.evaluate(**args)
                self.assertFalse(args['output'].exists())

    def test_extra_npz_member_and_nonfinite_rejected(self):
        for bad in ('query','nan'):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as tmp:
                args=fixture(Path(tmp));path=args['support_features']/entry.CACHE_NAME
                with np.load(path) as f:data={key:f[key] for key in f.files}
                if bad=='query':data['query']=np.zeros(1)
                else:data['z_id'][0,0]=np.nan
                np.savez(path,**data)
                with self.assertRaises(ValueError):entry.evaluate(**args)

    def test_failed_core_never_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp))
            with patch.object(entry,'probe_branch_support',side_effect=RuntimeError('synthetic failure')),contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):entry.evaluate(**args)
            self.assertFalse((args['output']/'probe_complete.json').exists())
            with self.assertRaises(FileExistsError):entry.evaluate(**args)

    def test_k1_only_no_factorization(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=fixture(Path(tmp),ks=(1,))
            with contextlib.redirect_stdout(io.StringIO()):marker=entry.evaluate(**args)
            self.assertEqual(marker['factorization_count'],0);self.assertEqual(marker['oof_episodes'],0)


def test_actual_exporter_to_real_probe(export_fixture):
    import export_d92_branch_support_features as exporter
    args, *_ = export_fixture
    with contextlib.redirect_stdout(io.StringIO()):
        exporter.export(**args)
        result = entry.evaluate(support_features=args['output'],capsule=args['capsule'],output=args['output'].parent/'probe',
            expected_capsule_id=args['expected_capsule_id'],expected_checkpoint_sha256=args['expected_checkpoint_sha256'],
            expected_model_seed=args['seed'],config=dict(algorithm=entry.FROZEN_CONFIG,
                matrix=dict(receivers=['target-rx'],scenarios=['test-scene'],ks=[1,2],new_counts=[1],support_seeds=[12])))
    assert result['episodes'] == 2 and result['factorization_count'] == 12
    assert sum(result['payload_audit']['metadata_file_bytes'].values()) > 0
    assert result['query_rows_used'] == 0


if __name__=='__main__':unittest.main()
