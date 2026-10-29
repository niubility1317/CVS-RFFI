import importlib
import json
from pathlib import Path
import threading

import pytest
import test_pamf_phase3_source_embedding_launch as old


def module():
    path = Path(__file__).resolve().parents[1] / 'experiments/pamf_phase3_open_world/source_embedding_budget_launch.py'
    assert path.is_file(), 'Five-stage budget launcher is not implemented'
    return importlib.import_module('experiments.pamf_phase3_open_world.source_embedding_budget_launch')


def runtime(root, m):
    rt = old.EmbeddingLaunchTests().runtime(root)
    rt.update(run_id=m.RUN_ID, run_root=str(root / m.RUN_ID)); return rt


def identity(m, rt, arm):
    return dict(run_id=m.RUN_ID, row_id=arm + '-s2026092701', arm=arm,
        code_commit=rt['code_commit'], checkpoint=str(Path(rt['run_root']) / 'rows' / arm / 'train/final.pt'),
        updates=2220, cumulative_updates=13320)


def marker(m, rt, arm, stage, eligible=False):
    value = dict(source_only=True, target_reads=0, arm=arm, encoder_state_identity=identity(m, rt, arm),
        new_encoder_updates=0, new_identity_head_updates=0)
    if stage == 'train': value.update(status='SOURCE_OPEN_EMBEDDING_FROZEN', updates=2220, parent_updates=11100,
        cumulative_updates=13320, default_head_joint=True, identity_head_replaced=False, optimizer_reset=True,
        model_classes=6, parameters=232993, U_reads=0, validation_reads=0, novel_validation_reads=0,
        OE_truth_reads=0, L_sampled_exposures=284160, As_generated_exposures=142080,
        L_forward_packet_views=1704960, OE_sampled_exposures=142080 if arm=='uniform_oe' else 0, lr=.0001)
    elif stage == 'export': value.update(status='SOURCE_BUDGET_BRANCH_FEATURES_EXPORTED',
        head_changed=False, default_head_replaced=False, counts=m.exporter.COUNTS, model_classes=6,
        parameters=232993, optimizer_updates=0, original_U_s_reads=0, OE_truth_reads=0)
    elif stage == 'fit': value.update(status='SOURCE_BUDGET_READOUTS_FROZEN', geometry_fitted=True,
        descriptor_schema=m.run.consensus.DESCRIPTOR_SCHEMA, readout_training_regime=m.run.REGIME,
        parameter_counts={'linear_oe':101,'mlp16_oe':1633}, readout_steps_per_model=200,
        source_validation_reads=0, novel_validation_reads=0, OE_truth_reads=0)
    elif stage == 'predict': value.update(status='SOURCE_BUDGET_READOUT_PREDICTIONS_COMPLETE',
        unknown_truth_read=False, query_fit=False, default_head_replaced=False, method_names=list(m.METHODS),
        source_V_labels_read=True, source_known_label_calibration=True, encoder_updates=2220,
        identity_head_updates=2220, new_readout_updates=0, readout_steps_per_model=200,
        known_count=27000, novel_count=3150, count=30150, calibration_schema=m.run.calibration.SCHEMA)
    else: value.update(status='SCORED_SOURCE_BUDGET_READOUT_COMPLETE', truth_last=True, final_phase3_complete=False,
        eligible_methods=['mlp16_oe'] if eligible else [], selected_method='mlp16_oe' if eligible else None,
        next_step=m.PASS_NEXT if eligible else 'NO_PROMOTION')
    return value


def test_budget_runtime_fixed_and_five_commands_keep_truth_only_in_scorer(tmp_path):
    m = module(); rt = runtime(tmp_path, m); m.validate_runtime(rt)
    for arm in m.ARMS:
        train, export = m.configurations(rt, Path(rt['run_root']), arm)
        assert train['steps']==2220 and train['lr']==.0001 and train['L_seed']==2026092701
        assert ('diverse_data_root' in train)==(arm=='uniform_oe')
        assert set(export)==m.exporter.KEYS and Path(export['checkpoint']).parent.name=='train'
        commands=m.commands(rt,Path(rt['run_root']),arm);assert tuple(commands)==m.STAGES
        assert all('--novel-truth' not in commands[s] for s in m.STAGES[:-1])
        assert '--novel-truth' in commands['score']
    for change in [{'steps':222}, {'retention':.99}, {'max_added_workers_per_gpu':3}]:
        with pytest.raises(ValueError):m.validate_runtime(dict(rt,**change))
    with pytest.raises(ValueError):m.validate_runtime(dict(rt,model=dict(rt['model'],steps=222)))
    bad=marker(m,rt,'supervised_only','train');bad['updates']=222
    with pytest.raises(ValueError):m._validate_marker('train',bad,'supervised_only')


@pytest.mark.parametrize('failed,bad_marker,eligible', [(None,False,False),(None,False,True),('supervised_only',True,True)])
def test_two_rows_five_stages_capacity_two_only_gpu_train_export_and_partial_no_promotion(tmp_path,monkeypatch,failed,bad_marker,eligible):
    m=module();rt=runtime(tmp_path,m);seen=[];barrier=threading.Barrier(2);lock=threading.Lock();active=[0,0]
    class Child:
        pid=9191
        def __init__(self,command,**kw):
            if command[4].endswith('source_open_embedding_train') or command[4].endswith('source_embedding_budget_export'):
                stage='train' if command[4].endswith('source_open_embedding_train') else 'export'
                cfg=json.loads(Path(command[command.index('--config')+1]).read_text());output=Path(cfg['output_root'])
            else:
                stage=command[5] if command[4].endswith('source_embedding_budget_run') else 'score'
                output=Path(command[command.index('--output')+1])
            self.stage,self.arm,self.output=stage,output.parent.name,output
            with lock:seen.append((self.arm,stage,kw['env']['CUDA_VISIBLE_DEVICES']));active[0]+=1;active[1]=max(active)
        def wait(self):
            if self.stage=='train':barrier.wait(timeout=5)
            self.output.mkdir();value=marker(m,rt,self.arm,self.stage,eligible)
            if self.arm==failed and self.stage=='train':
                value['updates']=222;(self.output/'partial.pt').write_text('preserve')
            (self.output/'complete.json').write_text(json.dumps(value))
            with lock:active[0]-=1
            return 0
    monkeypatch.setattr(m,'select_gpus',lambda rt:[0,0]);monkeypatch.setattr(m.subprocess,'Popen',Child)
    monkeypatch.setattr(m.os,'kill',lambda *a,**k:pytest.fail('Never stop foreign work'))
    result=m.dispatch(rt);assert active[1]==2
    for arm in m.ARMS:
        actual=[(s,g) for a,s,g in seen if a==arm]
        expected=[('train','0')] if arm==failed else [('train','0'),('export','0'),('fit',''),('predict',''),('score','')]
        assert actual==expected
    assert result['foreign_processes_stopped']==0 and not result['final_phase3_complete']
    if failed:
        assert result['status']=='PARTIAL_SOURCE_ONLY' and result['selected_candidate'] is None
        assert (Path(rt['run_root'])/'rows/supervised_only/train/partial.pt').is_file()
    else:
        assert result['status']=='ANALYZED_SOURCE_ONLY'
        assert result['selected_candidate']==('supervised_only::mlp16_oe' if eligible else None)
    with pytest.raises(FileExistsError):m.dispatch(rt)
