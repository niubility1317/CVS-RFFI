import json
from pathlib import Path
import torch
from cvsrffi.xuc_fusion.response_config import make_row
from cvsrffi.xuc_fusion.runtime import resolve_args,train
from ir_fixtures import assert_nested


def test_native_entrypoint_pause_and_resume(tmp_path):
    root=Path(__file__).resolve().parents[1]
    recipe=json.loads((root/'configs/core90_recipe_reference.json').read_text(encoding='utf-8'))
    row=make_row('ir_resume_entrypoint',dict(method='BR_IR_EG'))
    def args(path):
        a=resolve_args(recipe,row,dataset='NOT_READ',output=path,device='cpu',synthetic=True,test_epochs=2,test_steps=1)
        a.joint['diagnostic_interval']=0;a.joint['probe_interval']=0
        return a
    train(args(tmp_path/'full'),row)
    part=args(tmp_path/'part');part.xuc_stop_after_epoch=1;train(part,row)
    resumed=args(tmp_path/'resumed');resumed.xuc_resume=str(tmp_path/'part/latest_ssdg.pth');train(resumed,row)
    a=torch.load(tmp_path/'full/final_ssdg.pth',weights_only=False)
    b=torch.load(tmp_path/'resumed/final_ssdg.pth',weights_only=False)
    for key in ('model','ema','optimizer','prototype'):assert_nested(a[key],b[key],1e-6,1e-4)
    assert_nested(a['solver'],b['solver'])
    assert torch.equal(a['resume_state']['rng'].cpu,b['resume_state']['rng'].cpu)
    assert b['step']==2 and b['resume_lineage']['restored_step']==1
    assert 'br_history' in b['solver'] and b['daot_rc4']['commits']==2
    # The actual launcher defaults must publish both detailed data formats.
    for directory, epochs in [('full',[1,2]),('part',[1]),('resumed',[2])]:
        path=tmp_path/directory
        compact=[json.loads(line) for line in (path/'training_metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        raw=[json.loads(line) for line in (path/'logs.jsonl').read_text(encoding='utf-8').splitlines()]
        assert [r['epoch'] for r in compact]==epochs
        assert (path/'training_metrics.csv').is_file() and (path/'training_config.json').is_file()
        config=json.loads((path/'training_config.json').read_text(encoding='utf-8'))
        assert config['actual_optimizer']['type']=='AdamW'
        assert config['actual_optimizer']['parameter_groups'][0]['betas']==[.9,.999]
        assert '[VAL-SOURCE]' in (path/'training.log').read_text(encoding='utf-8')
        for metric, original in zip(compact,raw):
            assert metric['train_loss_mean']==original['mean_loss']
            assert metric['source_val_accuracy']==original['source_validation']['accuracy']
            assert metric['target_accuracy'] is None and metric['epoch_seconds']>0
