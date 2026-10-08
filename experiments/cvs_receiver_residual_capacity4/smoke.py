"""Bounded scheduler checks without CUDA, data, or training."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from experiments.cvs_receiver_residual_capacity4.control import choose_gpu, match_worker, process, write, read
from experiments.cvs_receiver_residual_capacity4 import control


def queue_recovery():
    with tempfile.TemporaryDirectory() as folder:
        project=Path(folder);base=project/'runs'/'run';base.mkdir(parents=True)
        rows=[dict(row_id=r,config=dict(output_root=str(base/r/'source'))) for r in ('done','live')]
        d=SimpleNamespace(PROJECT=str(project),RUN='run')
        write(base/'done/source/completion.json',dict(status='SOURCE_TRAINED'))
        receipt=dict(row_id='live',kind='source',pid=31,start_ticks=77,cwd=folder,gpu=0)
        write(base/'launch_source.json',dict(rows=[receipt]))
        calls=[]
        def fake_alive(r):
            calls.append(r['pid'])
            if len(calls)==1:return dict(pid=31)
            write(base/'live/source/completion.json',dict(status='SOURCE_TRAINED'));return None
        with patch.object(control,'alive',fake_alive):
            control.queue(d,None,dict(rows=rows),'unused',Path('unused'),'source')
        state=read(base/'queue_state.json')
        assert state['completed']==['done','live'] and not state['active'] and not state['pending']
        assert read(base/'launch_source.json')['rows']==[receipt]  # no replay/duplicate launch
        (base/'live/source/completion.json').unlink()
        with patch.object(control,'alive',lambda r:None):
            try:control.queue(d,None,dict(rows=rows),'unused',Path('unused'),'source')
            except RuntimeError:pass
            else:raise AssertionError('Failed launched row was retried')


def main():
    caps={0:dict(pids={1,2,3,4},free_mb=23000),1:dict(pids={5,6,7},free_mb=20000),
        2:dict(pids={8},free_mb=11000),3:dict(pids={9,10},free_mb=16000)}
    assert choose_gpu(caps)==3
    assert choose_gpu({0:caps[0],2:caps[2]}) is None
    root=str(Path('/immutable/worker').resolve());package='cvs_receiver_residual_v2';rid='baseline-s2026092702'
    actual=dict(pid=31,start_ticks=77,cwd=root,gpu='2',argv=['python','-u','-m','experiments.'+package+'.source',
        '--config',root+'/experiments/'+package+'/configs/'+rid+'.json'])
    matched=match_worker(actual,root,package,{rid:None})
    assert matched['row_id']==rid and matched['kind']=='source' and matched['start_ticks']==77
    assert match_worker(dict(actual,cwd='/other'),root,package,{rid:None}) is None
    for bad in (dict(actual,gpu=''),dict(actual,argv=actual['argv'][:-1]+['/other/'+rid+'.json'])):
        try:match_worker(bad,root,package,{rid:None})
        except ValueError:pass
        else:raise AssertionError('Bad identity accepted')
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)/'31';p.mkdir();(p/'cwd').mkdir()
        fields=['S']+['0']*18+['77']+['0']*3
        (p/'stat').write_text('31 (worker with spaces) '+' '.join(fields))
        (p/'environ').write_bytes(b'CUDA_VISIBLE_DEVICES=2\0');(p/'cmdline').write_bytes(b'python\0worker\0')
        assert process(31,Path(folder))['start_ticks']==77
        original=Path.read_bytes
        def restricted(path):
            if path.name=='environ':raise PermissionError('transient /proc exit read restriction')
            return original(path)
        with patch.object(Path,'read_bytes',restricted):
            assert control.running(31,Path(folder)) is True
            assert process(31,Path(folder))['start_ticks']==77
        (p/'stat').write_text('31 (worker) Z '+' '.join(fields[1:]))
        assert process(31,Path(folder)) is None
        dest=Path(folder)/'atomic.json';write(dest,dict(test='中文',limit=4));assert read(dest)['limit']==4
    queue_recovery()
    print(json.dumps(dict(status='PASS',checks=['global_capacity4','memory_floor','worker_identity','proc_start_ticks','zombie_exclusion','proc_exit_read_race','atomic_json','adopt_without_replay','failed_row_no_retry'])))


if __name__=='__main__':main()
