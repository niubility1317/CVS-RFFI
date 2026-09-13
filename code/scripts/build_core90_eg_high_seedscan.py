"""Freeze six scratch EG/high-LR runs from the matched V2 E/F configurations."""
import json
from copy import deepcopy
from pathlib import Path

RUN='core90_eg_high_seedscan_20260913_r1'

def main():
    root=Path(__file__).resolve().parents[2]
    source=root/'code/configs/core90_game_v2_20260911_r3/matrix.json'
    old=json.loads(source.read_text(encoding='utf-8'))
    matrix={k:old[k] for k in ('schema','status','source_development_only','target_evaluation','checkpoint_selection','split_seed')}
    matrix.update(runs=[],research_scope='SOURCE_ONLY_FOLLOWUP_AFTER_BENCHMARK_EXPOSURE',selection_permitted=False)
    out=root/'code/configs'/RUN
    out.mkdir(parents=True,exist_ok=True)
    def save(path,data):
        if path.exists():
            assert json.loads(path.read_text(encoding='utf-8'))==data, 'Refusing to overwrite changed config'
        else:path.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    remote='/home/szu2070436088/2510044040/CV-SincNet'
    matrix.update(run_id=RUN,release=remote+'/releases/'+RUN)
    for row in old['runs']:
        if row['row'] not in ('V2_E','V2_F'):continue
        r=deepcopy(row)
        name='V2_EG_HIGH_'+('ADV0' if row['row']=='V2_E' else 'ADV035')
        r['row']=name;r['run_id']=name+'_seed'+str(row['seed'])
        r['description']='Complete EG + lr=4e-4; matched V2 seed; scratch E200; source-only exploratory follow-up'
        c=r['config'];c['lr']=0.0004;c['run_name']=r['run_id']
        c['output_dir']=remote+'/runs/'+RUN+'/'+r['run_id']
        r['config_path']=remote+'/releases/'+RUN+'/code/configs/'+RUN+'/'+r['run_id']+'.json'
        r['argv']=['code/SSDG/train_core90_game.py','--output_dir',c['output_dir'],'--game_config_json',r['config_path']]
        assert c['from_scratch'] and not c['baseline_ckpt'] and not c['game_resume']
        assert c['game_solver']=='extragradient' and c['epochs']==200 and c['game_export_source_predictions']
        assert {k for k in set(c)|set(row['config']) if c.get(k)!=row['config'].get(k)}=={'lr','run_name','output_dir'}
        save(out/(r['run_id']+'.json'),c)
        matrix['runs'].append(r)
    assert len(matrix['runs'])==6
    save(out/'matrix.json',matrix)
    # Parse each saved config through the real training parser, without loading data/GPU.
    import sys
    sys.path.insert(0,str(root/'code'))
    from cvsrffi.game_tracking.config import parse_args
    for r in matrix['runs']:
        args=parse_args(['--game_config_json',str(out/(r['run_id']+'.json')),'--output_dir',r['config']['output_dir']])
        assert args.lr==0.0004 and args.game_solver=='extragradient' and args.epochs==200
        assert args.seed==r['seed'] and args.lambda_adv==r['config']['lambda_adv']
        assert args.from_scratch and not args.baseline_ckpt and not args.game_resume and not args.amp
    print('VERIFIED six persisted configs; only lr/name/output differ from matched V2 E/F')

if __name__=='__main__':main()
