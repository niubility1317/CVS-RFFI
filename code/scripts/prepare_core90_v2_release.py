"""Freeze the authorized F1 and strong-source matrix for N607 release."""
import json
from pathlib import Path
from build_core90_game_matrix import build_matrix,V2_ROWS,V2_SEEDS,STRONG_SOURCE_ROWS,STRONG_SELECTION_RULE
from cvsrffi.game_tracking.config import parse_args

RUN='core90_game_v2_20260911_r3'
ROOT='/home/szu2070436088/2510044040/CV-SincNet'
RELEASE=ROOT+'/releases/'+RUN

def main():
    repo=Path(__file__).resolve().parents[2]
    folder=repo/'code/configs'/RUN
    folder.mkdir(parents=True,exist_ok=False)
    common=dict(runs_root=ROOT+'/runs/'+RUN,dataset_path=ROOT+'/Dataset_WigSig/ManySig.pkl')
    rows=build_matrix(rows=V2_ROWS,seeds=V2_SEEDS,**common)+build_matrix(rows=STRONG_SOURCE_ROWS,seeds=(392005,),**common)
    for row in rows:
        path=folder/(row['run_id']+'.json')
        path.write_text(json.dumps(row['config'],indent=2)+'\n',encoding='utf-8')
        args=parse_args(['--output_dir',row['config']['output_dir'],'--game_config_json',str(path)])
        assert args.from_scratch and not args.baseline_ckpt and not args.game_resume
        assert args.epochs==200 and args.game_evidence_version==2 and args.game_deterministic
        assert args.game_no_audit and args.game_control=='off' and not args.amp
        row['config_path']=RELEASE+'/code/configs/'+RUN+'/'+path.name
        row['argv']=['code/SSDG/train_core90_game.py','--output_dir',row['config']['output_dir'],'--game_config_json',row['config_path']]
    manifest=dict(schema='core90_game_matrix_v2',status='CONFIGURED_NOT_LAUNCHED',source_development_only=True,target_evaluation=False,
        checkpoint_selection='fixed_final_E200',split_seed=392002,runs=rows,strong_source_selection_rule=STRONG_SELECTION_RULE,
        strong_candidate_status='SOURCE_CANDIDATES_NOT_SELECTED_OR_CONFIRMED',
        b8_implementation_acceptance='PASS_FIXED_CONTEXT_FP32_18_LONG_ENDPOINTS_NOT_PERFORMANCE',
        release=RELEASE,run_id=RUN)
    (folder/'matrix.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='PASS',rows=len(rows),matrix=str(folder/'matrix.json'))))

if __name__=='__main__':main()
