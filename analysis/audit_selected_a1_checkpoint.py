import json
import sys
from pathlib import Path, PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
sys.path.insert(0,str(ROOT/'code/scripts'))
import torch
from SSDG.train_ssdg import build_arg_parser
from run_a1_fast_matched_core90 import build_train_command
checkpoint=torch.load(sys.argv[1],map_location='cpu',weights_only=False)
matrix=json.loads((ROOT/'configs/a1_fast_matched_core90_20260908.json').read_text(encoding='utf-8'))
command=build_train_command(matrix,project_root=PurePosixPath('/home/szu2070436088/2510044040/CV-SincNet'),run_root=PurePosixPath('/unused'),row_id=matrix['rows'][0]['id'],row=matrix['rows'][0])
current=vars(build_arg_parser().parse_args(command[3:]))
old=checkpoint['args']
keys=[k for k in current if k.startswith('wisig_')]+['seed','split_mode','labeled_ratio','unlabeled_ratio','source_val_ratio','source_cal_ratio','source_select_ratio','phase1_source_role_protocol','num_classes','sat_seed','sat_view_seed','model_size','model_variant','representation_mode']
report={'epoch':checkpoint['epoch'],'rows':[{ 'key':k,'checkpoint':old.get(k),'a1_config':current[k],'equal':str(old.get(k))==str(current[k])} for k in keys], 'split_info':checkpoint.get('split_info')}
Path(sys.argv[2]).write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='split_info'},ensure_ascii=False,indent=2))
print('split_info keys:',list((checkpoint.get('split_info') or {}).keys()))
