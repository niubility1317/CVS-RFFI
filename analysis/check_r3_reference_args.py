import sys,json,math
from pathlib import Path, PurePosixPath
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/scripts'))
from run_a1_r3_cross_rx import r3_cross_rx_matrix
from run_a1_fast_v2 import v2_command
from SSDG import train_ssdg as train
p=Path('E:/type10-7/automation_reports/CV-SincNet/a1_completed_results_20260909_1448/evidence.json')
d=json.loads(p.read_text(encoding='utf-8'))
argv=d['runs']['a1_r3_scratch_s392005_20260908_r1']['rows']['R3_REFERENCE']['launch_config.json']['argv']
m=r3_cross_rx_matrix()
a=train.build_arg_parser().parse_args(argv[3:])
b=train.build_arg_parser().parse_args(v2_command(m,project_root=PurePosixPath('/home/szu2070436088/2510044040/CV-SincNet'),run_root=PurePosixPath('/new'),row=m['rows'][0])[3:])
def equal(x,y):return x==y or (isinstance(x,float) and isinstance(y,float) and math.isnan(x) and math.isnan(y))
diff={k:{'reference':getattr(a,k),'new':getattr(b,k)} for k in vars(a) if not equal(getattr(a,k),getattr(b,k))}
assert set(diff)=={'a1_ecrs_cross_rx_weight','output_dir','phase2_export_path','run_id','candidate_id'},diff
out={'status':'PASS','reference_source':'actual N607 launch_config argv','differences':diff}
Path(__file__).with_name('a1_r3_cross_rx_actual_reference_diff.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
