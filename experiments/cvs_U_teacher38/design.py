"""Frozen source-only default-head teacher qualification; no training or target."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
PROJECT=Path('/home/szu2070436088/2510044040/CV-SincNet')
RUN='20261018-diagnostic-FcrossRX-clean-teacher-sourceV-e30-s2026092701-m2-r01'; RELEASE='cvs_FcrossRX_clean_teacher_source_20261018_r01'; OWNER='codex/root/Uteacher38'
BASE=PROJECT/'runs'/RUN
SOURCE=PROJECT/'runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'
REFERENCE=PROJECT/'releases/cvs_FcrossRX_cleanretain_E30_20261017_r01'
PANEL=PROJECT/'runs/20261017-diagnostic-FcrossRX-fullsource-residual-s2026092701-m3-r01/diagnostic'
E20=PROJECT/'runs/20261017-phase1-F-crossRX-clock-factorial-e20-s2026092701-m3-r01/B01-s2026092701/source'
E30=PROJECT/'runs/20261017-phase1-F-crossRX-clock-factorial-e20plus10-s2026092701-m5-r03/B01-s2026092701/source'
SCENES=('practical_low_suburban','practical_mid_urban','practical_low_urban')
RX=(1,3,4,6,8)
RULE=dict(tau=.95,min_precision=.99,min_coverage=.75,min_RX_precision=.98,min_transfer_opportunity=.10)
POLICY=dict(cudnn_allow_tf32=False,cuda_matmul_allow_tf32=False,cudnn_benchmark=False,cudnn_deterministic=False,matmul_precision='highest')
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def rows():return [dict(row_id='E20',source=E20,epochs=20,segment_epochs=20,segment_updates=4440),dict(row_id='E30',source=E30,epochs=30,segment_epochs=10,segment_updates=2220)]
def validate_metadata(row,contract,expected,init,args,terminal):
 if row not in rows():raise ValueError('Only both fixed own-reference endpoints; no checkpoint selection')
 if contract['role_ids']!=expected['role_ids'] or {k:len(v) for k,v in contract['role_ids'].items()}!={'L_s':6300,'U_s':56700,'V':27000}:raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
 if init.get('target_contact') is not False or init.get('source_roles')!='EXACT_MATCH' or args.get('teacher_ckpt') or args.get('freeze_backbone') or args.get('source_fold_held_rx') is not None:raise ValueError('CHECKPOINT_PROVENANCE_UNVERIFIED')
 if args.get('checkpoint_selection')!='final_only' or int(args['epochs'])!=row['segment_epochs']:raise ValueError('Not fixed source final checkpoint')
 if row['row_id']=='E20':
  if not init.get('scratch_only') or init.get('checkpoint_sources') or init.get('ancestors') or not args.get('from_scratch') or args.get('baseline_ckpt') or terminal.get('target_access') or terminal.get('epoch')!=20 or terminal.get('optimizer_steps')!=4440:raise ValueError('Parent must be legal scratch fixed E20')
 else:
  if init.get('scratch_only') or init.get('parent_epochs')!=20 or init.get('incremental_epochs')!=10 or init.get('cumulative_epochs')!=30 or args.get('from_scratch') or args.get('baseline_ckpt')!=str(E20/'final_ssdg.pth') or terminal.get('selection_source')!='training_final_only' or terminal.get('selected_checkpoint')!=str(E30/'final_ssdg.pth') or not terminal.get('selected_checkpoint_exists'):raise ValueError('Only its own fixed E20 + E10; technical heldout closure failure retained')
 return True
