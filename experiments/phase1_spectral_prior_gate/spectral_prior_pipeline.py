"""One-owner source-only spectral-prior experiment; no target test."""
import argparse,json,subprocess,sys
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run-root',type=Path,required=True);ap.add_argument('--commit',required=True);ap.add_argument('--device',default='cpu');a=ap.parse_args()
release=Path(__file__).resolve().parents[2]
if (release/'release_commit.txt').read_text().strip()!=a.commit:raise ValueError('Pinned spectral-prior release required')
if a.run_root.exists():raise FileExistsError('Immutable run exists')
a.run_root.mkdir(parents=True)
for name,smoke in (('source_smoke',True),('source',False)):
 cmd=[sys.executable,'-X','utf8','-u',str(Path(__file__).with_name('spectral_prior_source.py')),'--root',str(a.root),'--out',str(a.run_root/name),'--commit',a.commit,'--device',a.device]
 if smoke:cmd.append('--smoke')
 print('PIPELINE_EXEC',json.dumps(cmd),flush=True);subprocess.run(cmd,check=True)
selection=json.loads((a.run_root/'source/source_selection.json').read_text(encoding='utf8'));result=dict(status='SOURCE_SELECTION_COMPLETE',selected=selection['selected'],source_only=True,target_predictions=False)
with (a.run_root/'completion.json').open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print('SPECTRAL_PRIOR_PIPELINE_COMPLETE',json.dumps(result),flush=True)
