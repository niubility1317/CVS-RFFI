"""Complete-log, same-row analysis of the fixed same-IQ CSIL experiment."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.cvs_sameiq_distill30.publish import remote
from experiments.cvs_sameiq_distill30 import design as d

SCRIPT=r'''
import csv,gzip,json,math,re
from pathlib import Path
import numpy as np
c=CONFIG;root=Path(c['project'])/'runs'/c['run_id']
def read(p):return json.loads(p.read_text())
if not (root/'complete.json').is_file():raise ValueError('Training/prediction/scoring still incomplete')
summary=read(root/'automatic_summary.json');result={'complete':read(root/'complete.json'),'summary':summary,'logs':{},'teacher_repair':{}};seq={}
def records(path):
 with gzip.open(path,'rt') if str(path).endswith('.gz') else path.open() as f:return [json.loads(line) for line in f if line.strip()]
for row,label in [('C01-s2026092701','CE'),('D01-s2026092701','CSIL')]:
 source=root/row/'source';reference=records(source/'reference_steps.jsonl.gz');shape=records(source/'shape_steps.jsonl.gz');kd=records(source/'distill_steps.jsonl.gz');unit=records(source/'unit_steps.jsonl.gz');epochs=records(source/'metrics_epoch.jsonl');seq[label]=records(source/'training_sequence.jsonl.gz')
 if any(len(x)!=2220 for x in (reference,shape,kd,unit)) or len(epochs)!=10:raise ValueError('Complete fixed training records missing')
 stdout=(root/'logs'/(row+'.log')).read_text();errors=[]
 for line in stdout.splitlines():
  if re.search(r'Traceback|CUDA out of memory|FloatingPointError|^Killed$|RuntimeError|ModuleNotFoundError',line):errors.append(line)
 formula=[];curves=[]
 for r,s,k in zip(reference,shape,kd):
  expected=.5*r['common_RD']+.5*r['extra_RD']+.25*s['shape_RD']+r['SupCon_weight']*r['SupCon']+(.25*6*k['KD'] if label=='CSIL' else 0)
  formula.append(abs(r['total_L']-expected))
  if not all(math.isfinite(float(v)) for v in [r['total_L'],k['KD'],k['As_KL_removed']]):raise ValueError('Nonfinite full scalar records')
  if k['teacher_packets']!=64 or k['new_student_NN'] or not k['all_As_CE'] or not k['same_actual_IQ']:raise ValueError('Actual fixed teacher/IQ/supervision count differs')
 for e in range(10):
  rr=reference[e*222:(e+1)*222];kk=kd[e*222:(e+1)*222]
  mean=lambda xs:sum(xs)/len(xs)
  curves.append(dict(epoch=e+21,total_L=mean([r['total_L'] for r in rr]),AsKL_removed=mean([r['As_KL_removed'] for r in kk]),KD=mean([r['KD'] for r in kk]),teacher_correct_fraction=sum(r['teacher_correct'] for r in kk)/(222*64)))
 result['logs'][label]=dict(full_records=dict(reference=len(reference),shape=len(shape),distill=len(kd),unit=len(unit),native_epochs=len(epochs),sequence=len(seq[label])),stdout_lines=len(stdout.splitlines()),all_stdout_errors=errors,max_formula_absolute_error=max(formula),curves=curves,native_epochs=epochs,actual=read(source/'result.json'),optimizer=read(source/'optimizer_initialization.json'))
result['sequence_C_equals_D']=seq['CE']==seq['CSIL']
old=Path(c['project'])/'runs/20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01/B01-s2026092701/source_qualification/predictions.npz'
teacher=Path(c['project'])/'runs/20261019-diagnostic-FcrossRX-sameIQ-experts-s2026092701-m2-r02/csil/predictions.npz'
with np.load(old,allow_pickle=False) as f:y=np.tile(f['TX'],2);base=f['satellite'][:,2].argmax(-1).reshape(-1);ids=f['ids'].copy()
with np.load(teacher,allow_pickle=False) as f:
 if not np.array_equal(f['ids'],ids):raise ValueError('Same-observation teacher panel order differs')
 t=f['satellite'][:,2].argmax(-1).reshape(-1)
potential=(t==y)&(base!=y);risk=(t!=y)&(base==y)
for row,label in [('C01-s2026092701','CE'),('D01-s2026092701','CSIL')]:
 with np.load(root/row/'source_qualification/predictions.npz',allow_pickle=False) as f:
  if not np.array_equal(f['ids'],ids):raise ValueError('New source panel order differs')
  v=f['satellite'][:,2].argmax(-1).reshape(-1)
 result['teacher_repair'][label]=dict(teacher_correct_oldPAMF_wrong=int(potential.sum()),new_correct_in_potential=int(((v==y)&potential).sum()),teacher_wrong_oldPAMF_correct=int(risk.sum()),new_wrong_in_risk=int(((v!=y)&risk).sum()))
print(json.dumps(result))
'''

def main(runtime,output):
 runtime=Path(runtime);output=Path(output);output.mkdir(parents=True,exist_ok=False)
 rt=json.loads(runtime.read_text(encoding='utf8'));c=dict(project=rt['project'],run_id=d.RUN)
 x=remote(SCRIPT.replace('CONFIG',repr(c)),rt);(output/'complete_analysis.json').write_text(json.dumps(x,indent=2)+'\n',encoding='utf8')
 render(x,output)
 print(json.dumps(dict(status=x['complete']['status'],output=str(output),sequence_C_equals_D=x['sequence_C_equals_D'],teacher_repair=x['teacher_repair'])))

def render(x,output):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors={'NOAS30':'#999999','AS30':'#315587','CE':'#df8f32','CSIL':'#29917d'}
 labels=['clean','high','mid','low sub','high urb','mid urb','low urb']
 names={'NOAS30':'No As','AS30':'Original As','CE':'As CE control','CSIL':'PAMF + CSIL KD'}
 for label,values in x['summary']['summary'].items():axes[0,0].plot(range(7),[100*values[v]['accuracy'] for v in d.VIEWS],'-o',label=names[label],color=colors[label],ms=4)
 axes[0,0].set(xticks=range(7),xticklabels=labels,ylabel='Accuracy (%)');axes[0,0].legend(frameon=False)
 for label,values in x['logs'].items():
  cs=values['curves'];e=[r['epoch'] for r in cs]
  axes[0,1].plot(e,[r['total_L'] for r in cs],'-o',color=colors[label],label=label)
  axes[1,0].plot(e,[r['KD'] for r in cs],'-o',color=colors[label],label=label)
  axes[1,1].plot(e,[100*r['teacher_correct_fraction'] for r in cs],'-o',color=colors[label],label=label)
 axes[0,1].set(xlabel='Cumulative epoch',ylabel='L objective (includes active KD)')
 axes[1,0].set(xlabel='Cumulative epoch',ylabel='Correct-masked teacher KL / 64')
 axes[1,1].set(xlabel='Cumulative epoch',ylabel='Teacher correct on labeled As (%)')
 for ax in axes.flat:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
 fig.savefig(output/'results_and_full_training.png',dpi=180);fig.savefig(output/'results_and_full_training.pdf');plt.close(fig)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--runtime',required=True);a.add_argument('--output',required=True);v=a.parse_args();main(v.runtime,v.output)
