"""Fresh-process source metadata check; no query/score/weight access."""
SCRIPT=r'''
import sys,types,json,os
from pathlib import Path
c=CONFIG;dependency,fold,release=(Path(c[k]) for k in ('dependency','fold','original'))
sys.path[:0]=[str(dependency),str(dependency/'code'),str(dependency/'experiments/adv3b02_xuc/code')]
for suffix,root in (('cvs_sigtime_rdrop_legacy',release),('cvs_rdrop_fold20',release),('cvs_sigtime_locked20',fold),('cvs_rx_u_increment20',fold)):
 name='experiments.'+suffix;p=types.ModuleType(name);p.__path__=[str(root/'experiments'/suffix)];sys.modules[name]=p
from experiments.cvs_sigtime_rdrop_legacy import design as d
from experiments.cvs_sigtime_rdrop_legacy.provenance import source_proof
from experiments.cvs_sigtime_rdrop_legacy.gate import qualify
assert (release/'release_commit.txt').read_text().strip()==c['model_commit'] and os.environ.get('CUDA_VISIBLE_DEVICES')==''
values={}
for row in d.rows():
 if row['epochs']!=50 or row['model_seed']!=d.SEEDS[0]:continue
 config=d.config(row);p=source_proof(config);values[row['mode']]=dict(done=d.read(p/'completion.json'),mechanism=d.read(p/'source_training_mechanism.json'))
gate=qualify(50,values);assert gate['qualified'] and gate==d.read(d.BASE/'source_gates.json')['50']
for rid in c['remaining']:
 row=next(r for r in d.rows() if r['row_id']==rid);p=Path(d.config(row)['output_root'])
 assert not p.exists() and not (d.BASE/'logs'/(rid+'-train_predict.log')).exists()
print(json.dumps(dict(status='VERIFIED',source_only=True,E50_gate=gate,unstarted_rows=c['remaining'],model_code_commit=c['model_commit'],checkpoint_deserializations=0,IQ_reads=0,truth_reads=0,target_metrics_read=False)))
'''
