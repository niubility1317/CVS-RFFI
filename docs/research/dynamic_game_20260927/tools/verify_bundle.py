"""Portable read-only validation of the published historical analysis bundle."""
from pathlib import Path
import collections, csv, gzip, hashlib, json, math, re, statistics, sys, tokenize
from urllib.parse import unquote

D=Path(__file__).resolve().parents[1]
ROOT=D.parents[2]
OLD=D.parent/'daot_fasttrust_game_20260914'

def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert abs(float(a)-float(b))<1e-10,(a,b)

checks={};errors=[]
manifest=D/'bundle_checksums.csv'
if manifest.exists():
    mm=rows(manifest)
    for m in mm:
        p=ROOT/m['path'];b=p.read_bytes()
        assert len(b)==int(m['bytes']) and hashlib.sha256(b).hexdigest()==m['sha256'],m['path']
    checks['checksummed_files']=len(mm)
else:errors.append('bundle_checksums.csv missing')

# Only current authored documents promise portable links. Legacy source files
# retain their original context and external local/remote path references.
docs=[D/'README.md',D/'RESULTS.md',D/'DELIVERY.md',D/'LITERATURE.md',*sorted((D/'chapters').glob('*.md'))]
link_count=0
for p in docs:
    text=p.read_text(encoding='utf-8')
    assert '\ufffd' not in text,p
    for raw in re.findall(r'\]\(([^)]+)\)',text):
        target=raw.split(' "',1)[0].strip('<>')
        if re.match(r'^(https?://|mailto:|#)',target):continue
        target=unquote(target.split('#',1)[0]);target=re.sub(r':\d+$','',target)
        if not (p.parent/target).exists():errors.append(f'{p.relative_to(D)}: broken link {raw}')
        link_count+=1
checks['current_document_links']=link_count

python_count=0
for base in [D,OLD]:
    for p in base.rglob('*.py'):
        with tokenize.open(p) as f:source=f.read()
        compile(source,str(p),'exec');python_count+=1
checks['python_files_syntax_checked']=python_count

result=D/'records/response_matrix_eval_20260917/results'
ss=rows(result/'summary.csv');curves=rows(result/'source_curves.csv')
assert len(ss)==33 and len(curves)==6600
by={(x['method'],int(x['seed'])):x for x in ss}
for r in ss:
    close(r['leo_mean_accuracy'],statistics.mean(float(r[s+'_accuracy']) for s in ['leo_clear_weak','leo_low_elev_weak','leo_rain_weak']))
    cc=[x for x in curves if x['row_id']==r['row']]
    assert [int(x['epoch']) for x in cc]==list(range(1,201)) and int(cc[-1]['total_step'])==44400
    assert all(math.isfinite(float(x['mean_loss'])) for x in cc)
stats=json.loads((D/'evidence/response_recomputed.json').read_text(encoding='utf-8'))
for p in stats['paired']:
    vals=[100*(float(by[p['a'],s]['leo_mean_accuracy'])-float(by[p['b'],s]['leo_mean_accuracy'])) for s in p['seeds']]
    for a,b in zip(vals,p['delta_pp']):close(a,b)
    close(statistics.mean(vals),p['mean_pp']);close(statistics.stdev(vals),p['sd_pp'])
for item in stats['interaction']:
    s=item['seed'];v={m:100*float(by[m,s]['leo_mean_accuracy']) for m in ['R0-00','R0-01','R0-10','R0-11']}
    close(v['R0-11']-v['R0-10']-v['R0-01']+v['R0-00'],item['interaction_pp'])
checks['response_scored_rows']=33;checks['frozen_source_epoch_records']=6600

full=json.loads(gzip.decompress((D/'evidence/full9_recount.json.gz').read_bytes()))
assert full['status']=='VERIFIED' and full['total_prediction_records']==6048000 and len(full['rows'])==9
final={r['row']:r for r in rows(D/'tables/full9_final_scores.csv')}
for name,x in full['rows'].items():
    assert x['id_coverage_exact'] and x['count']==672000
    for scene,cm in x['confusion'].items():
        assert len(cm)==6 and all(len(rr)==6 and sum(rr)==28000 for rr in cm)
        correct=sum(cm[i][i] for i in range(6));m=x['score']['metrics'][scene]
        assert correct==m['correct'] and sum(map(sum,cm))==m['total']==168000
        close(correct/168000,m['accuracy']);close(m['accuracy']*100,final[name][scene])
checks['full9_prediction_records_recount_evidence']=6048000

epochs=rows(D/'tables/all_available_epoch_records.csv');states=rows(D/'tables/current_run_state.csv')
assert len(epochs)==11133 and len(states)==66
assert len({(r['run'],r['row']) for r in states})==66
held=[r for r in states if r['epochs']=='0'];assert len(held)==5
native=rows(D/'tables/native_amp_skips.csv');assert len(native)==3
for x in native:
    assert int(x['train_skipped_nonfinite_loss'])==0
    assert int(x['train_optimizer_step_applied'])+int(x['train_skipped_nonfinite_grad'])==int(x['attempted_batches'])==44400
checks['remote_epoch_records']=len(epochs);checks['observed_or_registered_rows']=len(states)
print(json.dumps({'status':'FAILED' if errors else 'VERIFIED','checks':checks,'errors':errors},ensure_ascii=False,indent=2))
sys.exit(1 if errors else 0)
