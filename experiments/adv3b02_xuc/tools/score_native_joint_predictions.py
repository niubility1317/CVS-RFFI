"""Independent scorer: join complete fixed predictions to truth by opaque ID."""
import argparse
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from cvsrffi.xuc_fusion.joint_metrics import task_metrics

def read(path):
    if path.suffix=='.json':
        doc=json.loads(path.read_text(encoding='utf-8'));rows=doc['records']
        if rows and 'predicted_class' in rows[0]:
            return [dict(sample_id=r['sample_id'],scene=r['scenario'],prediction=r['predicted_class']) for r in rows]
        if rows and 'label' in rows[0]:
            from cvsrffi.truth_last import stable_sample_id
            metadata={}
            for tx in range(6):
                for rx in [0,2,5,7,9,10,11]:
                    for day in range(4):
                        for sig in range(1000):
                            sid=stable_sample_id(f'tx{tx}:rx{rx}:day{day}:eq1:sig{sig}',split_binding=doc['split_binding'])
                            metadata[sid]=(tx,rx,day)
            if set(metadata)!={r['sample_id'] for r in rows}:raise ValueError('FULL truth physical metadata mismatch')
            expanded=[]
            for r in rows:
                tx,rx,day=metadata[r['sample_id']]
                if tx!=r['label']:raise ValueError('truth metadata mismatch')
                for scene in ['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']:
                    expanded.append(dict(sample_id=r['sample_id'],scene=scene,truth=tx,rx=rx,day=day))
            return expanded
        raise ValueError('unknown JSON prediction/truth schema')
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--truth',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--reference',type=Path)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    pred=read(a.predictions);truth=read(a.truth)
    def indexed(rows):
        out={}
        for row in rows:
            key=(row['sample_id'],row['scene'])
            if key in out:raise ValueError('duplicate ID/scene')
            out[key]=row
        return out
    pi,ti=indexed(pred),indexed(truth)
    if set(pi)!=set(ti):raise ValueError('incomplete/mismatched prediction coverage')
    scenes=['clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak']
    if set(k[1] for k in pi)!=set(scenes) or any(sum(k[1]==s for k in pi)!=168000 for s in scenes):
        raise ValueError('expected fixed four-scene FULL scope: 168000 each')
    keys=sorted(pi);ref=indexed(read(a.reference)) if a.reference else None
    if ref is not None and set(ref)!=set(pi):raise ValueError('reference coverage differs')
    result=task_metrics([int(ti[k]['truth']) for k in keys],[int(pi[k]['prediction']) for k in keys],
        [int(ti[k]['rx']) for k in keys],[int(ti[k]['day']) for k in keys],[k[1] for k in keys],
        reference=[int(ref[k]['prediction']) for k in keys] if ref is not None else None)
    result.update(prediction_file=str(a.predictions),truth_join='independent_opaque_ID_after_complete_predictions',
        target_feedback=False,independent_confirmation=False)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
if __name__=='__main__':main()
