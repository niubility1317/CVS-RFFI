"""Read-only N607 extraction. This subset is diagnostic, NOT the CVS capsule."""
import argparse
import io
import json
import subprocess
from pathlib import Path
import numpy as np
from runtime import validate_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists() or out.with_suffix('.json').exists(): raise FileExistsError(out)
    remote = '''import io,pickle,sys,numpy as np
path="/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl"
with open(path,"rb") as f: ds=pickle.load(f)
x=[]; y=[]; ids=[]
for t in range(6):
    a=np.asarray(ds["data"][t][0][0][0],dtype=np.float32)
    if a.shape[0]<192 or a.shape[1:]!=(256,2): raise ValueError(a.shape)
    x.append(a[:192]); y.extend([t]*192)
    ids.extend([f"ManySig:{ds['tx_list'][t]}:{ds['rx_list'][0]}:{ds['capture_date_list'][0]}:eq0:{s}" for s in range(192)])
b=io.BytesIO(); np.savez_compressed(b,x=np.concatenate(x),y=np.array(y),ids=np.array(ids),tx=np.array(ds['tx_list']),rx=np.array(ds['rx_list'][0]),day=np.array(ds['capture_date_list'][0]))
sys.stdout.buffer.write(b.getvalue())
'''
    cmd=['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','N607',
         '/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -']
    result=subprocess.run(cmd,input=remote.encode('utf-8'),capture_output=True,timeout=60)
    if result.returncode: raise RuntimeError(result.stderr.decode('utf-8',errors='replace'))
    with np.load(io.BytesIO(result.stdout),allow_pickle=False) as z:
        train=np.tile(np.arange(192)<128,6); query=~train
        validate_split(z['ids'][train],z['ids'][query])
        if not np.isfinite(z['x']).all(): raise ValueError('Nonfinite real IQ')
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('xb') as f: np.savez_compressed(f,**{k:z[k] for k in z.files},train=train)
        meta={'status':'VERIFIED','source_host':'N607','source_path':'/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl',
              'tx':z['tx'].tolist(),'rx':str(z['rx']),'day':str(z['day']),'equalized':0,
              'train_per_class':128,'query_per_class':64,'old_classes':3,'new_classes':3,
              'split_rule':'front128 train, next64 query; deterministic diagnostic only',
              'claim':'Real WiSig execution acceptance only; no LEO, not formal CVS or original ISSL benchmark',
              'train_ids':z['ids'][train].tolist(),'query_ids':z['ids'][query].tolist()}
    out.with_suffix('.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Independent local readback after network transfer.
    with np.load(out,allow_pickle=False) as z: print(json.dumps({'status':'VERIFIED','shape':list(z['x'].shape),'train':int(z['train'].sum()),'query':int((~z['train']).sum()),'output':str(out)}))

if __name__=='__main__': main()
