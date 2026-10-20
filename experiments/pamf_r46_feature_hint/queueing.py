from pathlib import Path
from . import design as d
from experiments.pamf_restoration.runner import read_json

def predecessor_closed(args,*,proc_root=Path('/proc')):
    gpu=d.GPUS[args.arm];root=args.output.parent/d.PREDECESSOR_RUN;arm=d.PREDECESSOR_ROW[gpu]
    if (root/'failure.json').is_file() or (root/arm/'failure.json').is_file():raise ValueError('Predecessor technical failure;preserve new output,no automatic retry')
    marker=root/arm/'complete.json'
    if not marker.is_file():return False
    if read_json(marker).get('status')!='ANALYZED' or read_json(root/'launch.json').get('commit')!=d.PREDECESSOR_COMMIT:raise ValueError('Fixed completed R58 predecessor required')
    owned=[v for v in read_json(root/'workers.json')['workers'] if v['GPU']==gpu and v['arm']==arm]
    if len(owned)!=1:raise ValueError('Exactly one registered same-card predecessor worker required')
    proc=proc_root/str(owned[0]['pid'])
    try:
        state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0];argv=(proc/'cmdline').read_bytes().decode().split(chr(0))
        if state!='Z' and root.name in ' '.join(argv) and d.PREDECESSOR_MODULE in argv:return False
    except FileNotFoundError:pass
    return True
