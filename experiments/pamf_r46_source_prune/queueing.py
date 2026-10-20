from pathlib import Path
from . import design as d
from experiments.pamf_restoration.runner import read_json

def predecessor_closed(args,*,proc_root=Path('/proc')):
    gpu=d.GPUS[args.arm];root=args.output.parent/d.PREDECESSOR_RUN
    marker=root/d.PREDECESSOR_ROW[gpu]/'complete.json'
    if not marker.is_file():return False
    if read_json(marker).get('status')!='ANALYZED' or read_json(root/'launch.json').get('commit')!=d.PREDECESSOR_COMMIT:raise ValueError('Fixed completed R53 predecessor required')
    for item in read_json(root/'workers.json')['workers']:
        if item['GPU']!=gpu:continue
        proc=proc_root/str(item['pid'])
        try:
            state=(proc/'stat').read_text().rsplit(')',1)[1].split()[0];argv=(proc/'cmdline').read_bytes().decode().split(chr(0))
            if state!='Z' and str(root) in argv and d.PREDECESSOR_MODULE in argv:return False
        except FileNotFoundError:pass
    return True
