"""Read-only resource and output-path preflight for the two fixed MVKME runs."""
import json
from pathlib import Path
import subprocess
from read_d92_run import FLAGS

ROOT=Path(__file__).resolve().parents[1]


def main():
    specs=[json.loads((ROOT/f'configs/d92_mvkme_repeat_{c}_20260928.json').read_text(encoding='utf-8')) for c in ('rx3','rx1')]
    paths=[]
    for spec in specs:
        release=spec['code']['cwd']
        paths += [release,release+'.tar',spec['execution']['remote_run_root']]
    script='import json,os,shutil,subprocess,getpass,socket\nfrom pathlib import Path\n'
    script+='paths='+repr(paths)+'\n'
    script+='assert not any(Path(p).exists() for p in paths), "Output collision"\n'
    script+='gpu=subprocess.check_output(["nvidia-smi","--query-gpu=index,memory.used,memory.total,utilization.gpu","--format=csv,noheader,nounits"],text=True)\n'
    script+='processes=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader,nounits"],text=True)\n'
    script+='print(json.dumps(dict(status="VERIFIED",user=getpass.getuser(),host=socket.gethostname(),load=os.getloadavg(),cpu_count=os.cpu_count(),disk_free=shutil.disk_usage("/home/szu2070436088/2510044040/CV-SincNet").free,gpu=gpu,processes=processes,absent_paths=paths,source_samples_read=False,query_read=False)))\n'
    result=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True,check=True)
    value=json.loads(result.stdout)
    if value['user']!='szu2070436088':raise ValueError('Unexpected remote user')
    out=Path('E:/type10-7/local_artifacts/d92_upgrade_20260928/mvkme_preflight.json')
    with out.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(value))


if __name__=='__main__':main()
