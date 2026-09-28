"""Read-only resource and output-path preflight for the two fixed MVKME runs."""
import argparse
import json
from pathlib import Path
import subprocess
from read_d92_run import FLAGS

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--specs',nargs=2,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    specs=[json.loads((ROOT/path).read_text(encoding='utf-8')) for path in args.specs]
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
    out=args.output
    with out.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(value))


if __name__=='__main__':main()
