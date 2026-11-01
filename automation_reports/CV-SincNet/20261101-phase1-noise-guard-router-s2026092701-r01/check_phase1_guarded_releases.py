import subprocess
from pathlib import Path

remote=r'''
from pathlib import Path
import hashlib,json
root=Path('/home/szu2070436088/2510044040/CV-SincNet/releases')
base='20261101-phase1-noise-guard-router-s2026092701-r01-release'
expected='0c54f2d155617422e6dfc72766402dcf84d2addcd822f227ed813538b6b8620a'
checks={}
for suffix in ('','-v2'):
 release=root/(base+suffix);asset=release/'experiments/pamf_phase1_flow/guarded_router_weights.json'
 assert (release/'release_commit.txt').read_text().strip()=='a055a0c556a1a0dda1c34aadfe7c6eb86c56a353'
 checks[release.name]=hashlib.sha256(asset.read_bytes()).hexdigest()
assert checks[base]!=expected and checks[base+'-v2']==expected
print(json.dumps({'status':'VERIFIED','initial_release':'FAILED_ASSET_HASH','corrected_release':'VERIFIED_ASSET_HASH','checks':checks,'test_IQ_reads':0,'test_truth_reads':0}))
'''
p=subprocess.run(['ssh','-F','E:/type10-7/tools/n607_ssh_config','-o','BatchMode=yes','-o','ConnectTimeout=10','N607','python3','-'],input=remote.encode(),capture_output=True,timeout=30)
print(p.stdout.decode('utf8'));print(p.stderr.decode('utf8'))
if p.returncode:raise SystemExit(p.returncode)
path=Path(__file__).parent/'phase1_guarded_release_preflight.json'
if path.exists():assert path.read_bytes()==p.stdout
else:path.write_bytes(p.stdout)
