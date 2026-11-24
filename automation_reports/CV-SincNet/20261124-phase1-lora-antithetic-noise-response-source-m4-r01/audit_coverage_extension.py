from pathlib import Path
import json,subprocess
old=Path("automation_reports/CV-SincNet/20261124-diagnostic-lora-frozen-complementarity-source-cpu-r01")
out=Path('automation_reports/CV-SincNet')/"20261124-phase1-lora-antithetic-noise-response-source-m4-r01"
manifest=json.loads((old/'candidate-matrix.json').read_text(encoding='utf-8'))
manifest.append(dict(run="20261124-phase1-lora-antithetic-noise-response-source-m4-r01",family='antithetic',main=['pointwise_half','antithetic_half'],
    extra=['pointwise_coarse','antithetic_coarse','pointwise_full','antithetic_full']))
analysis=(old/'analyze_frozen.py').read_text(encoding='utf-8').rsplit("if __name__ == '__main__':",1)[0]
program=analysis+'\nmain('+repr(manifest)+')\n'
release='/home/szu2070436088/2510044040/CV-SincNet/releases/20261124-phase1-lora-empirical-iq-prior-source-m2-r01-release'
args=['ssh.exe','-F','E:/type10-7/tools/n607_ssh_config','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','N607',
      f'cd {release} && /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -']
p=subprocess.run(args,input=program.encode('utf-8'),capture_output=True,timeout=65,check=False)
if p.returncode:
    raise RuntimeError(p.stderr.decode('utf-8',errors='replace'))
x=json.loads(p.stdout)
previous=json.loads((old/'pair-overlap-with-sensitivity.json').read_text(encoding='utf-8'))
assert x['n']==previous['n']==900 and x['included'][:-1]==previous['included']
assert not x['omitted'] and x['target_reads']==x['new_predictions']==x['fitted_parameters']==0
results={}
for route,case in x['routes'].items():
    results[route]={}
    for key in ('primary','all'):
        before=previous['routes'][route]['choices'][key]['counts']
        after=case['choices'][key]['counts']
        assert all(after['views'][v]>=n for v,n in before['views'].items())
        results[route][key]=dict(before=before,after=after,
             added=sum(after['views'][v]-n for v,n in before['views'].items() if v!='clean'),
             by_view={v:after['views'][v]-n for v,n in before['views'].items()})
result=dict(status='VERIFIED',n=x['n'],target_reads=0,truth_after_validated_predictions=True,
            previous=old.name, additions=manifest[-1], results=results)
with (out/'frozen-coverage-extension.json').open('x',encoding='utf-8',newline='\n') as f:
    f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result))
