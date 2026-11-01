"""Exercise fixed blind-SNR gate and source-ID exclusions on real source IQ."""
import ast
import json
import subprocess
from pathlib import Path

p = Path(__file__).parent
tree = ast.parse((p / 'phase1_source_gated_low_snr_probe.py').read_text(encoding='utf8'))
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
inner = ast.literal_eval(node)
old = 'n=len(fixed);assert n>300'
new = old + ";sel=np.concatenate((np.flatnonzero(source_rx==4)[:1],np.flatnonzero(source_rx==6)[:1],np.flatnonzero(source_rx==8)[:2]));fixed=fixed[sel];source_rx=source_rx[sel];source_day=source_day[sel];split=split[sel];n=len(fixed);assert n==4"
assert inner.count(old) == 1
inner = inner.replace(old, new)
source = json.loads((p / 'phase1_source_cycle_low_snr_net.json').read_text(encoding='utf8'))
assert source['selected']['kind'] == 'cycle_low_snr_net'
payload = dict(model_state=source['model_state'], alpha=.25, gate_db=10.)
remote = '''from pathlib import Path
import os,subprocess
root=Path('/home/szu2070436088/2510044040/CV-SincNet');release=root/'releases/20261101-phase1-physical-rich-router-s2026092701-r01-release';native=root/'releases/20261027-phase1-pamf-repl-shuffled-views-m16-r02-release'
env=dict(os.environ,PYTHONPATH=':'.join(map(str,(release,native))),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
subprocess.run(['/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python','-u','-c',INNER],cwd=release,env=env,input=PAYLOAD,text=True,check=True,timeout=300)
'''.replace('INNER', repr(inner), 1).replace('PAYLOAD', repr(json.dumps(payload, separators=(',', ':'))), 1)
proc = subprocess.run(['C:/Program Files/Git/usr/bin/ssh.exe', '-F', 'E:/type10-7/tools/n607_ssh_config', 'N607', 'python3', '-'], input=remote.encode(), capture_output=True, timeout=320)
stdout = proc.stdout.decode('utf8', errors='replace')
print('\n'.join(s for s in stdout.splitlines() if not s.startswith('SOURCE_GATED_LOW_SNR ')))
print(proc.stderr.decode('utf8', errors='replace'))
if proc.returncode:
    raise SystemExit(proc.returncode)
line = next(s[len('SOURCE_GATED_LOW_SNR '):] for s in stdout.splitlines() if s.startswith('SOURCE_GATED_LOW_SNR '))
result = json.loads(line)
assert result['source_ids'] == 4 and result['heldout_ids'] == 2 and result['validation_ids'] == 2
assert result['heldout']['clean']['gate_enabled'] == 0 and result['quality']['clean']['baseline'] == result['quality']['clean']['gated']['aligned_mse']
assert result['predictions_before_source_tx_labels'] and not result['clean_target_used_for_inference'] and not result['simulated_snr_used_for_inference']
for group in (result['validation'], result['heldout']):
    for row in group.values():
        c = row['blind_snr_confusion']
        assert c['true_positive'] + c['false_negative'] == c['true_low']
        assert c['true_positive'] + c['false_positive'] + c['false_negative'] + c['true_negative'] == row['count']
print('VERIFIED_SOURCE_GATED_LOW_SNR_SMOKE', result['source_ids'])
