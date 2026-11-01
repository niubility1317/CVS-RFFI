"""Build fresh-source replay with all simulator SNR metadata replaced by zeros."""
import ast
from pathlib import Path

p = Path(__file__).parent
source = (p / 'phase1_source_gated_low_snr_probe.py').read_text(encoding='utf8')
tree = ast.parse(source)
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'inner' for t in n.targets))
old_inner = ast.literal_eval(node)
new_inner = old_inner
for old, new in (("snrs.append(float(meta['signal_to_added_noise_db']))", 'snrs.append(0.)'), ("if view=='clean':received=clean_iq;snrs=[100.]*n", "if view=='clean':received=clean_iq;snrs=[0.]*n")):
    assert new_inner.count(old) == 1
    new_inner = new_inner.replace(old, new)
assert source.count('inner=' + repr(old_inner)) == 1
source = source.replace('inner=' + repr(old_inner), 'inner=' + repr(new_inner), 1)
source = source.replace('phase1_source_gated_low_snr_probe.json', 'phase1_source_gated_low_snr_isolation.json')
source = source.replace('SAVED_SOURCE_GATED_LOW_SNR', 'SAVED_SOURCE_GATED_LOW_SNR_ISOLATION')
dest = p / 'verify_phase1_source_gated_low_snr.py'
assert not dest.exists()
compile(new_inner, '<fresh-gated-isolation>', 'exec')
compile(source, str(dest), 'exec')
dest.write_text(source, encoding='utf8', newline='\n')
print('CREATED fresh-source SNR isolation replay', dest)
