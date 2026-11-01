"""Create post-hoc source tolerance replay; never treat it as independent."""
from pathlib import Path

p = Path(__file__).parent
s = (p / 'phase1_source_carrier_regression.py').read_text(encoding='utf8')
old = "guards=all(z['net']>=0 and z['clean_teacher_net']>=0 and min(z['per_view'].values())>=0 for z in groups_v.values())"
new = "guards=all(z['net']>=0 and z['clean_teacher_net']>=0 and min(z['per_view'].values())>=-.005*int(((split==1)&(source_rx==int(group))).sum()) for group,z in groups_v.items())"
assert s.count(old) == 1
s = s.replace(old, new)
s = s.replace('SOURCE_ONLY_CARRIER_REGRESSION', 'SOURCE_ONLY_CARRIER_REGRESSION_TOLERANCE')
s = s.replace('phase1_source_carrier_regression.json', 'phase1_source_carrier_regression_tolerance.json')
s = s.replace('SAVED_SOURCE_CARRIER_REGRESSION', 'SAVED_SOURCE_CARRIER_REGRESSION_TOLERANCE')
s = s.replace('Source clean waveform is a training and quality reference, not an inference input.', 'The 0.5%-per-group tolerance was chosen after inspecting this source validation, so this replay is post-hoc and not independent. Source clean waveform is a training and quality reference, not an inference input.')
dest = p / 'phase1_source_carrier_regression_tolerance.py'
assert not dest.exists()
compile(s, str(dest), 'exec')
dest.write_text(s, encoding='utf8', newline='\n')
print('CREATED post-hoc source carrier tolerance replay', dest)
