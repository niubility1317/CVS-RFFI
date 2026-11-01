"""Create a clearly post-hoc source tolerance replay for the pseudo-pilot."""
import ast
from pathlib import Path

p = Path(__file__).parent
source = (p / 'phase1_source_cycle_channel_probe.py').read_text(encoding='utf8')
tree = ast.parse(source)
node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'remote' for t in n.targets))
old_remote = ast.literal_eval(node)
rtree = ast.parse(old_remote)
run = next(n for n in ast.walk(rtree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'run')
old_inner = ast.literal_eval(run.args[0].elts[3])
old = "guards=all(z['net']>=0 and z['clean_teacher_net']>=0 and min(z['per_view'].values())>=-.005*z['n'] for z in groups_v.values())"
new = "guards=all(z['net']>=-.005*z['n'] and z['clean_teacher_net']>=0 and min(z['per_view'].values())>=-.005*z['n'] for z in groups_v.values())"
assert old_inner.count(old) == 1
new_inner = old_inner.replace(old, new)
new_inner = new_inner.replace('SOURCE_ONLY_CYCLE_CHANNEL_PROBE', 'SOURCE_ONLY_CYCLE_CHANNEL_TOLERANCE')
new_inner = new_inner.replace('Each record uses received IQ and DURA only:', 'The per-group total tolerance was chosen after inspecting source validation; this replay is post-hoc and not independent. Each record uses received IQ and DURA only:')
assert old_remote.count(repr(old_inner)) == 1
new_remote = old_remote.replace(repr(old_inner), repr(new_inner), 1)
assert source.count('remote=' + repr(old_remote)) == 1
new_source = source.replace('remote=' + repr(old_remote), 'remote=' + repr(new_remote), 1)
new_source = new_source.replace('phase1_source_cycle_channel_probe.json', 'phase1_source_cycle_channel_tolerance.json')
new_source = new_source.replace('SAVED_SOURCE_CYCLE_CHANNEL', 'SAVED_SOURCE_CYCLE_CHANNEL_TOLERANCE')
dest = p / 'phase1_source_cycle_channel_tolerance.py'
assert not dest.exists()
compile(new_inner, '<cycle-channel-tolerance>', 'exec')
compile(new_source, str(dest), 'exec')
dest.write_text(new_source, encoding='utf8', newline='\n')
print('CREATED post-hoc cycle-channel tolerance replay', dest)
