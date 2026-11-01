"""Build source phase averaging in DURA's observed invariant output frame."""
from pathlib import Path

p = Path(__file__).parent
source = (p / 'phase1_source_equivariant_dura.py').read_text(encoding='utf8')
changes = [
    ('back=torch.complex(out[:,0],out[:,1])*complex(np.cos(phase),-np.sin(phase))', 'back=torch.complex(out[:,0],out[:,1])'),
    ("test=np.array([1+2j,-.5+.2j]);averaged=np.mean(np.stack([(test*np.exp(1j*t))*np.exp(-1j*t) for t in phases]),axis=0);assert np.max(abs(averaged-test))<1e-12", "test=np.array([1+2j,-.5+.2j]);direct=np.mean(np.stack([test for t in phases]),axis=0);wrong=np.mean(np.stack([test*np.exp(-1j*t) for t in phases]),axis=0);assert np.max(abs(direct-test))<1e-12 and np.max(abs(wrong))<1e-12"),
    ("kind='phase_equivariant_dura'", "kind='phase_invariant_dura'"),
    ('SOURCE_ONLY_EQUIVARIANT_DURA', 'SOURCE_ONLY_INVARIANT_DURA'),
    ('SOURCE_EQUIVARIANT_DURA ', 'SOURCE_INVARIANT_DURA '),
    ('VERIFIED_SOURCE_EQUIVARIANT_DURA_SMOKE', 'VERIFIED_SOURCE_INVARIANT_DURA_SMOKE'),
    ('SAVED_SOURCE_EQUIVARIANT_DURA', 'SAVED_SOURCE_INVARIANT_DURA'),
    ('phase1_source_equivariant_dura.json', 'phase1_source_invariant_dura.json'),
    ('Every candidate is a waveform produced by averaging inverse-rotated DURA outputs;', 'Every candidate averages DURA outputs in the observed common output frame without inverse rotation; this design follows inspection of these same source records and is post-hoc;'),
]
for old, new in changes:
    assert source.count(old) == (2 if old == 'SOURCE_EQUIVARIANT_DURA ' else 1), old
    source = source.replace(old, new)
dest = p / 'phase1_source_invariant_dura.py'
assert not dest.exists()
compile(source, str(dest), 'exec')
dest.write_text(source, encoding='utf8', newline='\n')
print('CREATED invariant-frame phase-average source pilot', dest)
