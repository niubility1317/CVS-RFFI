"""Deployment retains teacher ancestry metadata, never U probability tensors."""
from pathlib import PurePosixPath
from . import design as d


def teacher_path(seed):
    return str(PurePosixPath(d.PROJECT)/'runs'/'20261011-phase1-ce-epoch10-20-30-40-manysig-m12-r01'/f'E20-s{seed}'/'source'/'final_ssdg.pth')


def expected_sources(c):
    return [c['parent_checkpoint']]+[teacher_path(seed) for seed in c['teacher']['seeds']]


def validate(c, teachers):
    expected={str(s) for s in c['teacher']['seeds']}
    if set(teachers)!=expected:
        raise ValueError('Teacher ancestry keys differ')
    for key, proof in teachers.items():
        seed=int(key)
        if seed==c['model_seed'] or seed not in d.SEEDS:
            raise ValueError('Student itself or unregistered teacher used')
        wanted=dict(status='VERIFIED',seed=seed,checkpoint=teacher_path(seed),epoch=20,
                    scratch_only=True,ancestors=[],source_roles='EXACT_MATCH',training_target_access=False)
        if any(proof.get(k)!=v for k,v in wanted.items()):
            raise ValueError('Independent E20 teacher ancestry invalid')
    return teachers
