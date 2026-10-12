"""Scratch student initialization and separately retained pretrained teachers."""
from pathlib import PurePosixPath
from . import design as d


def teacher_path(seed):
    return str(PurePosixPath(d.PROJECT)/'runs'/'20261011-phase1-ce-epoch10-20-30-40-manysig-m12-r01'/f'E20-s{seed}'/'source'/'final_ssdg.pth')


def expected_sources(c):
    """No student checkpoint is loaded; every actual teacher remains an ancestor."""
    if c['parent_checkpoint'] is not None:
        raise ValueError('Scratch student cannot have an initialization checkpoint')
    sources=[teacher_path(seed) for seed in c['teacher']['seeds']]
    if c['checkpoint_sources']!=sources:
        raise ValueError('Method checkpoint sources must retain exactly the registered teachers')
    return sources


def student_initialization(c):
    if c['parent_checkpoint'] is not None or c['initialization_checkpoint_sources']!=[]:
        raise ValueError('Scratch student cannot inherit parent weights')
    return dict(status='SCRATCH_INITIALIZATION',seed=c['model_seed'],checkpoint=None,parent_checkpoint=None,
        initialization_checkpoint_sources=[],ancestors=[],scratch_only=True,scratch_scope='student_identity_only',
        source_roles='EXACT_MATCH',training_target_access=False,reference_E20_loaded=False,
        parent_training_epochs=0,auxiliary_inheritance=False)


def validate(c,teachers):
    seeds=c['teacher']['seeds']
    required=[] if c['arm']=='CE' else [seed for seed in d.SEEDS if seed!=c['model_seed']]
    if seeds!=required or c['teacher']['source']!=('none' if c['arm']=='CE' else 'peer'):
        raise ValueError('Registered other-two teacher definition differs')
    if not isinstance(teachers,dict) or set(teachers)!={str(seed) for seed in seeds}:
        raise ValueError('Teacher ancestry keys differ')
    for key,proof in teachers.items():
        seed=int(key)
        if seed==c['model_seed'] or seed not in d.SEEDS:
            raise ValueError('Student itself or unregistered teacher used')
        wanted=dict(status='VERIFIED',seed=seed,checkpoint=teacher_path(seed),epoch=20,
                    scratch_only=True,ancestors=[],source_roles='EXACT_MATCH',training_target_access=False)
        if any(proof.get(k)!=v for k,v in wanted.items()):
            raise ValueError('Independent E20 teacher ancestry invalid')
    return teachers
