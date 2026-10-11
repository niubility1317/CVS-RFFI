"""Random native student construction; no initial checkpoint is read."""
import torch
from . import design as d
from .provenance import student_initialization,validate as validate_teachers,expected_sources
from experiments.cvs_stage2_rx_risk.model import OriginalIdentity
from experiments.cvs_phase1_stack.runtime import installed
from experiments.cvs_stage2_decouple.common import parameter_groups


def groups(model,c):
    # A margin module has no trainable heads; no empty optimizer group is added.
    pg,names=parameter_groups(model.base)
    for group in pg:group['lr']=c['optimizer']['peak_lr']
    names={key:['base.'+n for n in value] for key,value in names.items()}
    return [group for group in pg if group['params']],{key:value for key,value in names.items() if value}


def build(c,device,checkpoint=None):
    d.validate(c)
    provenance=student_initialization(c)
    # Static native args come from the registered reference architecture, never
    # from a parent checkpoint. This branch is valid with torch.load forbidden.
    args=d.baseline.make_args(c['reference_config'],str(device))
    args.num_domains=5;args.input_len=256
    args.from_scratch=True;args.baseline_ckpt='';args.teacher_ckpt=''
    torch.manual_seed(c['model_seed'])
    with installed(c['reference_config']) as native:
        base=native.build_baseline_model(args,torch.device(device))
    base._stage2_baseline_args=vars(args).copy()
    model=OriginalIdentity(base).to(device)
    if c['multilevel']['enabled']:
        if c['multilevel']['statistic']!='margin':raise ValueError('Scratch matrix has no learned probe-head arm')
        from .multilevel import BoundarySensitivity
        model.multilevel=BoundarySensitivity(base,seed=c['model_seed']).to(device)
    if checkpoint is not None:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        expected=dict(config=c,model_seed=c['model_seed'],epoch=20,total_epochs=20,parent_training_epochs=0,
            incremental_epochs=20,optimizer_steps=4440,parent_checkpoint=None,checkpoint_selection='final_only',
            target_access=False,run_id=c['run_id'],candidate_id=c['row_id'],parent_provenance=provenance,
            scratch_only=True,scratch_scope='student_identity_only',reference_E20_loaded=False)
        if any(payload.get(key)!=value for key,value in expected.items()):
            raise ValueError('Own scratch20 checkpoint identity/provenance differs')
        validate_teachers(c,payload.get('teacher_provenance',{}))
        if payload.get('checkpoint_sources')!=expected_sources(c):
            raise ValueError('Scratch checkpoint omitted independent teacher ancestry')
        model.load_state_dict(payload['model'],strict=True)
    model._parent_provenance=provenance
    return model
