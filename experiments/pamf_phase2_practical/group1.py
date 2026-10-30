"""Group1: full Tri-SIG-TIME SSL PAMF and native external midpoint-PL controls."""
import copy
import torch
from torch.nn import functional as F

RUN='20261030-phase2-group1-trisigtime-threepool-m4-r04'
SEEDS=(2026092701,2026092702,2026092703,2026092704)


def catalog():
    rows=[
        ('pamf_tri_sig_time_ssl_milkr','both',50,'pamf',None),
        ('protonet_cda','DA',200,'protonet',None),
        ('mrior_sda','DA',200,'mrior',None),
        ('dadda_sda','DA',200,'dadda',None),
        ('twostage','DA',200,'twostage',None),
        ('feature_separation_hu2024','DA',200,'feature_separation_hu2024',None),
        ('poster_ft','DA',200,'poster_ft',None),
        ('radionet_ada_knn','DA',200,'radionet_ada',None),
        ('tweak','DA',100,'tweak','native_M10_observation_tracks_unbound'),
        ('csil','CIL',200,'csil','author_MATLAB_Fisher_path_unbound'),
        ('mopc_hr','CIL',200,'mopc_hr','original_encoder_binding_unresolved_no_generic_encoder_substitution'),
        ('orthogonal','CIL',200,'orthogonal',None),
        ('issl','CIL',300,'issl_author_behavior',None),
        ('lora_triplet_knn','CIL',1000,'lora_triplet_knn',None),
    ]
    return [dict(id=i,category=c,epochs=e,PL_start_epoch=14 if m=='pamf' else e//2+1,method=m,
                 B='native' if c!='CIL' else None,C='native' if c!='DA' else None,
                 active=reason is None,unavailable_reason=reason,
                 native_scope='original_task_mechanism_with_declared_WiSig_input_and_source_PL_adapter')
            for i,c,e,m,reason in rows]


def row(name):
    if name=='pamf_pl_milkr':
        return dict(catalog()[0],id=name,PL_start_epoch=26,source_method='historical_ordinary_PL_not_current_PAMF')
    return next(r for r in catalog() if r['id']==name)


def workpoints(episode):
    if episode=='MIX01':
        return [1,5,10,20],[(k,n) for k in [1,5,10,20] for n in [2,5,10,20]]
    return [5],[(5,20)]


def pl_enabled(epoch,epochs):
    return epoch>epochs//2


def metric_probabilities(features,centers,temperature=.1):
    if temperature<=0 or features.ndim!=2 or centers.ndim!=2 or features.shape[1]!=centers.shape[1]:
        raise ValueError('Invalid metric confidence geometry')
    if not bool(torch.isfinite(features).all() and torch.isfinite(centers).all()):
        raise FloatingPointError('Nonfinite metric confidence features')
    if bool((features.norm(dim=1)==0).any() or (centers.norm(dim=1)==0).any()):
        raise FloatingPointError('Zero metric confidence feature/center')
    d=torch.cdist(F.normalize(features,dim=1),F.normalize(centers,dim=1)).square()
    return (-d/temperature).softmax(1)


def masked_ce(logits,pseudo,confidence):
    return (F.cross_entropy(logits,pseudo.detach(),reduction='none')*(confidence.detach()>=.95)).mean()


def validate_source_config(c):
    forbidden={'data_root','target_root','target_truth','truth','capsules','shared_dura_root',
               'target_inputs','query','target_support','views_root','raw_directory','legacy_directory'}
    if forbidden & set(c):raise ValueError('Source config contains target/query input handles')
    if c.get('source_contract') is not None and c.get('source_cache') is None:
        raise ValueError('Source cache is required')


def source_scope(rt):
    keys=('run_id','run_root','release_root','phase1_release','frozen_sources','source_cache','source_contract',
          'augmentation_cache','augmentation_cpu_workers','code_commit','source_base_config','python','source_cpu_threads',
          'PAMF_sources','PAMF_definition','external_source_root')
    result={k:rt[k] for k in keys if k in rt}
    validate_source_config(result)
    return result


def registration_anchor(bundle):
    """Construct a native C anchor directly at A; no target-old calibration."""
    from .vendor_native.comparison_suite.registration import RegistrationState
    model=copy.deepcopy(bundle['model']).eval()
    summaries=copy.deepcopy(bundle.get('base_metadata',{}))
    state=RegistrationState(bundle['method'],model,tuple(bundle['classes']),len(bundle['classes']),
        base_metadata=summaries,protocol_metadata=dict(phase='A_FROZEN_DIRECT_C',
            B_executed=False,query_fitting=False,source_base_metadata_fixed_before_B=True))
    if bundle['method']=='mopc_hr':state.prototypes=summaries['old_prototypes'].clone()
    return state
