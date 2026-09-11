"""Source-only paired feature caches; source row metadata never enters predictors."""
from dataclasses import dataclass,asdict,replace
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from .evidence_conditions import received_conditions

SCENES=('leo_clear_weak','leo_low_elev_weak','leo_rain_weak')


@dataclass(frozen=True)
class CacheIdentity:
    checkpoint_sha256:str
    data_contract_sha256:str
    preprocessing_version:str
    view_recipe:dict
    split_seed:int
    role:str

    def validate(self):
        if self.role not in {'L_s','V'}:raise ValueError('cache role must be L_s or V')
        for digest in (self.checkpoint_sha256,self.data_contract_sha256):
            if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('invalid cache identity digest')
        if not self.preprocessing_version or self.view_recipe.get('version')!='paired_leo_v1':raise ValueError('unknown cache preprocessing/view recipe')
        if set(self.view_recipe)!={'version','seed'} or not isinstance(self.view_recipe['seed'],int):raise ValueError('invalid view recipe')


@dataclass
class FeatureRows:
    h:torch.Tensor
    baseline_inference_logits:torch.Tensor
    quality:torch.Tensor
    observed:torch.Tensor
    labels:torch.Tensor
    physical_ids:tuple
    view_ids:tuple
    receiver:torch.Tensor
    day:torch.Tensor
    valid:torch.Tensor
    identity:CacheIdentity

    def __len__(self):return len(self.h)

    def validate(self):
        self.identity.validate();n=len(self)
        if self.h.ndim!=2 or n==0 or self.baseline_inference_logits.ndim!=2 or self.baseline_inference_logits.shape[0]!=n:
            raise ValueError('invalid cache feature/logit shape')
        if self.quality.shape!=(n,3) or self.observed.shape!=self.h.shape or self.observed.dtype!=torch.bool:
            raise ValueError('invalid quality/mask schema')
        for key in ('h','baseline_inference_logits','quality'):
            if not torch.isfinite(getattr(self,key)).all():raise ValueError('nonfinite cache '+key)
        for key in ('labels','receiver','day'):
            t=getattr(self,key)
            if t.shape!=(n,) or t.dtype!=torch.long:raise ValueError('invalid '+key)
        if self.valid.shape!=(n,) or self.valid.dtype!=torch.bool:raise ValueError('invalid valid mask')
        if (self.labels<0).any() or (self.labels>=self.baseline_inference_logits.shape[1]).any():raise ValueError('invalid class labels')
        if len(self.physical_ids)!=n or len(self.view_ids)!=n or any(not isinstance(x,str) or not x for x in self.physical_ids):
            raise ValueError('invalid physical keys')
        if len(set(zip(self.physical_ids,self.view_ids)))!=n:raise ValueError('duplicate physical/view key')
        groups={}
        for i,key in enumerate(self.physical_ids):
            metadata=(int(self.labels[i]),int(self.receiver[i]),int(self.day[i]))
            if key in groups and groups[key]!=metadata:raise ValueError('physical metadata disagrees across views')
            groups[key]=metadata
        if any(v not in ('clean',*SCENES) for v in self.view_ids):raise ValueError('unsupported source view')
        return self

    def take(self,indices):
        indices=torch.as_tensor(indices,dtype=torch.long).cpu()
        data={key:getattr(self,key)[indices.to(getattr(self,key).device)] for key in TENSOR_FIELDS}
        data.update(physical_ids=tuple(self.physical_ids[i] for i in indices.tolist()),view_ids=tuple(self.view_ids[i] for i in indices.tolist()),identity=self.identity)
        return FeatureRows(**data).validate()


TENSOR_FIELDS=('h','baseline_inference_logits','quality','observed','labels','receiver','day','valid')


def normal_cpu(t):
    # clone outside inference_mode, even when a caller supplied inference tensors.
    with torch.inference_mode(False):return t.detach().cpu().clone()


def save_source_cache(rows,path,shard_rows=4096):
    rows.validate();path=Path(path)
    if shard_rows<1:raise ValueError('positive shard size required')
    path.mkdir(parents=True,exist_ok=False)
    manifest={'schema':'anchored_source_cache_v1','identity':asdict(rows.identity),'rows':len(rows),'shards':[]}
    for start in range(0,len(rows),shard_rows):
        end=min(start+shard_rows,len(rows));name=f'rows_{start:08d}.pt'
        values={k:normal_cpu(getattr(rows,k)[start:end]) for k in TENSOR_FIELDS}
        values.update(physical_ids=list(rows.physical_ids[start:end]),view_ids=list(rows.view_ids[start:end]))
        torch.save(values,path/name);manifest['shards'].append(name)
    (path/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf-8')


def load_source_cache(path,expected_identity):
    path=Path(path);expected_identity.validate()
    manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema')!='anchored_source_cache_v1' or manifest.get('identity')!=asdict(expected_identity):
        raise ValueError('cache identity mismatch')
    shards=manifest.get('shards',[])
    if not shards or len(shards)!=len(set(shards)):raise ValueError('invalid cache shards')
    rows=[]
    for name in shards:
        if Path(name).name!=name or not name.endswith('.pt'):raise ValueError('invalid shard path')
        row=torch.load(path/name,map_location='cpu',weights_only=True)
        if set(row)!=set(TENSOR_FIELDS)|{'physical_ids','view_ids'}:raise ValueError('invalid cache row schema')
        rows.append(row)
    data={key:normal_cpu(torch.cat([s[key] for s in rows])) for key in TENSOR_FIELDS}
    data.update(physical_ids=tuple(k for s in rows for k in s['physical_ids']),view_ids=tuple(k for s in rows for k in s['view_ids']),identity=expected_identity)
    result=FeatureRows(**data).validate()
    if len(result)!=manifest['rows']:raise ValueError('cache row count mismatch')
    return result


class IdentityAnchor(nn.Module):
    def __init__(self,model):
        super().__init__()
        if getattr(model,'representation_mode','dual')!='dual' or not hasattr(model,'id_backbone'):
            raise ValueError('requires original dual H0 identity backbone')
        if hasattr(model,'evidence_head') or getattr(model,'use_crra',False) or getattr(model,'sat_anchor_identity_adapter',None) is not None or getattr(model,'identity_capacity',None) is not None:
            raise ValueError('unsupported modified H0 inference definition')
        head=model.id_backbone.cls_head.head
        if head.__class__.__name__!='CosFaceHead':raise ValueError('H0 must expose actual CosFace directions/scale')
        self.original_model=model.requires_grad_(False).eval()

    @property
    def w0(self):return self.original_model.id_backbone.cls_head.head.weight.detach()

    @property
    def tau0(self):return self.original_model.id_backbone.cls_head.head.s

    @property
    def class_count(self):return len(self.w0)

    @torch.no_grad()
    def extract_aux(self,x):
        from model_dual_cvsincnet import backbone_forward_compat
        self.original_model.eval()
        x=x.to(next(self.original_model.parameters()).device)
        aux=backbone_forward_compat(self.original_model.id_backbone,x,y=None,return_aux=True,
                                   domain_labels=None,update_crra_support=False,update_nmfdu_support=False)
        return aux

    @torch.no_grad()
    def extract(self,x):
        aux=self.extract_aux(x)
        h=aux['feat_joint'];scores=aux['logits']
        conditions=received_conditions(x.to(h.device))
        if not torch.isfinite(h).all() or not torch.isfinite(scores).all():raise ValueError('nonfinite H0 inference')
        return h.detach(),scores.detach(),conditions['valid'].detach()


def paired_scene_schedule(physical_ids,labels,receiver,day,seed):
    if not (len(physical_ids)==len(labels)==len(receiver)==len(day)) or len(set(physical_ids))!=len(physical_ids):
        raise ValueError('unique physical records required for scene assignment')
    groups={}
    for pid,y,rx,d in zip(physical_ids,labels,receiver,day):groups.setdefault((int(y),int(rx),int(d)),[]).append(pid)
    result={}
    for rotation,group in enumerate(sorted(groups)):
        ordered=sorted(groups[group],key=lambda pid:hashlib.sha256(f'{seed}:{pid}'.encode()).digest())
        for i,pid in enumerate(ordered):result[pid]=SCENES[(i+rotation)%len(SCENES)]
    return result


def build_source_cache(anchor,loader,identity,view_recipe,output,*,contract,shard_rows=4096,batch_size=128,layout='joint'):
    """Loader yields source dictionaries, not arbitrary target datasets."""
    identity.validate()
    if layout not in {'joint','blocks'} or batch_size<1:raise ValueError('invalid feature layout/batch size')
    expected_preprocessing='received_iq_v1'+(':blocks_t_f_pa_fixed_v1' if layout=='blocks' else '')
    if identity.preprocessing_version!=expected_preprocessing:raise ValueError('feature coordinates must be bound in cache identity')
    if view_recipe!=identity.view_recipe:raise ValueError('view identity mismatch')
    raw=[]
    allowed={'x','y','physical_ids','receiver','day'}
    for batch in loader:
        if set(batch)!=allowed:raise ValueError('source loader whitelist violation')
        n=len(batch['x'])
        if any(len(batch[k])!=n for k in allowed if k!='x'):raise ValueError('source batch length mismatch')
        for i in range(n):raw.append((batch['x'][i],int(batch['y'][i]),str(batch['physical_ids'][i]),int(batch['receiver'][i]),int(batch['day'][i])))
    ids=[r[2] for r in raw]
    if len(set(ids))!=len(ids) or set(ids)!=set(contract.get('roles',{}).get(identity.role,[])):
        raise ValueError('source physical IDs differ from contract role')
    other=set(pid for role,values in contract['roles'].items() if role!=identity.role for pid in values)
    if set(ids)&other:raise ValueError('source role physical overlap')
    if any(r[3] not in contract['source_receivers'] or r[4] not in contract['source_days'] for r in raw):
        raise ValueError('source receiver/day violates contract')
    schedule=paired_scene_schedule(ids,[r[1] for r in raw],[r[3] for r in raw],[r[4] for r in raw],view_recipe['seed'])
    from .evidence_target_evaluation import _scene_iq
    data={k:[] for k in TENSOR_FIELDS};pids=[];views=[]
    pending=[];block_sizes=None
    def flush():
        nonlocal block_sizes
        received=torch.cat([r[0] for r in pending])
        if layout=='joint':h,s0,valid=anchor.extract(received)
        else:
            from .evidence_observation import extract_evidence
            aux=anchor.extract_aux(received);obs=extract_evidence(aux,'blocks')
            h=obs.z;s0=aux['logits'];valid=received_conditions(received)['valid'];block_sizes=obs.block_sizes
        q=received_conditions(received)['quality']
        for key,value in dict(h=h,baseline_inference_logits=s0,quality=q,observed=torch.ones_like(h,dtype=torch.bool),
                              labels=torch.tensor([r[1] for r in pending]),receiver=torch.tensor([r[3] for r in pending]),
                              day=torch.tensor([r[4] for r in pending]),valid=valid).items():data[key].append(normal_cpu(value))
        pids.extend(r[2] for r in pending);views.extend(r[5] for r in pending);pending.clear()
    for x,y,pid,rx,day in raw:
        x=normal_cpu(x)[None]
        opaque=torch.tensor([list(hashlib.sha256(pid.encode()).digest())],dtype=torch.uint8)
        for scene in ('clean',schedule[pid]):
            received=_scene_iq(x,opaque,scene,view_recipe['seed'])
            pending.append((received,y,pid,rx,day,scene))
            if len(pending)==batch_size:flush()
    if pending:flush()
    result=FeatureRows(**{k:torch.cat(v) for k,v in data.items()},physical_ids=tuple(pids),view_ids=tuple(views),identity=identity).validate()
    save_source_cache(result,output,shard_rows)
    if block_sizes is not None:
        (Path(output)/'block_schema.json').write_text(json.dumps(dict(names=['t','f','pa'],sizes=list(block_sizes)))+'\n',encoding='utf-8')
    return result
