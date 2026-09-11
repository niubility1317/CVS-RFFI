"""Explicit source stages, never an automatic target evaluation chain."""
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace
import hashlib,io,json,time
import torch
from .anchored_cache import CacheIdentity,IdentityAnchor,build_source_cache,load_source_cache
from .anchored_fit import FitConfig,ExpertArtifact,fit_expert,write_csv,paired_weights
from .anchored_geometry import OrdinaryAngleHead
from .anchored_pipeline import FrozenAnchoredSystem,state_identity,validate_config
from .anchored_crossfit import OOFArtifact,RXFold,run_expert_oof,run_nested_fusion_audit,validate_oof,outer_fixed_actions
from .anchored_fusion import realize_actions,action_utilities,protection_quantile
from .anchored_calibration import FinalProbabilityCalibrator,physical_weights
from .anchored_reporting import save_json,fusion_reports,risk_coverage,profile_stages,partial_pair_diagnostics,nested_reports,assess_source_promotion


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def load_ground(ground_path,contract_path,device):
    from .evidence_pipeline import verify_checkpoint_contract
    from post_stage_common import build_baseline_model
    contract=json.loads(Path(contract_path).read_text(encoding='utf-8'))
    payload=torch.load(ground_path,map_location='cpu',weights_only=True)
    verify_checkpoint_contract(payload,contract)
    model=build_baseline_model(SimpleNamespace(**payload['baseline_args']),torch.device(device))
    model.load_state_dict(payload['model'],strict=True)
    return IdentityAnchor(model),payload,contract


def cache_identity(config,ground,contract,role,layout):
    return CacheIdentity(file_digest(ground),file_digest(contract),config['cache']['preprocessing_version']+(':blocks_t_f_pa_fixed_v1' if layout=='blocks' else ''),
                         {'version':'paired_leo_v1','seed':config['view_seed']},config['split_seed'],role)


def source_loader(path,contract,role,batch_size):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if set(data)!={'x','y','physical_ids'} or set(data['physical_ids'])!=set(contract['roles'][role]):raise ValueError('source input differs from role contract')
    parsed=[list(map(int,p.split(':'))) for p in data['physical_ids']]
    if any(len(p)!=4 or p[0]!=int(y) for p,y in zip(parsed,data['y'])):raise ValueError('physical class mapping mismatch')
    if len(data['x'])!=len(parsed) or data['y'].shape!=(len(parsed),):raise ValueError('source input length mismatch')
    for start in range(0,len(parsed),batch_size):
        end=min(start+batch_size,len(parsed))
        yield dict(x=data['x'][start:end],y=data['y'][start:end],physical_ids=data['physical_ids'][start:end],
                   receiver=torch.tensor([p[1] for p in parsed[start:end]]),day=torch.tensor([p[2] for p in parsed[start:end]]))


def load_oof(path,rows,config,seed,w0,tau0):
    p=Path(path);value=torch.load(p/'oof_predictions.pt',map_location='cpu',weights_only=True)
    folds=[RXFold(v['heldout_rx'],tuple(v['train_rx']),tuple(v['train_physical_ids']),tuple(v['predict_physical_ids'])) for v in value['folds']]
    experts={f.train_rx:ExpertArtifact.load(p/f'rx_{f.heldout_rx}'/'expert.pt') for f in folds}
    final=ExpertArtifact.load(p/'all_source'/'expert.pt');experts[tuple(final.train_rx)]=final
    oof=OOFArtifact(value['scores'],folds,experts,final,value['cache_identity'],value['binding'])
    validate_oof(rows,oof,config,seed,w0,tau0);return oof


def _save_system(system,path):
    with Path(path).open('xb') as f:torch.save(system.export_state(),f)


def run_stage(args,config):
    config=validate_config(config);stage=args.stage;output=Path(args.output)
    if output.exists():raise FileExistsError('stage output already exists; preserve it and use a new stage directory')
    if args.candidate not in config['candidates'] or args.seed not in config['head_seeds']:raise ValueError('candidate/seed outside frozen matrix')
    if stage not in config['stages']:raise ValueError('only explicit source stages are supported')
    anchor,payload,contract=load_ground(args.ground,args.contract,args.device)
    if contract['source_receivers']!=config['source_receivers'] or contract['source_days']!=config['source_days']:raise ValueError('source RX/day mismatch')
    if contract.get('split_seed')!=config['split_seed']:raise ValueError('source split seed mismatch')
    layout='blocks' if args.candidate=='P1' else 'joint'
    role=args.role if stage=='cache' else 'V' if stage=='calibrate' else 'L_s'
    identity=cache_identity(config,args.ground,args.contract,role,layout)
    if stage in {'calibrate','export','profile'}:_verify_stage_input(args,config,identity)
    start=time.perf_counter()
    if stage=='cache':
        rows=build_source_cache(anchor,source_loader(args.source,contract,role,config['cache']['batch_size']),identity,identity.view_recipe,output,
                                contract=contract,shard_rows=config['cache']['shard_rows'],batch_size=config['cache']['batch_size'],layout=layout)
        save_json(output/'stage_report.json',dict(stage=stage,role=role,rows=len(rows),physical_count=len(set(rows.physical_ids)),seconds=time.perf_counter()-start))
        _stage_report(output,args,config,'SOURCE_CACHE_COMPLETE')
        return
    rows=None
    if stage in {'fit','oof','fuse','calibrate'}:
        rows=load_source_cache(args.cache,identity)
        if set(rows.physical_ids)!=set(contract['roles'][role]):raise ValueError('cache physical IDs differ from expected role')
    cfg=replace(FitConfig.parse(config['fit']),candidate=args.candidate if args.candidate in {'A2','A3','A4','C_angle','C_angle_keep'} else 'A4')
    kwargs=dict(w0=anchor.w0.cpu(),tau0=anchor.tau0,device=args.device)
    architecture=payload['baseline_args']
    if stage=='oof':
        if args.candidate not in {'A2','A3','A4','C_angle','C_angle_keep'}:raise ValueError('OOF stage requires trainable expert')
        run_expert_oof(rows,cfg,args.seed,output,**kwargs)
    elif stage=='fit':
        if args.candidate=='P1':
            from .partial_evidence_fit import fit_partial_evidence
            from .mask_pattern_calibration import PatternSpec
            schema=json.loads((Path(args.cache)/'block_schema.json').read_text(encoding='utf-8'))
            p=config['partial'];head=fit_partial_evidence(rows,PatternSpec(tuple(schema['names']),tuple(schema['sizes'])),rank=p['rank'],shrinkage=p['shrinkage'],
                                                       observation_floor=p['variance_floor'],observation_ceiling=p['variance_ceiling'],output=output)
            write_csv(output/'pair_information.csv',partial_pair_diagnostics(head))
        else:
            if args.candidate in {'A0','A1'}:
                output.mkdir(parents=True)
                w=anchor.w0.cpu()
                if args.candidate=='A1':
                    clean=torch.tensor([v=='clean' for v in rows.view_ids])
                    w=torch.stack([rows.h[clean&(rows.labels==y)].mean(0) for y in range(anchor.class_count)])
                state=OrdinaryAngleHead(w,anchor.tau0).export_state()
            elif args.candidate in {'A2','A3','A4','C_angle','C_angle_keep'}:
                state=fit_expert(rows,config['source_receivers'],cfg,args.seed,output,**kwargs).state
            else:raise ValueError('A5/A6 require fuse stage after A4 OOF')
            system=FrozenAnchoredSystem(anchor,state,fixed_action=0 if args.candidate=='A0' else 4,protection_threshold=1.,architecture=architecture,class_labels=contract.get('tx_mapping'),mode='mixture' if args.candidate=='A0' else 'expert')
            _save_system(system,output/'system_state.pt')
    elif stage=='fuse':
        if args.candidate not in {'A5-0','A5-25','A5-50','A5-75','A5-100','A6'}:raise ValueError('fuse requires A5/A6')
        oof=load_oof(args.input,rows,cfg,args.seed,anchor.w0.cpu(),anchor.tau0)
        f=config['fusion'];threshold=protection_quantile(rows.baseline_inference_logits,f['protection_q'])
        actions,guard_folds=outer_fixed_actions(rows,oof,f['protection_q'])
        output.mkdir(parents=True)
        save_json(output/'fixed_guard_folds.json',dict(folds=guard_folds,final_protection_threshold=threshold))
        torch.save(dict(schema='source_actual_actions_v1',physical_ids=list(rows.physical_ids),view_ids=list(rows.view_ids),
                        s0=rows.baseline_inference_logits,sG=oof.scores,actions=actions,folds=guard_folds),output/'oof_predictions.pt')
        records=fusion_reports(rows,actions,output,lambda_h=f['lambda_h'])
        if args.candidate=='A6':
            trigger=any(r['weighted_net']>0 and r['action']>0 for r in records if r['group']=='all')
            if not trigger:
                save_json(output/'activation.json',dict(status='NOT_ACTIVATED_NO_SOURCE_UTILITY'))
                _stage_report(output,args,config,'NOT_ACTIVATED_NO_SOURCE_UTILITY');return
            nested=run_nested_fusion_audit(rows,oof,cfg,args.seed,output/'nested',**kwargs,**f)
            nested_metrics=nested_reports(rows,nested,output)
            save_json(output/'source_promotion.json',assess_source_promotion({args.seed:nested_metrics}))
            system=FrozenAnchoredSystem(anchor,oof.final_expert.state,gate_state=nested['final_gate_state'],
                                        protection_threshold=nested['final_protection_threshold'],architecture=architecture,class_labels=contract.get('tx_mapping'))
            save_json(output/'activation.json',dict(status='SOURCE_NESTED_EVALUATED',promotion='requires all RX/TX/clean and three-seed assessment; not automatic'))
        else:
            action={'A5-0':0,'A5-25':1,'A5-50':2,'A5-75':3,'A5-100':4}[args.candidate]
            system=FrozenAnchoredSystem(anchor,oof.final_expert.state,fixed_action=action,protection_threshold=threshold,architecture=architecture,class_labels=contract.get('tx_mapping'))
        _save_system(system,output/'system_state.pt')
    elif stage=='calibrate':
        if args.candidate=='P1':
            _calibrate_partial(args,config,rows,anchor,payload,output)
        else:
            system=FrozenAnchoredSystem.from_state(torch.load(args.input,map_location='cpu',weights_only=True))
            if system.calibrator is not None:raise ValueError('already calibrated input')
            _verify_anchor(system,anchor)
            result=system.predict_features(rows.h,rows.baseline_inference_logits,rows.quality,rows.valid)
            cal=FinalProbabilityCalibrator(**config['calibration']).fit(result['log_probabilities'],rows.labels,weights=physical_weights(rows.physical_ids),role='V',system_identity=system.identity)
            system.attach_calibrator(cal);output.mkdir(parents=True);_save_system(system,output/'system_state.pt')
            save_json(output/'calibration_report.json',dict(**cal.state_dict(),role='V',internal_temperatures_frozen=True,argmax_unchanged=True,physical_weighting=True))
            predicted=system.predict_features(rows.h,rows.baseline_inference_logits,rows.quality,rows.valid)
            write_csv(output/'risk_coverage.csv',risk_coverage(predicted['confidence'],predicted['top_class']==rows.labels,physical_weights(rows.physical_ids)))
            w=physical_weights(rows.physical_ids);accepted=predicted['accepted'];wrong=predicted['top_class']!=rows.labels
            coverage=float(w[accepted].sum())
            write_csv(output/'selective_policy.csv',[dict(policy='frozen_V_threshold_and_validity',coverage=coverage,
                risk=float(w[accepted&wrong].sum())/coverage if coverage else None,closed_accuracy=float(w[~wrong].sum()),
                correct_rejected=int((~wrong&~accepted).sum()),accepted_rows=int(accepted.sum()))])
    elif stage=='export':
        state=torch.load(args.input,map_location='cpu',weights_only=True)
        if args.candidate=='P1':
            system=FrozenPartialSystem.from_state(state);_verify_anchor(system,anchor)
        else:system=FrozenAnchoredSystem.from_state(state);_verify_anchor(system,anchor)
        output.mkdir(parents=True);system.save(output/'frozen_system.pt')
    elif stage=='profile':
        state=torch.load(args.input,map_location='cpu',weights_only=True)
        system=FrozenPartialSystem.from_state(state) if args.candidate=='P1' else FrozenAnchoredSystem.from_state(state)
        _verify_anchor(system,anchor)
        batch=next(source_loader(args.source,contract,'L_s',config['cache']['batch_size']));x=batch['x']
        from .evidence_conditions import received_conditions
        from .evidence_target_evaluation import _scene_iq
        opaque=torch.tensor([list(hashlib.sha256(p.encode()).digest()) for p in batch['physical_ids']],dtype=torch.uint8)
        h,s0,valid=anchor.extract(x);quality=received_conditions(x)['quality']
        functions={'IQ_augmentation':lambda:_scene_iq(x,opaque,'leo_clear_weak',config['view_seed']),
                   'identity_backbone':lambda:anchor.extract(x),'save':lambda:torch.save(system.export_state(),io.BytesIO())}
        if args.candidate!='P1':
            sg=system.head(h.cpu()).detach()
            functions.update(head=lambda:system.head(h.cpu()),fusion_calibration=lambda:system.predict_scores(s0,sg,quality,valid))
        else:functions['partial_full_pipeline']=lambda:system.predict(x,'full')
        result=profile_stages(functions,**config['profile'],device=args.device)
        result['stage_devices']={name:(args.device if name=='identity_backbone' else 'cpu') for name in functions}
        output.mkdir(parents=True);save_json(output/'runtime_profile.json',result)
    save_json(output/'protocol_manifest.json',dict(stage=stage,status='SOURCE_SYSTEM_FROZEN' if stage=='export' else 'SOURCE_STAGE_COMPLETE',
              candidate=args.candidate,head_seed=args.seed,config=config,checkpoint_lineage=payload['checkpoint_lineage'],cache_identity=asdict(identity),
              contract_sha256=identity.data_contract_sha256,source_only=True,backbone_updated=False,support_used=False,
              prior_target_observation='Design was informed by previously observed target errors; no independent confirmation claimed.',
              elapsed_seconds=time.perf_counter()-start))
    _stage_report(output,args,config,'SOURCE_SYSTEM_FROZEN' if stage=='export' else 'SOURCE_STAGE_COMPLETE')


def _stage_report(output,args,config,status):
    lines=['# CORE90锚定几何source阶段报告','',f'状态：`{status}`。候选：`{args.candidate}`；head seed：`{args.seed}`；阶段：`{args.stage}`。','',
           '本目录只证明当前阶段产物，不表示全部source矩阵或独立确认已完成。H0骨干见过所有source RX，内部交叉拟合仅隔离头部。设计受此前target结果启发；入口不访问target。','',
           '|候选|本目录记录|','|---|---|']
    for candidate in config['candidates']:lines.append(f'|{candidate}|'+(status if candidate==args.candidate else 'NOT_RECORDED_IN_THIS_STAGE')+'|')
    lines+=['','固定预算：每专家80轮；完整source每轮35步；三head seeds。基础专家矩阵共90次拟合，A6按source触发后最多增加30次，共120次上限。该预算不包含已训练H0或P1独立成本。','',
            'source晋级需完整三个seed的嵌套分组报告；单目录或一次seed结果不产生科学晋级。20/40/80检查点仅用于source诊断，不按target选权重。','',
            '## 产物','']
    lines.extend(f'- [{p.name}]({p.name})' for p in sorted(Path(output).iterdir()) if p.is_file() and p.name!='report.md')
    with (Path(output)/'report.md').open('x',encoding='utf-8',newline='\n') as f:f.write('\n'.join(lines)+'\n')


def _verify_anchor(system,anchor):
    if system.anchor is None or state_identity(system.anchor.original_model.id_backbone.state_dict())!=state_identity(anchor.original_model.id_backbone.state_dict()):
        raise ValueError('frozen bundle H0 differs from verified checkpoint')


def _verify_stage_input(args,config,identity):
    """Use the existing stage manifest; never relabel an input as another arm."""
    manifest=json.loads((Path(args.input).parent/'protocol_manifest.json').read_text(encoding='utf-8'))
    allowed={'calibrate':{'fit','fuse'},'export':{'calibrate'},'profile':{'calibrate','export'}}
    if manifest.get('stage') not in allowed[args.stage] or manifest.get('candidate')!=args.candidate or manifest.get('head_seed')!=args.seed:
        raise ValueError('input stage/candidate/head seed mismatch')
    old=manifest.get('cache_identity',{})
    expected=asdict(identity)
    if any(old.get(k)!=v for k,v in expected.items() if k!='role') or old.get('role') not in {'L_s','V'}:
        raise ValueError('input checkpoint/data/feature identity mismatch')
    def behavior(c):
        return {k:v for k,v in c.items() if k not in {'profile','cache'}}|{'preprocessing_version':c.get('cache',{}).get('preprocessing_version')}
    if behavior(manifest.get('config',{}))!=behavior(config):raise ValueError('input method configuration mismatch')
    return manifest


def _calibrate_partial(args,config,rows,anchor,payload,output):
    from .partial_evidence_fit import PartialEvidenceHead
    from .mask_pattern_calibration import PatternCalibrator
    head=PartialEvidenceHead.from_export(torch.load(args.input,map_location='cpu',weights_only=True))
    fit_manifest=json.loads((Path(args.input).parent/'fit_manifest.json').read_text(encoding='utf-8'))
    if fit_manifest.get('role')!='L_s' or fit_manifest.get('cache_identity')!=asdict(replace(rows.identity,role='L_s')):
        raise ValueError('P1 fit checkpoint/data/feature identity mismatch')
    schema=json.loads((Path(args.cache)/'block_schema.json').read_text(encoding='utf-8'))
    if tuple(schema['names'])!=head.pattern_spec.block_names or tuple(schema['sizes'])!=head.pattern_spec.block_sizes or tuple(fit_manifest.get('block_sizes',()))!=head.pattern_spec.block_sizes:
        raise ValueError('P1 fit/calibration block schema mismatch')
    values=[];patterns=[];ids=[];quality=[];labels=[]
    for pattern in head.pattern_spec.patterns:
        result=head(rows.h,head.pattern_spec.mask(pattern,len(rows)),rows.quality)
        values.append(result);patterns.extend([pattern]*len(rows));ids.extend(rows.physical_ids);quality.append(rows.quality);labels.append(rows.labels)
    scored={key:torch.cat([v[key] for v in values]) for key in ('scores','mahalanobis','observed_count')}
    system=FrozenPartialSystem(anchor,head,architecture=payload['baseline_args'],class_labels=payload['training_data_contract'].get('tx_mapping'))
    p=config['partial'];cal=PatternCalibrator(**{k:p[k] for k in ('quality_bins','min_group_samples','consistency_quantile','min_confidence')})
    cal.fit(scored['scores'],scored['mahalanobis'],scored['observed_count'],torch.cat(quality),torch.cat(labels),patterns,ids,system_identity=system.identity)
    system.calibrator=cal;output.mkdir(parents=True);torch.save(system.export_state(),output/'system_state.pt')
    decision=cal.predict(scored['scores'],scored['mahalanobis'],scored['observed_count'],torch.cat(quality),patterns)
    report=[];curves=[];groups=[];qbin=(rows.quality.double().mean(-1).clamp(0,1)*cal.quality_bins).long().clamp_max(cal.quality_bins-1)
    for j,pattern in enumerate(head.pattern_spec.patterns):
        pattern_ix=torch.arange(len(rows))+j*len(rows)
        curves.extend(dict(pattern=pattern,policy='confidence_ranking_only',**r) for r in risk_coverage(decision['confidence'][pattern_ix],decision['top_class'][pattern_ix]==rows.labels,physical_weights(rows.physical_ids)))
        for group,tensor in (('RX',rows.receiver),('TX',rows.labels),('day',rows.day)):
            for value in tensor.unique(sorted=True):
                local=torch.where(tensor==value)[0];ix=local+j*len(rows);w=physical_weights([rows.physical_ids[i] for i in local.tolist()])
                accepted=decision['accepted'][ix];wrong=decision['top_class'][ix]!=rows.labels[local];coverage=float(w[accepted].sum())
                groups.append(dict(pattern=pattern,group=group,value=int(value),role='source_V_calibration_descriptive',rows=len(local),
                                   physical_count=len({rows.physical_ids[i] for i in local.tolist()}),closed_accuracy=float(w[~wrong].sum()),coverage=coverage,
                                   selective_risk=float(w[accepted&wrong].sum())/coverage if coverage else None,correct_rejected=int((~wrong&~accepted).sum())))
        for quality_bin in range(cal.quality_bins):
            local=torch.where(qbin==quality_bin)[0]
            if not len(local):continue
            ix=local+j*len(rows);accepted=decision['accepted'][ix];correct=decision['top_class'][ix]==rows.labels[local]
            w=physical_weights([rows.physical_ids[i] for i in local.tolist()])
            report.append(dict(pattern=pattern,quality_bin=quality_bin,support_level='source_only',physical_count=len({rows.physical_ids[i] for i in local.tolist()}),view_count=len(local),
                               available=int(decision['calibration_available'][ix].sum()),accepted=int(accepted.sum()),coverage=float(w[accepted].sum()),
                               correctly_classified_rejected=int((correct&~accepted).sum()),accepted_errors=int((~correct&accepted).sum()),
                               selective_risk=float(w[accepted&~correct].sum()/w[accepted].sum()) if accepted.any() else None,
                               defer=sum(decision['status'][i]=='defer' for i in ix.tolist()),
                               model_mismatch_candidates=sum(decision['status'][i]=='model_mismatch_candidate' for i in ix.tolist())))
    write_csv(output/'mask_calibration.csv',report);write_csv(output/'mask_source_groups.csv',groups)
    write_csv(output/'mask_risk_coverage.csv',curves);save_json(output/'calibration_report.json',cal.state_dict())


class FrozenPartialSystem:
    def __init__(self,anchor,head,*,architecture,calibrator=None,class_labels=None):
        from .anchored_pipeline import ARCH_KEYS
        self.anchor=anchor;self.head=head;self.architecture={k:v for k,v in architecture.items() if k in ARCH_KEYS};self.calibrator=calibrator
        self.class_labels=tuple(class_labels or range(len(head.means)))
        if len(self.class_labels)!=len(head.means) or len(set(self.class_labels))!=len(self.class_labels):raise ValueError('invalid partial class mapping')
        self.identity=state_identity(self._core())
        self._frozen_versions=self._versions()

    def _versions(self):
        tensors=list(self.head.parameters())+list(self.head.buffers())+list(self.anchor.original_model.id_backbone.parameters())+list(self.anchor.original_model.id_backbone.buffers())
        return (tuple((id(t),t._version) for t in tensors),self.head.error.floor,self.head.error.ceiling,self.class_labels)

    def _core(self):
        return dict(schema='anchored_partial_system_v1',stage='Phase1',architecture=self.architecture,class_labels=list(self.class_labels),
                    identity_backbone={k:v.cpu().clone() for k,v in self.anchor.original_model.id_backbone.state_dict().items()},partial=self.head.export_state())

    @torch.no_grad()
    def predict(self,x,pattern):
        if self._versions()!=self._frozen_versions:raise ValueError('partial parameters changed after freezing')
        from .evidence_observation import extract_evidence
        from .evidence_conditions import received_conditions
        obs=extract_evidence(self.anchor.extract_aux(x),'blocks');q=received_conditions(x)['quality'];mask=self.head.pattern_spec.mask(pattern,len(x))
        result=self.head(obs.z,mask,q)
        if self.calibrator is None:raise RuntimeError('P1 requires frozen V pattern calibration')
        decision=self.calibrator.predict(result['scores'],result['mahalanobis'],result['observed_count'],q,[pattern]*len(x))
        decision['predicted_labels']=[self.class_labels[i] for i in decision['top_class'].tolist()]
        return decision

    def export_state(self):
        core=self._core()
        if state_identity(core)!=self.identity:raise ValueError('partial system changed after freezing')
        if self.calibrator is not None and self.calibrator.final.system_identity!=self.identity:raise ValueError('partial calibration identity mismatch')
        return dict(**core,system_identity=self.identity,calibrator=None if self.calibrator is None else self.calibrator.state_dict())

    def save(self,path):
        if self.calibrator is None:raise RuntimeError('P1 V calibration required')
        with Path(path).open('xb') as f:torch.save(self.export_state(),f)

    @classmethod
    def from_state(cls,state):
        from post_stage_common import build_baseline_model
        from .partial_evidence_fit import PartialEvidenceHead
        from .mask_pattern_calibration import PatternCalibrator
        from .anchored_pipeline import ARCH_KEYS
        if set(state)!={'schema','stage','architecture','identity_backbone','partial','system_identity','calibrator','class_labels'} or state['schema']!='anchored_partial_system_v1' or state['stage']!='Phase1':raise ValueError('invalid partial deployment schema')
        if set(state['architecture'])-set(ARCH_KEYS):raise ValueError('invalid architecture')
        model=build_baseline_model(SimpleNamespace(**state['architecture']),torch.device('cpu'));model.id_backbone.load_state_dict(state['identity_backbone'],strict=True)
        obj=cls(IdentityAnchor(model),PartialEvidenceHead.from_export(state['partial']),architecture=state['architecture'],
                calibrator=None if state['calibrator'] is None else PatternCalibrator.from_state_dict(state['calibrator']),class_labels=state['class_labels'])
        if obj.identity!=state['system_identity']:raise ValueError('partial identity mismatch')
        obj.export_state();return obj
