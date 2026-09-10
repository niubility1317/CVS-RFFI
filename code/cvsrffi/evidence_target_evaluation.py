"""Phase1 frozen R_t prediction sealing and separate truth-connected scoring.

This is not Phase2 and issues no VALIDATED_ONCE/data-capsule claim. Callers must
validate R_t rows and finish all candidate seals before calling the scorer.
"""
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import Dataset, DataLoader
from .eval import apply_sat_channel_for_scenario
from .evidence_decision import selective_metrics

SCENES = ('clean','leo_clear_weak','leo_low_elev_weak','leo_rain_weak')


def opaque_record_id(record):
    """Stable opaque physical ID, shared with the independent truth builder."""
    fields = ('tx_i','rx_i','day_i','sig_i')
    values = [int(record[key] if isinstance(record,dict) else getattr(record,key)) for key in fields]
    return hashlib.sha256((':'.join(map(str,values))).encode('ascii')).hexdigest()


class _IQOnlyTarget(Dataset):
    """Adapter removes labels, roles, receivers, and metadata before prediction."""
    def __init__(self,dataset):
        self._dataset = dataset
        declared = getattr(dataset,'role','R_t')
        if declared != 'R_t':
            raise ValueError('requires caller-validated R_t dataset')
        current = dataset
        while current is not None:
            if getattr(current,'transform',None) is not None or getattr(current,'crop_mode','center') == 'random':
                raise ValueError('target dataset must use deterministic IQ access without transforms')
            current = getattr(current,'base',None)
        if not hasattr(dataset,'index') or len(dataset.index)!=len(dataset):
            raise ValueError('target dataset must expose matching physical index')
        self._ids = [opaque_record_id(record) for record in dataset.index]
        if not self._ids or len(set(self._ids)) != len(self._ids):
            raise ValueError('target physical IDs must be nonempty and unique')
    def __len__(self):
        return len(self._ids)
    def __getitem__(self,index):
        # The underlying dataset packages truth; this boundary discards it.
        item = self._dataset[index]
        x = item if torch.is_tensor(item) else item[0]
        if x.ndim!=2 or x.shape[0]!=2 or not torch.isfinite(x).all():
            raise ValueError('target IQ must be finite [2,T]')
        return x.detach().cpu(), torch.tensor(list(bytes.fromhex(self._ids[index])),dtype=torch.uint8)


def _scene_iq(x,ids,scene,seed):
    if scene=='clean':
        return x
    transformed = []
    for row,identity in zip(x,ids):
        opaque = bytes(identity.tolist()).hex()
        digest = hashlib.sha256(f'{int(seed)}:{scene}:{opaque}'.encode('ascii')).digest()
        row_seed = int.from_bytes(digest[:8],'big') % (2**63-1)
        generator = torch.Generator(device='cpu').manual_seed(row_seed)
        result,_ = apply_sat_channel_for_scenario(row[None],scene,SimpleNamespace(),gen=generator)
        transformed.append(result[0])
    return torch.stack(transformed)


def seal_target(model,target_dataset,device,output_dir,seed=392005,batch_size=64,calibrator=None):
    """Seal four scenes without truth, fitting, candidate seeds or covariance retention."""
    if batch_size<1:
        raise ValueError('batch_size must be positive')
    folder = Path(output_dir)
    if folder.exists():
        raise FileExistsError('prediction output already exists')
    iq_only = _IQOnlyTarget(target_dataset)
    folder.mkdir(parents=True)
    if calibrator is not None and not calibrator.fitted:
        raise ValueError('target accepts only already frozen source calibration')
    calibration_before = None if calibrator is None else calibrator.state_dict()
    modes = [(module,module.training) for module in model.modules()]
    versions = {key:value._version for key,value in model.state_dict().items()}
    model.eval()
    manifest = dict(schema='phase1_evidence_target_seal_v1',status='INCOMPLETE',
                    rows=len(iq_only),seed=int(seed),scenes={},truth_access=False,
                    phase='Phase1 frozen R_t',temperature=1. if calibrator is None else calibrator.temperature,
                    channel_fs_hz=25e6,channel_fc_hz=2.462e9,
                    channel_rng='sha256(seed,scene,opaque physical ID); CPU per-row transform',
                    scientific_promotion_claim=False)
    try:
        with torch.no_grad():
            for scene in SCENES:
                started = time.perf_counter()
                rows = {}
                for x,ids in DataLoader(iq_only,batch_size=batch_size,shuffle=False,num_workers=0):
                    x = _scene_iq(x,ids,scene,seed).to(device)
                    result = model(x,y_tx=None,grl_lambda=0.,return_aux=True)
                    logits = result['tx_logits']
                    evidence = result.get('evidence')
                    accepted = torch.ones(len(x),device=logits.device,dtype=torch.bool)
                    selected = dict(ids=ids,scores=logits,temperature=logits.new_full((len(x),),manifest['temperature']))
                    if evidence is not None:
                        for key in ('mahalanobis','observed_count'):
                            selected[key] = evidence[key]
                        # No calibration support => defer, never invent confidence.
                        accepted = torch.zeros_like(accepted)
                        if calibrator is not None:
                            decision = calibrator.predict(logits,evidence['mahalanobis'],evidence['observed_count'],
                                evidence['quality'],state_coverage=evidence['state_in_domain'])
                            accepted = decision['accepted']
                            selected['calibration_available'] = decision['calibration_available']
                    selected['accepted'] = accepted
                    for key,value in selected.items():
                        rows.setdefault(key,[]).append(value.detach().cpu())
                    del result,evidence
                artifact = {key:torch.cat(value) for key,value in rows.items()}
                path = folder/(scene+'.pt')
                torch.save(artifact,path)
                manifest['scenes'][scene] = dict(file=path.name,rows=len(artifact['ids']),seconds=time.perf_counter()-started)
        if any(value._version!=versions[key] for key,value in model.state_dict().items()):
            raise RuntimeError('target prediction mutated frozen model')
        if calibrator is not None and calibrator.state_dict()!=calibration_before:
            raise RuntimeError('target prediction mutated calibration')
        manifest['status']='SEALED'
        (folder/'seal.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        return manifest
    finally:
        for module,training in modes:
            module.training=training


def _compact_metrics(scores,labels,accepted):
    return {key:value for key,value in selective_metrics(scores,labels,accepted).items() if key!='risk_coverage'}


def score_target(pred_dir,truth_mapping_file,output):
    """Independent scorer; caller invokes only after ALL candidate seals complete.

    Truth JSON: {"rows": {opaque_id: {"label": int, "receiver": value,"day":value}}}.
    Prediction ordering never controls label matching; exact opaque ID sets do.
    """
    destination = Path(output)
    if destination.exists():
        raise FileExistsError('scoring output already exists')
    folder = Path(pred_dir)
    manifest = json.loads((folder/'seal.json').read_text(encoding='utf-8'))
    if manifest.get('status')!='SEALED' or set(manifest.get('scenes',{}))!=set(SCENES):
        raise ValueError('all four Phase1 scenes must be sealed before truth access')
    # Only now open the independent truth mapping.
    truth = json.loads(Path(truth_mapping_file).read_text(encoding='utf-8'))['rows']
    report = dict(schema='phase1_evidence_target_scores_v1',phase='Phase1 frozen R_t',
                  source=str(folder),scenes={},scientific_promotion_claim=False,
                  interpretation='single-seed descriptive frozen evaluation; no target fitting or selection')
    for scene in SCENES:
        artifact = torch.load(folder/(scene+'.pt'),map_location='cpu',weights_only=True)
        if any(not torch.is_tensor(value) for value in artifact.values()):
            raise ValueError('prediction artifact must contain tensors only')
        forbidden = {'labels','label','truth','receiver','day','tx','rx'} & artifact.keys()
        if forbidden:
            raise ValueError('prediction contains forbidden truth fields')
        ids = [bytes(row.tolist()).hex() for row in artifact['ids']]
        if len(set(ids))!=len(ids) or set(ids)!=set(truth) or len(ids)!=manifest['rows']:
            raise ValueError('prediction/truth physical ID sets differ')
        labels = torch.tensor([int(truth[key]['label']) for key in ids])
        temperatures = artifact['temperature']
        if not torch.isfinite(temperatures).all() or (temperatures<=0).any():
            raise ValueError('invalid frozen temperature')
        scores = artifact['scores']/temperatures[:,None]
        accepted = artifact['accepted']
        overall = selective_metrics(scores,labels,accepted)
        summary = dict(overall={key:value for key,value in overall.items() if key!='risk_coverage'},
                       risk_coverage=overall['risk_coverage'],by_receiver={},by_day={},
                       by_receiver_day={},by_class={},by_receiver_class={})
        for fields,name in ((('receiver',),'by_receiver'),(('day',),'by_day'),
                            (('receiver','day'),'by_receiver_day'),(('label',),'by_class'),
                            (('receiver','label'),'by_receiver_class')):
            groups = {}
            for i,key in enumerate(ids):
                group = json.dumps([truth[key][field] for field in fields],separators=(',',':'))
                groups.setdefault(group,[]).append(i)
            for group,indices in groups.items():
                take = torch.tensor(indices)
                summary[name][group] = _compact_metrics(scores[take],labels[take],accepted[take])
        summary['receiver_accuracy_floor'] = min(value['accuracy'] for value in summary['by_receiver'].values())
        summary['receiver_closed_set_accuracy_floor'] = min(value['closed_set_accuracy'] for value in summary['by_receiver'].values())
        for name,key in (('class','by_class'),('receiver_class','by_receiver_class')):
            summary[name+'_accuracy_floor'] = min(value['accuracy'] for value in summary[key].values())
            summary[name+'_closed_set_accuracy_floor'] = min(value['closed_set_accuracy'] for value in summary[key].values())
        seconds = float(manifest['scenes'][scene]['seconds'])
        summary['resources'] = dict(prediction_seconds=seconds,
            mean_seconds_per_row=seconds/len(ids),artifact_bytes=(folder/(scene+'.pt')).stat().st_size,
            timing_scope='IQ access, per-row channel transform, batched forward and tensor save; not isolated kernel latency')
        report['scenes'][scene]=summary
    report['resources'] = dict(total_prediction_seconds=sum(scene['resources']['prediction_seconds'] for scene in report['scenes'].values()),
        total_artifact_bytes=sum(scene['resources']['artifact_bytes'] for scene in report['scenes'].values()))
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
