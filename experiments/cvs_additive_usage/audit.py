"""Frozen internal model interventions on complete source V; never reselect."""
import argparse,copy,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'code')]
import torch
import torch.nn.functional as F
from experiments.cvs_additive_identity.usage import MODES,counterfactual_logits,assert_state_unchanged
from experiments.cvs_additive_identity.model import VARIANTS,build
from experiments.cvs_additive_identity.dispatch import SEEDS,PROJECT,read_source_record,validate_spec,select_source_candidate
from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY,numerical_context,actual_flags
from experiments.cvs_identity_ce.source import source_args
from baselines.common.practical_source import build_contract_split
from baselines.common.cvs_data import make_cvs_loader

RUN='20261002-diagnostic-cvs-additive-usage-source-manysig-m40-r01'
RELEASE='cvs_additive_usage_source_20261002_r01'
SOURCE_RUN='20261002-phase1-cvs-additive-identity-manysig-m8-r01'
SOURCE_COMMIT='d6c29b2bfbc5921a5ff654735253df21b1b1f709'
SOURCE=PROJECT+'/runs/'+SOURCE_RUN
MATRIX=PROJECT+'/releases/cvs_additive_identity_20261002_r01/experiments/cvs_additive_identity/configs/launch_spec.json'
CONTRACT=PROJECT+'/runs/phase1_daot_rc4_pure_game_m3_20260917_r2/source_contract.json'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,allow_nan=False,indent=2)


def validate_config(c):
    for k in ('target','target_truth','p1_capsule','p1_truth','resume','teacher','initial_checkpoint','augmentation'):
        if c.get(k):raise ValueError('Source-only frozen diagnostic rejects '+k)
    fixed=dict(method='cvs_additive_usage_source',run_id=RUN,source_contract=CONTRACT,source_matrix=MATRIX,
        source_root=SOURCE,dataset=PROJECT+'/Dataset_WigSig/ManySig.pkl',output_root=PROJECT+'/runs/'+RUN+'/source',
        modes=list(MODES),variants=list(VARIANTS),model_seeds=sorted(SEEDS),split_seed=392005,source_receivers=[1,3,4,6,8],source_days=[1,2,3],
        numerical_policy=FULL_FP32_POLICY,source_selection_frozen=True,used_for_selection=False,optimizer_steps=0)
    if any(c.get(k)!=v for k,v in fixed.items()):raise ValueError('Fixed complete source diagnostic differs')
    return c


def validate_payload(payload,row,resolved,contract,initial):
    expected=dict(epoch=200,source_contract=contract,initialization=initial,selection='fixed_last_epoch',method='cvs_additive_identity',
                  variant=row['variant'],config=resolved,classes=contract['classes'],num_classes=6)
    if any(payload.get(k)!=v for k,v in expected.items()):raise ValueError('Checkpoint payload/data/provenance mismatch')


def verified_sources(c):
    matrix=validate_spec(read(c['source_matrix']));original=read(c['source_contract'])
    records=[read_source_record(r,original,'cvs_residual_identity') for r in matrix['source_controls']]
    records += [read_source_record(r,original,'cvs_additive_identity') for r in matrix['rows']]
    selection=select_source_candidate(records)
    if selection!=read(Path(c['source_root'])/'source_selection.json'):raise ValueError('Actual source selection not frozen')
    sources=[]
    for r in matrix['rows']:
        q=Path(r['source_output']);resolved=read(q/'resolved_config.json');contract=read(q/'source_contract.json');initial=read(q/'initialization.json')
        if resolved['commit']!=SOURCE_COMMIT or resolved['backend_flags']!=FULL_FP32_POLICY:raise ValueError('Unexpected actual source release/precision')
        payload=torch.load(q/'last.pt',map_location='cpu',weights_only=False)
        validate_payload(payload,r,resolved,contract,initial)
        model=build(r['variant']);model.load_state_dict(payload['model'],strict=True);model.eval()
        with torch.no_grad():
            smoke=counterfactual_logits(model,torch.zeros(2,2,256))
        if tuple(smoke)!=MODES:raise ValueError('Counterfactual smoke missing modes')
        sources.append(dict(row=r,resolved=resolved,contract=contract,model=model))
    return sources,selection


def metrics(cm,ce,changed,delta):
    n=int(cm.sum());tp=cm.diag();den=cm.sum(0)+cm.sum(1)
    return dict(count=n,accuracy=int(tp.sum())/n,macro_f1=float(torch.where(den>0,2*tp.double()/den.clamp_min(1),0.).mean()),
                ce=ce/n,prediction_changed_count=changed,logit_response_max=delta,confusion=cm.tolist())


def run(c):
    validate_config(c)
    with numerical_context(FULL_FP32_POLICY):return _run(c)


@torch.no_grad()
def _run(c):
    out=Path(c['output_root']);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);start=time.perf_counter()
    sources,selection=verified_sources(c) # Verify all eight states before opening source IQ.
    split=build_contract_split(source_args(c))
    actual=read(out/'source_contract.json');original=read(c['source_contract'])
    if split.test or split.named_tests or split.split_info['counts']!={'L_s':6300,'U_s':56700,'V':27000} or any(actual.get(k)!=v for k,v in original.items()):raise ValueError('Source physical roles differ')
    if any(s['contract']!=actual for s in sources):raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    device=torch.device(c.get('device','cuda:0'))
    resolved=dict(c,pid=os.getpid(),cwd=os.getcwd(),argv=sys.argv,python=sys.executable,commit=(ROOT/'release_commit.txt').read_text().strip(),
        source_commit=SOURCE_COMMIT,source_counts=split.split_info['counts'],source_role='V',L_s_use='not_iterated',U_s_use='unused',
        target_access=False,source_selection_unchanged=True,model_updated=False,hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        backend_flags=actual_flags(),checkpoint_provenance='VERIFIED_ALL8_ACTUAL_SCRATCH_E200_FULL_CONTRACT_PAYLOAD_STRICT_STATE',
        claim='Frozen internal-coordinate ablations;not hardware interventions or TX/RX causal identification')
    write(out/'resolved_config.json',resolved);write(out/'source_selection_before.json',selection)
    print('RESOLVED_CONFIG '+json.dumps(resolved),flush=True)
    rows=[]
    for s in sources:
        model=s['model'].to(device).eval();before=copy.deepcopy(model.state_dict());r=s['row'];count=0
        accum={m:dict(cm=torch.zeros(6,6,dtype=torch.int64),ce=0.,changed=0,delta=0.,groups={}) for m in MODES}
        for batch in make_cvs_loader(split.val,batch_size=256,shuffle=False,num_workers=0,device=device,drop_last=False):
            x=batch['iq'].to(device);y=batch['label'].to(device);scores=counterfactual_logits(model,x);base=scores['trained'];base_pred=base.argmax(1).cpu();count+=len(y)
            for mode,logits in scores.items():
                a=accum[mode];pred=logits.argmax(1).cpu();yc=y.cpu();cm=torch.bincount(yc*6+pred,minlength=36).reshape(6,6)
                a['cm']+=cm;a['ce']+=float(F.cross_entropy(logits,y,reduction='sum'));a['changed']+=int((pred!=base_pred).sum());a['delta']=max(a['delta'],float((logits-base).abs().max()))
                for i in range(len(y)):
                    key=(int(yc[i]),int(batch['receiver'][i]),int(batch['day'][i]));g=a['groups'].setdefault(key,[0,0,0])
                    g[0]+=1;g[1]+=int(pred[i]==yc[i]);g[2]+=int(pred[i]!=base_pred[i])
        assert_state_unchanged(model,before)
        if count!=27000:raise ValueError('Incomplete source V')
        expected_cells={(t,rx,d) for t in range(6) for rx in (1,3,4,6,8) for d in (1,2,3)}
        for mode,a in accum.items():
            if set(a['groups'])!=expected_cells or any(g[0]!=300 for g in a['groups'].values()):raise ValueError('Incomplete full source cells')
            groups=[dict(tx=t,receiver=rx,day=d,count=g[0],correct=g[1],changed_count=g[2]) for (t,rx,d),g in sorted(a['groups'].items())]
            row=dict(row_id=r['row_id']+'--'+mode,variant=r['variant'],model_seed=r['model_seed'],mode=mode,groups=groups,**metrics(a['cm'],a['ce'],a['changed'],a['delta']))
            if mode=='trained':
                original_final=read(Path(r['source_output'])/'completion.json')['final_source_metrics']
                if row['accuracy']!=original_final['source_val_accuracy']:raise ValueError('Actual trained forward differs from frozen E200 validation')
            path=out/row['row_id'];path.mkdir();write(path/'result.json',row);rows.append(row)
            print('ROW '+json.dumps({k:v for k,v in row.items() if k not in ('groups','confusion')}),flush=True)
        model.cpu();del before
    if read(Path(c['source_root'])/'source_selection.json')!=selection:raise ValueError('Frozen source selection changed')
    result=dict(status='SOURCE_DIAGNOSTIC_COMPLETE',run_id=RUN,rows=rows,row_count=40,source_V_count_per_row=27000,
        source_commit=SOURCE_COMMIT,source_selection=selection,source_selection_unchanged=True,used_for_selection=False,
        model_updated=False,optimizer_steps=0,training_augmentation=False,target_access=False,formal_CVS_training=False,
        elapsed_seconds=time.perf_counter()-start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None)
    write(out/'usage_statistics.json',result);write(out/'completion.json',{k:v for k,v in result.items() if k not in ('rows','source_selection')})
    print('COMPLETE '+json.dumps(read(out/'completion.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(read(p.parse_args().config))
