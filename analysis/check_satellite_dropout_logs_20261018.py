"""Synthetic real-format own20+10 coverage; no real logs, IQ, CP or target."""
import copy
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from analysis import cvs_satellite_dropout_logs_20261018 as m


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False)+'\n',encoding='utf8')


def jl(path, values):
    path.parent.mkdir(parents=True,exist_ok=True)
    with (gzip.open(path,'wt',encoding='utf8') if path.suffix=='.gz' else path.open('w',encoding='utf8')) as f:
        for v in values:
            f.write(json.dumps(v,allow_nan=False)+'\n')


def native(path, values):
    jl(path.with_suffix('.jsonl'),values)
    with path.with_suffix('.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader()
        w.writerows({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in values)


def pid(i,role):
    return [0,m.RX[i%5],1+(i//5)%3,i+(100000 if role=='U' else 0),i+(100000 if role=='U' else 0)]


def sat_record(arm,i,offset):
    det=arm=='AS_DET'
    return dict(step=i,cumulative_step=offset*222+i,mode=arm,As_supervised=64,extra_NN=0,parameter_updates_extra=0,
                dropout_calls=2,bypassed_elements=16384 if det else 0,original_mixed_KL=.02 if det else .05,
                ordinary_half_KL=.04,satellite_half_KL=0. if det else .06,used_mixed_KL=.02,removed_satellite_KL=0. if det else .03,
                satellite_pair_logit_maxdiff=0. if det else 1.)


def fixture(project,arm):
    new=arm.startswith('AS_');sat=arm!='B00'
    row=project/'runs'/(m.RUN if new else m.REFERENCE)/(arm+'-s2026092701')
    contract=project/'contract.json';dump(contract,dict(role_ids={'L_s':['opaque-L'],'U_s':['opaque-U'],'V':['opaque-V']}))
    for stage,epochs,labels,offset in m.SEGMENTS:
        b=row/stage;steps=epochs*222;nu=(epochs-labels)*56700
        parents=[] if not offset else [str(row/'stage20/final_ssdg.pth')]
        cfg=dict(run_id=m.RUN if new else m.REFERENCE,row_id=row.name,model_seed=2026092701,source_contract=str(contract),epochs=epochs,
                 label_epochs=labels,pseudo_epochs=epochs-labels,updates=steps,steps_per_epoch=222,epoch_offset=offset,step_offset=offset*222,
                 new_Ag=False,Ag_mode='NONE',As=sat,L_NN_per_step=768,total_L_loss_scale=1.25,target_access=False,checkpoint_sources=parents,
                 data_contract=dict(RX=list(m.RX),days=[1,2,3],L=6300,U=56700,V=27000,split=392005,source_RX_holdout=None,internal_L_split=False,U_TX_truth_hidden=True))
        if new:cfg['satellite_mode']=arm
        init=dict(checkpoint_sources=parents,scratch_only=not offset,source_roles='EXACT_MATCH',target_contact=False,ancestors=[] if not offset else [str(row/'stage20')],teacher_origin='own scratch student' if not offset else 'strict own parent EMA')
        dump(b/'initialization.json',init);dump(b/'source_contract.json',json.loads(contract.read_text()))
        dump(b/'source_frozen.json',dict(config=cfg,selection='final_only',target_access=False))
        dump(b/'completion.json',dict(config=cfg,epoch=epochs,optimizer_steps=steps,target_access=False,target_evaluated=False,head_fit_updates=0,native_disabled_probe_calls=1,native_fitted_probes_disabled=True,elapsed_seconds=100.))
        args=dict(epochs=epochs,label_epochs=labels,pseudo_epochs=epochs-labels,baseline_ckpt=parents[0] if offset else '',teacher_ckpt='',freeze_backbone=False,candidate_id=row.name,output_dir=str(b))
        dump(b/'actual_native_args.json',{'args':args})
        dump(b/'optimizer_initialization.json',dict(state_entries=0,reset=True,schedule='cosine'+str(epochs)))
        dump(b/'EMA_initialization.json',dict(origin=init['teacher_origin'],extra_teacher=False,optimizer_segment_counters_reset=True,temporal_cache_reset=True))
        dump(b/'ordinary_augmentation_actual.json',dict(base={'enabled':True},args={'aug_enabled':True},schedule_frozen_at=13,no_ordinary_before_practical=True,RDrop_same_IQ=True))
        result=dict(status='SEGMENT_TRAINED',config=cfg,optimizer_steps=steps,L_NN=steps*768,L_As=steps*(64 if sat else 0),Ag_packets=0,Ag_control_NN=steps*256,total_L_loss_scale=1.25,extra_U_NN=0,U_As=0,U_truth_read=False,target_access=False,head_fit_updates=0,source_RX=list(m.RX),source_RX_holdout=None,
                    student_train_NN=steps*768+nu,student_eval_NN=epochs*27000+54002,teacher_train_NN=0,teacher_eval_NN=nu,teacher_counter_offset={'train':0,'eval':2})
        if new:result['satellite_subnetwork']=dict(dropout_inventory=[{'name':'id_backbone.encoder.synthetic','p':.1}],mode=arm,steps=steps,extra_NN=0,bypassed_elements=steps*(16384 if arm=='AS_DET' else 0))
        dump(b/'result.json',result)
        refs=[];shapes=[];metrics=[];units=[];satellites=[]
        for i in range(1,steps+1):
            ep=(i-1)//222+1;cumep=offset+ep;weight=.1*min(max((cumep-5)/10,0),1)
            extra_kl=.02 if new else .05
            common=.2+.03;extra=.3+extra_kl;shape=.4+.05;before=.5*common+.5*extra+weight*.1;total=before+.25*shape
            r=dict(step=i,cumulative_step=offset*222+i,epoch=ep,L_forwards=6,As_packets=64 if sat else 0,Ag_packets=0,Ag_mode='NONE',extra_control_packets=128,ordinary_redraws=0,same_IQ_each_pair=True,all_satellite_L_supervised=True,
                   common_CE=.2,common_KL=.03,common_RD=common,extra_CE=.3,extra_KL=extra_kl,extra_RD=extra,SupCon=.1,SupCon_weight=weight,total_L=total,base_L_before_shape=before,ordinary_raw_MSE=.01+(i%5)/1000,
                   scene_counts={m.SCENES[i%3]:64} if sat else {'NONE':64},state_counts={'LOS':64} if sat else {'NOT_APPLICABLE':64},crossview_mode='ORD128',new_losses=0,extra_U_NN=0,
                   SupCon_details=[dict(phase=p,level=l,mode='ORD128',total_anchors=128,ordinary_anchors=128,additional_anchors=0,same_RX_positive_pairs=0,true_satellite_anchors=0,valid_anchors=128) for p in ('first','second') for l in ('time_down','t2.pool')] if weight else [])
            if new:
                q=sat_record(arm,i,offset);satellites.append(q);r['satellite_subnetwork']={k:v for k,v in q.items() if k!='cumulative_step'}
            refs.append(r)
            shapes.append(dict(step=i,cumulative_step=offset*222+i,mode='NONE',NN=256,packets=128,applied=0,weight=.25,same_IQ_each_pair=True,new_boundary_samples=0,relative_noise_RMS=0.,noise_peak_fraction_mean=0.,SNR_added_db=None,shape_CE=.4,shape_KL=.05,shape_RD=shape))
            met={'concat/L_forwards':4,'concat/As_packets':64,'concat/total_L':before,'shape/RD':shape,'train/loss':total,'train/pseudo_truth_available':0}
            if ep>labels:met.update({'train/pseudo_total':124 if i%222==0 else 256,'train/pseudo_selected':100})
            metrics.append(dict(step=i,epoch=ep,metrics=met));units.append(dict(step=i,cumulative_step=offset*222+i,physical_clip_calls=i))
        for name,values in [('reference_steps',refs),('shape_steps',shapes),('step_metrics',metrics),('unit_steps',units)]:jl(b/(name+'.jsonl.gz'),values)
        if new:jl(b/'satellite_subnetwork_steps.jsonl.gz',satellites)
        epoch=[dict(epoch=i,lr=.001,epoch_time_s=5.,val_tx_acc=98.,train_grad_backbone=.1,train_grad_total=.2,aug_gain=.1,native_unavailable=None,nested=[i]) for i in range(1,epochs+1)]
        native(b/'metrics_epoch',epoch);native(b/'epoch_metrics',[{k:v for k,v in q.items() if not isinstance(v,(list,dict,tuple))} for q in epoch])
        def sequence():
            for i in range(steps):yield dict(role='L',ids=[pid((i*128+j)%6300,'L') for j in range(128)])
            for ep in range(epochs-labels):
                for i in range(222):yield dict(role='U',ids=[pid(i*256+j,'U') for j in range(124 if i==221 else 256)])
        jl(b/'training_sequence.jsonl.gz',sequence())
    log=row.parent/'logs'/(row.name+'.log');log.parent.mkdir(exist_ok=True);log.write_text('SMOKE PASS\nEpoch logs complete\n',encoding='utf8')
    return row


class CompleteLogs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='satellite_dropout_synthetic_');cls.project=Path(cls.temp.name)
        cls.row=fixture(cls.project,'AS_DET');cls.contract=json.loads((cls.project/'contract.json').read_text())['role_ids']
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def segment(self,stage='stage20'):
        spec=next(v for v in m.SEGMENTS if v[0]==stage)
        return m.parse_segment(self.row,'AS_DET',*spec,self.contract)
    def mutate_json(self,relative,key,value,body):
        path=self.row/relative;old=path.read_text(encoding='utf8');v=json.loads(old);v[key]=value;dump(path,v)
        try:body()
        finally:path.write_text(old,encoding='utf8')
    def test_complete_four_arm_run(self):
        for arm in ('B00','B01','AS_CE'):fixture(self.project,arm)
        value=m.run(self.project)
        self.assertEqual(value['status'],'VERIFIED_COMPLETE_SOURCE_TRAINING_LOGS')
        self.assertEqual(value['new_training']['updates'],13320)
        self.assertEqual(value['new_training']['native_NN'],15921368)
        self.assertEqual(value['reused_training']['native_NN'],15921368)
        self.assertFalse(value['analysis_cost']['target_scores_read'])
        self.assertEqual(value['matched_comparisons']['AS_DET']['source']['channel_scene_state_mismatch_vs_B01'],0)
        serialized=json.dumps(value,allow_nan=False)
        self.assertNotIn(str(self.project),serialized)
        # Equal packet counts and costs alone do not prove matched realizations.
        p=self.project/'runs'/m.RUN/'AS_CE-s2026092701/source/reference_steps.jsonl.gz'
        old=p.read_bytes();q=list(m.records(p));q[-1]['ordinary_raw_MSE']+=.001;jl(p,q)
        try:
            with self.assertRaisesRegex(ValueError,'Matched physical/ordinary'):m.run(self.project)
        finally:p.write_bytes(old)
    def test_complete_two_segments_6660(self):
        parts=[self.segment(s)[0] for s in ('stage20','source')]
        self.assertEqual(sum(s['updates'] for s in parts),6660)
        self.assertEqual([c['epoch'] for s in parts for c in s['epoch_curves']],list(range(1,31)))
        cost={k:sum(s['cost'][k] for s in parts) for k in m.COST_KEYS}
        self.assertEqual(cost,dict(student_train_NN=6078780,student_eval_NN=918004,teacher_train_NN=0,teacher_eval_NN=963900))
        self.assertEqual(sum(s['counts']['As_packets'] for s in parts),426240)
        self.assertEqual(parts[0]['native_telemetry']['metrics_epoch']['null_field_counts']['root[].native_unavailable'],20)
        self.assertEqual(parts[1]['teacher_counter_offset'],{'train':0,'eval':2})
        json.dumps(parts,allow_nan=False)
    def test_missing_last_step_is_wait(self):
        path=self.row/'source/unit_steps.jsonl.gz';old=path.read_bytes();values=list(m.records(path));jl(path,values[:-1])
        try:
            with self.assertRaises(m.IncompleteTrainingLogs):self.segment('source')
        finally:path.write_bytes(old)
    def test_missing_completion_is_wait(self):
        path=self.row/'source/completion.json';old=path.read_bytes();path.unlink()
        try:
            with self.assertRaises(m.IncompleteTrainingLogs):self.segment('source')
        finally:path.write_bytes(old)
    def test_incorrect_budget_rejected(self):
        def body():
            with self.assertRaisesRegex(ValueError,'budget|epoch/update'):self.segment('source')
        self.mutate_json('source/completion.json','optimizer_steps',2219,body)
    def test_teacher_offset_not_double_charged(self):
        path=self.row/'source/result.json';old=path.read_text();v=json.loads(old);v['teacher_counter_offset']['eval']=4;dump(path,v)
        try:
            with self.assertRaisesRegex(ValueError,'offset'):self.segment('source')
        finally:path.write_text(old,encoding='utf8')
    def test_modes_and_removed_KL(self):
        inv=[{'name':'id_backbone.encoder.synthetic','p':.1}]
        for arm in ('AS_CE','AS_DET'):self.assertLess(m.satellite_record(sat_record(arm,1,20),arm,1,4441,inv),1e-12)
        for key,value in [('mode','AS_CE'),('bypassed_elements',0),('As_supervised',63),('dropout_calls',3),('used_mixed_KL',.04),('removed_satellite_KL',.1),('satellite_pair_logit_maxdiff',.1)]:
            q=sat_record('AS_DET',1,20);q[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.satellite_record(q,'AS_DET',1,4441,inv)
    def test_matched_full_sequence_and_difference(self):
        a=self.row/'source/training_sequence.jsonl.gz';b=self.project/'synthetic-comparison.jsonl.gz'
        b.write_bytes(a.read_bytes());self.assertEqual(m.compare_streams(a,b),dict(records=4440,mismatch=0,all_equal=True))
        values=list(m.records(b));values[-1]['ids'][0][-1]+=1;jl(b,values)
        self.assertEqual(m.compare_streams(a,b)['mismatch'],1)
    def test_truncated_line_is_wait(self):
        p=self.project/'truncated.jsonl';p.write_text('{"step": 1}',encoding='utf8')
        with self.assertRaises(m.IncompleteTrainingLogs):list(m.records(p))
    def test_public_path_redaction(self):
        value={'config':['E:/private/fixture','/private/fixture',r'F:\private\fixture','safe metadata']}
        self.assertEqual(m.public_values(value),{'config':['<private-path>','<private-path>','<private-path>','safe metadata']})
    def test_nonfinite_nested_value_rejected(self):
        with self.assertRaises(ValueError):m.scan_numbers([{'q':[float('nan')]}])


if __name__=='__main__':unittest.main(verbosity=2)
