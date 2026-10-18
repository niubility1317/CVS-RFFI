"""Small synthetic fixtures only: no CVS models, real IQ or checkpoints."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

path=Path(__file__).with_name('cvs_current_controls79_analysis_20261019.py')
spec=importlib.util.spec_from_file_location('current_controls79_analysis',path)
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def logits(y,correct):
    p=(y+1)%6;p=p.copy();p[:correct]=y[:correct]
    z=np.zeros((len(y),6),np.float32);z[np.arange(len(y)),p]=3.
    return z


def result(label):
    groups={k:dict(clean_norm=1.,satellite_norm=2.,dot=1.,cosine=.5,finite=True)for k in a.GROUPS if k!='whole'}
    groups['whole']=dict(clean_norm=8**.5,satellite_norm=32**.5,dot=8.,cosine=.5,finite=True)
    return dict(label=label,arm='B00'if label=='NOAS30'else'B01',status='SOURCE_DIAGNOSED',NN=121232,prediction_NN=120600,gradient_forward_NN=630,synthetic_smoke_NN=2,training_updates=0,head_fit_updates=0,target_access=False,state_unchanged=True,
        provenance=dict(run_id=a.PARENT,row_id=('B00'if label=='NOAS30'else'B01')+'-s2026092701',checkpoint='PROJECT/runs/'+a.PARENT+'/'+('B00'if label=='NOAS30'else'B01')+'-s2026092701/source/final_ssdg.pth',state_key='model',historical_epochs=30,historical_updates=6660,final_segment_epochs=10,final_segment_updates=2220,parent_epochs=20,parent_updates=4440),
        gates=dict(PA=[.1],reference=[.2],signature=[.03],kind='learned static coordinate gains; not invented per-packet gate weights'),
        gradients=[dict(draw=d,scene=s,count=90,clean_CE=1.,satellite_CE=2.,groups=copy.deepcopy(groups),scope='frozen weights, deterministic eval-mode smoothedCE, not stochastic training gradients')for d in (0,1)for s in a.SCENES])


def fixture(n=12):
    y=np.arange(n,dtype=np.int64)%6;ids=np.asarray(['pid'+str(i)for i in range(n)]);data={};branches={}
    for li,label in enumerate(a.LABELS):
        full=logits(y,6+li);sat=np.tile(full,(2,3,11,1,1));sat[:,:,1]=logits(y,7+2*li);sat[:,:,2]=logits(y,8+2*li);sat[:,:,3]=sat[:,:,1]
        sat[:,:,10]=full.copy();sat[:,:,10,0]=0.;sat[:,:,10,0,(int(y[0])+1)%6]=3.
        data[label]=dict(ids=ids,TX=y,RX=np.tile(np.asarray([1,3,4,6,8,1]),n//6),day=np.ones(n,dtype=np.int64),clean=logits(y,9+li),satellite=sat)
        b=np.ones((2,3,n,5,3),np.float32);b[...,0]=.5;branches[label]=b
    ph={k:np.ones((2,3,11,n),float)for k in a.PHYSICS}
    ph['state'][:]=0;ph['lock'][:]=0;ph['signal_to_added_noise_db'][:]=1.;ph['signal_to_added_noise_db'][:,:,0,n//2:]=9.
    return data,ph,branches,{label:result(label)for label in a.LABELS}


class Checks(unittest.TestCase):
    def test_metric_empty_macro_and_retention(self):
        y=np.asarray([0,0,1,1,2,2]);p=np.asarray([0,1,1,1,2,0]);z=np.zeros((6,6));z[np.arange(6),p]=30.;co=np.asarray([1,1,1,0,0,0],bool)
        m=a.metric(z,y,co,np.ones(6,bool));self.assertEqual(m['correct'],4);self.assertAlmostEqual(m['accuracy'],4/6);self.assertAlmostEqual(m['macro_f1'],(.5+.8+2/3)/6);self.assertEqual(m['fixed_NOAS30_clean_correct_retained'],2)
        e=a.metric(z[:0],y[:0],co[:0],co[:0]);self.assertTrue(e['empty_cell']);self.assertIsNone(e['accuracy']);self.assertIsNone(e['macro_f1']);self.assertEqual(sum(map(sum,e['confusion'])),0)
    def test_visible_metadata_shape_alignment(self):
        data,ph,b,r=fixture();x=data['NOAS30'];a.validate_predictions(x,x['ids'],n=12);a.validate_predictions(data['AS30'],x['ids'],x,n=12);a.validate_physics(ph,n=12)
        for key,value in [('TX',np.full(12,-1)),('RX',np.zeros(12,dtype=int)),('day',np.zeros(12,dtype=int)),('ids',x['ids'][::-1]),('satellite',x['satellite'][:,:,:10])]:
            v=dict(x);v[key]=value
            with self.assertRaises(ValueError):a.validate_predictions(v,x['ids'],n=12)
        v=copy.deepcopy(x);v['clean'][0,0]=np.nan
        with self.assertRaises(ValueError):a.validate_predictions(v,x['ids'],n=12)
        p=dict(ph);p['lock']=ph['lock'].copy();p['lock'][0,0,0,0]=3
        with self.assertRaises(ValueError):a.validate_physics(p,n=12)
    def test_branch_and_gradient_schema(self):
        data,ph,b,r=fixture();v=dict(names=np.asarray(a.BRANCHES),metrics=np.asarray(a.BRANCH_METRICS),values=b['NOAS30']);a.validate_branch(v,n=12)
        self.assertEqual(len(a.validate_result(r['AS30'],'AS30')),54)
        for key,value in [('NN',121231),('target_access',True),('state_unchanged',False)]:
            rr=copy.deepcopy(r['AS30']);rr[key]=value
            with self.assertRaises(ValueError):a.validate_result(rr,'AS30')
        rr=copy.deepcopy(r['AS30']);rr['provenance']['checkpoint']='old/P24/source/final_ssdg.pth'
        with self.assertRaises(ValueError):a.validate_result(rr,'AS30')
        rr=copy.deepcopy(r['AS30']);rr['gradients'][0]['groups']['whole']['dot']=9.
        with self.assertRaises(ValueError):a.validate_result(rr,'AS30')
        v['values']=v['values'].copy();v['values'][0,0,0,0,0]=np.inf
        with self.assertRaises(ValueError):a.validate_branch(v,n=12)
    def test_real_emitted_label_arm_provenance(self):
        for label,arm in [('NOAS30','B00'),('AS30','B01')]:
            r=result(label);self.assertEqual(r['label'],label);self.assertEqual(r['arm'],arm);self.assertEqual(len(a.validate_result(r,label)),54)
            for key,value in [('label',arm),('arm',label)]:
                bad=copy.deepcopy(r);bad[key]=value
                with self.assertRaises(ValueError):a.validate_result(bad,label)
            for key,value in [('run_id','old_P24'),('row_id',label+'-s2026092701'),('state_key','ema_model'),('checkpoint','old/P24/'+arm+'-s2026092701/source/final_ssdg.pth')]:
                bad=copy.deepcopy(r);bad['provenance'][key]=value
                with self.assertRaises(ValueError):a.validate_result(bad,label)
    def test_same_full_strata_empty_and_partition(self):
        data,ph,b,r=fixture();x=data['NOAS30'];f={k:ph[k][0,0,0]for k in('state','lock','signal_to_added_noise_db')};ss=a.strata(x['TX'],x['RX'],x['day'],f)
        for dim in {s[0]for s in ss}-{'overall'}:self.assertTrue(np.all(np.sum([v for k,_,v in ss if k==dim],axis=0)==1))
        self.assertTrue(any(k=='state'and label=='BLOCKED'and not v.any()for k,label,v in ss))
        ph['state'][:,:,1]=2
        self.assertEqual(sum(v.sum()for k,label,v in ss if k=='state'and label=='BLOCKED'),0)
    def test_paired_net_CM_and_margin_scale(self):
        y=np.arange(12)%6;co=np.ones(12,bool);z0=logits(y,6);z1=logits(y,8);q=a.paired(z0,z1,y,co,co)
        self.assertEqual((q['repairs'],q['new_errors'],q['net_repairs']),(2,0,2));self.assertAlmostEqual(q['accuracy_delta_pp'],100*2/12)
        m=a.metric(z1,y,co,co);base=a.metric(z0,y,co,co);self.assertEqual(m['correct']-base['correct'],q['net_repairs']);self.assertAlmostEqual(m['mean_margin_over30']-base['mean_margin_over30'],(4/12)*3/30)
    def test_control_training_interaction_and_nonadditivity(self):
        data,ph,b,r=fixture();t=a.analyze(data,ph,b,r)
        def one(table,**kw):return next(v for v in t[table]if all(v.get(k)==x for k,x in kw.items()))
        ctx=dict(draw=0,scene='practical_mid_urban',dimension='overall',stratum='ALL')
        no=one('effects',label='NOAS30',condition='no_noise/fixed',**ctx);sat=one('effects',label='AS30',condition='no_noise/fixed',**ctx)
        self.assertEqual((no['net_repairs'],sat['net_repairs']),(1,2))
        ix=one('interactions',condition='no_noise/fixed',**ctx);self.assertEqual(ix['training_x_control_net'],1);self.assertAlmostEqual(ix['training_x_control_pp'],100/12);self.assertTrue(ix['removal_effects_not_additive'])
        flat=one('effects',label='NOAS30',condition='flat_H/fixed',**ctx);self.assertEqual(flat['net_repairs'],no['net_repairs'])
        self.assertEqual(no['net_repairs']+flat['net_repairs'],2) # Same repaired packet is shared; no combined-effect estimate is produced.
        self.assertTrue(all('combined_effect'not in v for v in t['effects']))
        recv=one('receiver_effects',label='AS30',control='no_noise',**ctx);self.assertEqual(recv['net_repairs'],1)
        self.assertEqual(one('metrics',label='AS30',condition='full',draw=-1,scene='midlow_pooled',dimension='overall',stratum='ALL')['count'],48)
        self.assertTrue(one('metrics',label='AS30',condition='full',draw=0,scene='practical_mid_urban',dimension='state',stratum='BLOCKED')['empty_cell'])
        self.assertEqual(one('training_effects',condition='clean',draw=-1,scene='clean',dimension='overall',stratum='ALL')['net_repairs'],1)
    def test_fixed_receiver_metadata_EQoff(self):
        data,ph,b,r=fixture();x=data['NOAS30'];records={}
        for ci,c in enumerate(a.CONDITIONS):
            v={k:float(ph[k][0,0,ci,0])for k in a.PHYSICS[2:]};v.update(state='LOS',receiver_lock_status='locked',true_total_SNR_known=False,channel_equalization_applied=False,phase_tracking_applied=True,boundary='edge'if c=='edge_boundary'else'reflect',effect='fixed_receiver_state_and_compensation'if c.endswith('/fixed')else'recomputed_receiver_state');records[c]=v
        row=dict(draw=0,scene=a.SCENES[0],index=0,physical_id=str(x['ids'][0]),TX=int(x['TX'][0]),RX=int(x['RX'][0]),day=1,session='rx1:day1:eq1',seed=77,conditions=records)
        a.check_record(row,0,ph,x,12)
        for field,value in [('channel_equalization_applied',True),('agc_gain_db',3.),('receiver_lock_status','unlocked')]:
            rr=copy.deepcopy(row);rr['conditions']['no_noise/fixed'][field]=value
            with self.assertRaises(ValueError):a.check_record(rr,0,ph,x,12)
        rr=copy.deepcopy(row);rr['conditions']['full']['true_total_SNR_known']=True
        with self.assertRaises(ValueError):a.check_record(rr,0,ph,x,12)
    @unittest.skipUnless(importlib.util.find_spec('matplotlib') is not None,'Local plotting dependency unavailable; render on existing analysis environment')
    def test_artifact_writer_and_figures(self):
        data,ph,b,r=fixture();t=a.analyze(data,ph,b,r)
        with tempfile.TemporaryDirectory()as td:
            out=Path(td);a.csv_write(out/'control.csv',t['effects']);a.write(out/'summary.json',dict(NN=0,cohorts_not_parent=True));figures=a.plot(out,t)
            self.assertEqual(len(figures),8);self.assertTrue(all((out/p).is_file()and(out/p).stat().st_size>100 for p in figures))
            self.assertEqual({Path(p).stem for p in figures},{'control_gain','source_strata','training_effect','branch_gradients'})
            with self.assertRaises(FileExistsError):a.write(out/'summary.json',{})
    def test_cli_rejects_truth_and_wrong_run_root(self):
        import sys
        from unittest.mock import patch
        with patch.object(sys,'argv',['analysis','--project','P','--out','O','--truth','x']):
            with self.assertRaises(SystemExit):a.main()
        with self.assertRaises(ValueError):a.load_inputs(Path('P'),'other_run')


if __name__=='__main__':unittest.main()
