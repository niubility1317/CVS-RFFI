"""OVA source orchestration contracts using synthetic cached roles only."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class StarOvaRunTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_ova_run')
    def config(self,root):
        from test_pamf_phase3_star_consensus_run import StarConsensusRunTests
        c=StarConsensusRunTests().config(root);run='20261029-phase3-star-ova-evidence-s2026092701-r01'
        c['run_id']=run;c['source_config']['run_id']=run;c['source_config']['row_id']='star-ova-s2026092701'
        return c
    def role(self,name):
        from test_pamf_phase3_star_consensus_run import StarConsensusRunTests
        r=StarConsensusRunTests().role(name);r['restored'][:,1,255]=1;return r
    def state(self,m):
        return dict(schema=m.READOUT_SCHEMA,descriptor_schema=m.DESCRIPTOR_SCHEMA,descriptor_dim=106,class_count=6,
            model_names=list(m.MODEL_NAMES),mean=np.zeros(106),scale=np.ones(106),models={name:dict(
                weight1=np.zeros((32,106),np.float32),bias1=np.zeros(32,np.float32),
                weight2=np.zeros((1,32),np.float32),bias2=np.zeros(1,np.float32)) for name in m.MODEL_NAMES},
            training=dict(steps_per_arm=1000,total_optimizer_updates=2000,known_count=30,oe_count=30,
                model_seed=2026092701,initial_parameters_equal=True,parameter_counts={k:3457 for k in m.MODEL_NAMES},
                losses={k:[.5]*1000 for k in m.MODEL_NAMES}))
    def patches(self,m,c,events,*,defect=None,failed_calibration=False):
        from contextlib import ExitStack
        stack=ExitStack();roles={name:self.role(name) for name in ('L','OE','V','novel')}
        if defect=='baseline':roles['V']['baseline_closed'][:,1]=0
        def role(path,name,*args):
            events.append(('role',name))
            if name=='V':self.assertIn(('fit',),events)
            if name=='novel':
                cal=json.loads((Path(c['output_root'])/'calibration/complete.json').read_text())
                self.assertEqual(set(cal['methods']),set(m.METHODS));self.assertEqual(events.count(('calibrate',)),4)
                if defect=='novel':raise ValueError('missing novel')
            return roles[name]
        def encode(reader,x,device):
            p1=int(np.all(x[:,1,255]==1));y=x[:,0,0].astype(int)
            return {'fused160':np.full((len(x),1),p1,np.float32)},np.eye(6,dtype=np.float32)[(y+p1)%6]*5
        def desc(p1,l1,g1,p0,l0,g0):
            self.assertTrue(np.all(p1['fused160']==1));self.assertTrue(np.all(p0['fused160']==0))
            return np.tile(np.arange(106,dtype=np.float32),(len(l1),6,1))
        def fit(a,b,y,closed,kd,od,model_seed):
            events.append(('fit',));np.testing.assert_array_equal(closed,(y+1)%6);self.assertEqual(model_seed,2026092701)
            return self.state(m)
        def score(desc,state):
            if defect=='mutation':state['models'][m.MODEL_NAMES[0]]['bias2'][0]+=1
            raw=np.tile(np.arange(6,dtype=np.float64),(len(desc),2,1));raw[:,1]+=10
            return raw
        def calibrate(ids,y,closed,baseline,scores,rx,scene):
            events.append(('calibrate',));np.testing.assert_array_equal(closed,baseline)
            thresholds=np.arange(6,dtype=np.float64)/10
            state=dict(schema='pamf_phase3_star_class_risk_state_v1',global_threshold=-1.,conditional_thresholds=thresholds)
            return state,dict(known_only=True,unknown_inputs_used=False,group_count=42,
                methods=dict(global_control=dict(constraints_passed=not failed_calibration),
                    predicted_class=dict(constraints_passed=not failed_calibration,groups={})))
        items=[patch.object(m,'COUNTS',dict(L=30,OE=30,V=180,novel=30)),
            patch.object(m,'_backend',return_value=({'reader':object(),'dura':None,'reader_identity':roles['L']['manifest']['parent_F0_identity']},{})),
            patch.object(m,'_snapshot',return_value=[]),patch.object(m,'_check_models'),
            patch.object(m,'_smoke',side_effect=lambda *a:events.append(('smoke',)) or dict(status='PASS')),
            patch.object(m,'_role',side_effect=role),patch.object(m,'close_role'),patch.object(m,'_encode',side_effect=encode),
            patch.object(m,'_branches',side_effect=lambda reader,x,device,batch:encode(reader,x,device)),
            patch.object(m.geometry_core,'fit_geometry',return_value={'value':np.array([1.])}),
            patch.object(m.core,'descriptor',side_effect=desc),patch.object(m.core,'fit',side_effect=fit),
            patch.object(m.core,'score',side_effect=score),patch.object(m.risk,'calibrate_known',side_effect=calibrate),
            patch.object(m,'_save_geometry',side_effect=lambda p,g:m._npz(p,g))]
        for item in items:stack.enter_context(item)
        return stack

    def test_fixed_source_config_rejects_extra_roles_truth_and_old_parent(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td));self.assertIs(m.validate_config(c),c)
            for key in ('target_root','truth','U_public_root','unknown_count','checkpoint'):
                with self.subTest(key=key),self.assertRaises(ValueError):m.validate_config(dict(c,**{key:'x'}))
            for key in ('truth','trained_checkpoint','V_root'):
                bad=copy.deepcopy(c);bad['source_config'][key]='x'
                with self.subTest(nested=key),self.assertRaises(ValueError):m.validate_config(bad)
            bad=copy.deepcopy(c);bad['source_config']['row_id']='star-paired-s2026092701'
            with self.assertRaises(ValueError):m.validate_config(bad)

    def test_four_methods_two_models_P1_identity_freeze_order_and_marker_last(self):
        m=self.module();events=[]
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td));writes=[];write=m._write
            with self.patches(m,c,events),patch.object(m,'_write',side_effect=lambda p,v:writes.append(str(p)) or write(p,v)):
                done=m.run(c,'cpu')
            self.assertEqual(events[0],('smoke',));self.assertEqual([e[1] for e in events if e[0]=='role'],['L','OE','V','novel'])
            self.assertEqual(events.count(('fit',)),1);self.assertEqual(done['readout_optimizer_updates_total'],2000)
            self.assertEqual(done['readout_model_count'],2);self.assertEqual(done['primary_method'],'ova_max')
            self.assertEqual(done['new_encoder_updates'],0);self.assertEqual(done['G_forward_rows'],0)
            self.assertEqual(writes[-1],str(Path(c['output_root'])/'complete.json'))
            with np.load(Path(c['output_root'])/'predict/known_predictions.npz') as z:
                self.assertEqual(set(z.files),m.KNOWN_FIELDS);closed=z['closed'];self.assertEqual(z['scores'].shape,(180,4))
                np.testing.assert_array_equal(closed,z['baseline_closed']);np.testing.assert_array_equal(closed,self.role('V')['baseline_closed'][:,1])
                expected=np.column_stack((closed, np.full(len(closed),5),closed+10,np.full(len(closed),15)))-closed[:,None]/10
                np.testing.assert_allclose(z['scores'],expected)
            with np.load(Path(c['output_root'])/'thresholds.npz') as z:
                self.assertEqual(set(z.files),{'global','conditional'});self.assertEqual(z['conditional'].shape,(4,6))
            with np.load(Path(c['output_root'])/'predict/novel_predictions.npz') as z:self.assertEqual(set(z.files),m.NOVEL_FIELDS)
            with self.assertRaises(FileExistsError):m.run(c,'cpu')

    def test_failure_and_state_mutation_preserve_partial_without_final_marker(self):
        m=self.module()
        for defect in ('baseline','mutation','novel'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                c=self.config(Path(td));events=[]
                with self.patches(m,c,events,defect=defect),self.assertRaises(ValueError):m.run(c,'cpu')
                self.assertFalse((Path(c['output_root'])/'complete.json').exists())
                if defect!='novel':self.assertNotIn(('role','novel'),events)

    def test_failed_known_science_still_completes_all_fixed_methods(self):
        m=self.module();events=[]
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td))
            with self.patches(m,c,events,failed_calibration=True):done=m.run(c,'cpu')
            self.assertEqual(done['status'],m.STATUS);self.assertIn(('role','novel'),events)
            report=json.loads((Path(c['output_root'])/'calibration/complete.json').read_text())
            self.assertFalse(report['constraints_passed']);self.assertEqual(set(report['methods']),set(m.METHODS))

    def test_readout_codec_full_shapes_budget_and_pure_score_roundtrip(self):
        m=self.module();state=self.state(m);rng=np.random.default_rng(79);n=30;y=np.arange(n)%6
        branches={key:rng.normal(0,.1,(n,d)).astype(np.float32) for key,d in m.previous.previous.DIMENSIONS.items()}
        for values in branches.values():values[np.arange(n),y]+=2
        branches['reference29'][:,27]=0;logits=np.eye(6,dtype=np.float32)[y]*4
        geometry=m.geometry_core.fit_geometry(branches,y)
        x=m.core.descriptor(branches,logits,geometry,branches,logits,geometry)
        self.assertEqual(x.shape,(n,6,106));self.assertEqual(x.dtype,np.float32)
        with tempfile.TemporaryDirectory() as td,patch.object(m,'COUNTS',dict(L=30,OE=30,V=180,novel=30)):
            root=Path(td);m._save_readouts(root/'readouts.npz',state);m._write(root/'readouts.json',dict(training=state['training']))
            loaded=m.load_readouts(root/'readouts.npz',root/'readouts.json')
            np.testing.assert_array_equal(m.core.score(x,state),m.core.score(x,loaded))
            with np.load(root/'readouts.npz') as z:arrays={k:z[k].copy() for k in z.files}
            for defect in ('shape','dtype','missing','extra'):
                bad={k:v.copy() for k,v in arrays.items()}
                if defect=='shape':bad['model__membership__weight1']=np.zeros((32,100),np.float32)
                elif defect=='dtype':bad['mean']=bad['mean'].astype(np.float32)
                elif defect=='missing':bad.pop('model__ova_hard__bias2')
                else:bad['private_bank']=np.ones(2)
                path=root/(defect+'.npz');np.savez(path,**bad)
                with self.subTest(defect=defect),self.assertRaises(ValueError):m.load_readouts(path,root/'readouts.json')
            state['training']['total_optimizer_updates']=4000
            m._write(root/'bad.json',dict(training=state['training']))
            with self.assertRaises(ValueError):m.load_readouts(root/'readouts.npz',root/'bad.json')


if __name__=='__main__':unittest.main()
