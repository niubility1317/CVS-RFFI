"""Source-only paired runner contracts with synthetic cache rows."""
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


class StarPairedRunTests(unittest.TestCase):
    def module(self):return importlib.import_module('experiments.pamf_phase3_open_world.star_paired_run')
    def config(self,root):
        from test_pamf_phase3_star_consensus_run import StarConsensusRunTests
        c=StarConsensusRunTests().config(root);run='20261029-phase3-star-paired-evidence-s2026092701-r01'
        c['run_id']=run;c['source_config']['run_id']=run;c['source_config']['row_id']='star-paired-s2026092701'
        return c
    def role(self,role):
        from test_pamf_phase3_star_consensus_run import StarConsensusRunTests
        row=StarConsensusRunTests().role(role);row['restored'][:,1,255]=1
        return row
    def state(self,m):
        return dict(schema=m.READOUT_SCHEMA,pair_dim=200,method_names=list(m.LEARNED_METHODS),
            normalizers={name:dict(mean=np.zeros(200),scale=np.ones(200)) for name in ('p1_only','paired')},
            models={name:dict(weight1=np.zeros((16,200),np.float32),bias1=np.zeros(16,np.float32),
                weight2=np.zeros((1,16),np.float32),bias2=np.zeros(1,np.float32)) for name in m.LEARNED_METHODS},
            training=dict(steps_per_arm=1000,total_optimizer_updates=2000,checkpoint_steps=[200,1000],
                checkpoint_updates={name:200 if name.endswith('200') else 1000 for name in m.LEARNED_METHODS},known_count=30,oe_count=30,
                initial_parameters_equal=True,seed_reset_per_arm=True,fresh_optimizer_per_arm=True,
                parameter_counts={name:3233 for name in m.LEARNED_METHODS},losses={arm:[.5]*1000 for arm in ('p1_only','paired')}))
    def patches(self,m,c,events,*,novel_fail=False,baseline_bad=False,mutate=False):
        from contextlib import ExitStack
        stack=ExitStack();roles={name:self.role(name) for name in ('L','OE','V','novel')}
        if baseline_bad:roles['V']['baseline_closed'][:,1]=0
        def role(path,name,*args):
            events.append(('role',name))
            if name=='V':self.assertIn(('fit',),events)
            if name=='novel':
                self.assertTrue((Path(c['output_root'])/'calibration/complete.json').is_file())
                if novel_fail:raise ValueError('novel missing')
            return roles[name]
        def encode(reader,x,device):
            p1=int(np.all(x[:,1,255]==1));y=x[:,0,0].astype(int)
            return {'fused160':np.full((len(x),1),p1,np.float32)},np.eye(6,dtype=np.float32)[(y+p1)%6]*(5 if p1 else 50)
        def descriptor(b,logits,g):return np.tile(b['fused160'],(1,100))
        def fit(a,b,kd,od,model_seed):
            events.append(('fit',));self.assertTrue(np.all(a[:,:100]==1));self.assertTrue(np.all(a[:,100:]==0))
            return self.state(m)
        def score(pair,energy,state):
            if mutate:state['models'][m.LEARNED_METHODS[0]]['bias2'][0]+=1
            return np.column_stack((energy,np.ones((len(pair),4))))
        def calibrate(ids,y,closed,baseline,scores,rx,scene):
            events.append(('calibrate',));np.testing.assert_array_equal(closed,baseline)
            return np.zeros(5),dict(constraints_passed=True,group_count=42,known_only=True,unknown_inputs_used=False,
                methods={name:dict(threshold=0.,constraints_passed=True) for name in m.METHODS})
        patches=[patch.object(m,'COUNTS',dict(L=30,OE=30,V=180,novel=30)),
            patch.object(m,'_backend',return_value=({'reader':object(),'dura':None,'reader_identity':roles['L']['manifest']['parent_F0_identity']},{})),
            patch.object(m,'_snapshot',return_value=[]),patch.object(m,'_check_models'),
            patch.object(m,'_smoke',side_effect=lambda *args:events.append(('smoke',)) or dict(status='PASS')),
            patch.object(m,'_role',side_effect=role),patch.object(m,'close_role'),patch.object(m,'_encode',side_effect=encode),
            patch.object(m,'_branches',side_effect=lambda reader,x,device,batch:encode(reader,x,device)),
            patch.object(m.geometry_core,'fit_geometry',return_value=dict(value=np.array([1.]))),
            patch.object(m.geometry_core,'descriptor',side_effect=descriptor),patch.object(m.core,'fit_readouts',side_effect=fit),
            patch.object(m.core,'score_compact',side_effect=score),patch.object(m.core,'calibrate_known',side_effect=calibrate),
            patch.object(m,'_save_geometry',side_effect=lambda p,g:m._npz(p,dict(value=g['value'])))]
        for p in patches:stack.enter_context(p)
        return stack

    def test_fixed_config_and_private_input_negatives(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td));self.assertIs(m.validate_config(c),c)
            for key in ('truth','unknown_count','checkpoint','target_root'):
                with self.subTest(key=key),self.assertRaises(ValueError):m.validate_config(dict(c,**{key:'x'}))
            bad=copy.deepcopy(c);bad['source_config']['checkpoint']='oldE51'
            with self.assertRaises(ValueError):m.validate_config(bad)
            bad=copy.deepcopy(c);bad['source_config']['row_id']='star-consensus-s2026092701'
            with self.assertRaises(ValueError):m.validate_config(bad)

    def test_pair_order_P1_identity_energy_and_all_freeze_before_novel(self):
        m=self.module();events=[]
        with tempfile.TemporaryDirectory() as td:
            c=self.config(Path(td));writes=[];real=m._write
            with self.patches(m,c,events),patch.object(m,'_write',side_effect=lambda p,v:writes.append(str(p)) or real(p,v)):
                result=m.run(c,'cpu')
            self.assertEqual(events[0],('smoke',));self.assertEqual([x[1] for x in events if x[0]=='role'],['L','OE','V','novel'])
            self.assertEqual(events.count(('fit',)),1);self.assertEqual(events.count(('calibrate',)),1)
            self.assertEqual(result['classification_path'],'P1');self.assertEqual(result['readout_optimizer_updates_total'],2000)
            self.assertTrue(result['control_p0_zeroed']);self.assertEqual(writes[-1],str(Path(c['output_root'])/'complete.json'))
            with np.load(Path(c['output_root'])/'predict/known_predictions.npz') as z:
                self.assertEqual(set(z.files),m.KNOWN_FIELDS);self.assertEqual(z['scores'].shape,(180,5));self.assertEqual(z['closed'].ndim,1)
                np.testing.assert_array_equal(z['closed'],self.role('V')['baseline_closed'][:,1]);np.testing.assert_array_equal(z['baseline_closed'],z['closed'])
                self.assertTrue(np.all(z['scores'][:,0]<10))  # P0 energy would be about 50.
            with np.load(Path(c['output_root'])/'predict/novel_predictions.npz') as z:self.assertEqual(set(z.files),m.NOVEL_FIELDS)
            self.assertTrue((Path(c['output_root'])/'geometry/P0.npz').exists());self.assertTrue((Path(c['output_root'])/'geometry/P1.npz').exists())
            with self.assertRaises(FileExistsError):m.run(c,'cpu')

    def test_failure_or_aggregate_mutation_never_marks_complete(self):
        m=self.module()
        for defect in ('novel','baseline','mutation'):
            with self.subTest(defect=defect),tempfile.TemporaryDirectory() as td:
                c=self.config(Path(td));events=[]
                with self.patches(m,c,events,novel_fail=defect=='novel',baseline_bad=defect=='baseline',mutate=defect=='mutation'),self.assertRaises(ValueError):
                    m.run(c,'cpu')
                self.assertFalse((Path(c['output_root'])/'complete.json').exists())
                if defect!='novel':self.assertNotIn(('role','novel'),events)

    def test_codec_roundtrip_and_rejects_dim_dtype_missing_and_object_fields(self):
        m=self.module();state=self.state(m)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);m._save_readouts(root/'readouts.npz',state);m._write(root/'readouts.json',dict(training=state['training']))
            loaded=m.load_readouts(root/'readouts.npz',root/'readouts.json');self.assertTrue(m.previous.previous._same(state,loaded))
            with np.load(root/'readouts.npz',allow_pickle=False) as z:arrays={key:z[key].copy() for key in z.files}
            for defect in ('dimension','dtype','missing','object'):
                data={k:v.copy() for k,v in arrays.items()}
                if defect=='dimension':data['pair_dim']=np.asarray(100,np.int64)
                elif defect=='dtype':data['model__'+m.LEARNED_METHODS[0]+'__weight1']=np.zeros((16,200),np.float64)
                elif defect=='missing':data.pop('normalizer__paired__scale')
                else:data['hidden']=np.array([{}],dtype=object)
                bad=root/(defect+'.npz');np.savez(bad,**data)
                with self.subTest(defect=defect),self.assertRaises(ValueError):m.load_readouts(bad,root/'readouts.json')

    def test_actual_consensus200_core_fixed_budget_and_snapshot_codec(self):
        m=self.module();rng=np.random.default_rng(35);n=30;y=np.arange(n)%6;domains=np.arange(n)%15
        branches={key:rng.normal(0,.1,(n,d)).astype(np.float32) for key,d in m.previous.DIMENSIONS.items()}
        for values in branches.values():values[np.arange(n),y]+=2
        branches['reference29'][:,27]=0;logits=np.eye(6,dtype=np.float32)[y]*4
        geometry=m.geometry_core.fit_geometry(branches,y);desc=m.geometry_core.descriptor(branches,logits,geometry)
        pair=m.core.pair_descriptors(desc,desc*.9)
        state=m.core.fit_readouts(pair,pair+rng.normal(0,.1,pair.shape).astype(np.float32),domains,domains,model_seed=2026092701)
        self.assertEqual(state['training']['total_optimizer_updates'],2000)
        self.assertEqual(state['training']['checkpoint_steps'],[200,1000])
        expected=m.core.score_readouts(pair,state)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);m._save_readouts(root/'readouts.npz',state);m._write(root/'readouts.json',dict(training=state['training']))
            loaded=m.load_readouts(root/'readouts.npz',root/'readouts.json')
            np.testing.assert_array_equal(expected,m.core.score_readouts(pair,loaded))
        self.assertEqual(expected.shape,(n,4))

    def test_budget_rejects_extra_fit_missing_steps_or_wrong_snapshot_origin(self):
        m=self.module();state=self.state(m)
        with patch.object(m,'COUNTS',dict(L=30,OE=30,V=180,novel=30)):
            m._training_complete(state)
            for defect in ('updates','snapshot','trace','initialization'):
                bad=copy.deepcopy(state)
                if defect=='updates':bad['training']['total_optimizer_updates']=2400
                elif defect=='snapshot':bad['training']['checkpoint_updates'][m.LEARNED_METHODS[0]]=1000
                elif defect=='trace':bad['training']['losses']['paired']=bad['training']['losses']['paired'][:-1]
                else:bad['training']['initial_parameters_equal']=False
                with self.subTest(defect=defect),self.assertRaises(ValueError):m._training_complete(bad)


if __name__=='__main__':unittest.main()
