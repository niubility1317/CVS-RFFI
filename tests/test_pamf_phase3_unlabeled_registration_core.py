from copy import deepcopy
import importlib
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np


class RegistrationCoreTests(unittest.TestCase):
    def module(self):
        name='experiments.pamf_phase3_open_world.unlabeled_registration_core'
        self.assertTrue((Path(__file__).resolve().parents[1]/(name.replace('.','/')+'.py')).is_file())
        return importlib.import_module(name)

    def fixture(self,labels=None,shift=.4):
        module=self.module()
        labels=np.repeat(np.arange(3),8) if labels is None else np.asarray(labels)
        geometry=dict(schema='phase3_source_open_geometry_v1',class_count=6,branches={});branches={}
        for name,d in module.DIMENSIONS.items():
            mean=np.zeros(d) if name=='fused160' else np.linspace(-.2,.2,d)
            scale=np.ones(d) if name=='fused160' else np.linspace(.8,1.2,d)
            proto=np.eye(6,d)
            if name=='reference29':mean[27]=0.;scale[27]=1.
            geometry['branches'][name]=dict(mean=mean,scale=scale,prototypes=proto,whitener=np.eye(d))
            unit=proto[labels].copy();unit[:,-1]+=shift
            unit/=np.maximum(np.linalg.norm(unit,axis=1,keepdims=True),1e-12)
            raw=mean+scale*unit*(1+np.arange(len(labels))[:,None]/50)
            branches[name]=raw.astype(np.float32)
        logits=np.zeros((len(labels),6),np.float64);logits[np.arange(len(labels)),labels]=10.
        return branches,geometry,logits

    def density_fixture(self,count=8):
        groups=[(y,r,d) for y in range(6) for r in (1,3,4,6,8) for d in (1,2,3)]
        keys=np.repeat(np.asarray(groups,np.int64),count,axis=0)
        z=np.column_stack((np.tile(np.arange(count),90),np.zeros(90*count))).astype(np.float64)
        angles=np.tile(np.linspace(-.2,.2,count),90)
        unit=np.zeros((len(z),160));unit[:,0]=np.cos(angles);unit[:,1]=np.sin(angles)
        return z,unit,keys[:,0],keys[:,1],keys[:,2]

    def discover_fixture(self):
        sizes=(16,20,15,2,2);n=sum(sizes)
        z=np.concatenate([np.tile([x,0.],(count,1)) for x,count in zip((0.,3.,6.,9.,0.),sizes)])
        z[51:53,0]=[9.,12.]
        unit=np.zeros((n,160));start=0
        for col,count in zip((6,7,8,9,0),sizes):unit[start:start+count,col]=1.;start+=count
        q=np.full(n,-1.);q[-2:]=0.
        ids=np.asarray([prefix+str(i).zfill(3) for prefix,count in zip(('z','a','m','n','k'),sizes) for i in range(count)])
        anchors=np.eye(2,160);anchor_q=np.zeros(2)
        return z,unit,q,ids,anchors,anchor_q

    def test_embedding_is_equal_weight_source_standardized_whitened_688(self):
        module=self.module();b,g,l=self.fixture();before=deepcopy((b,g))
        name='time160';g['branches'][name]['whitener']=np.diag(np.linspace(.5,2.,160))
        result=module.embedding(b,g);expected=[]
        for key in module.EMBEDDING_BRANCHES:
            a=g['branches'][key];v=(b[key].astype(np.float64)-a['mean'])/a['scale']
            v=v@a['whitener'].T;v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-12)
            expected.append(v/np.sqrt(5))
        self.assertEqual((result.shape,result.dtype),((24,688),np.dtype('float32')))
        np.testing.assert_allclose(result,np.concatenate(expected,axis=1),atol=2e-8)
        np.testing.assert_allclose(np.linalg.norm(result,axis=1),1.,atol=1e-6)
        for key in b:self.assertTrue(np.array_equal(b[key],before[0][key]))
        self.assertIn('pa_moments48',module.EMBEDDING_BRANCHES)

    def test_bias_qualified_core_bound_direction_mask_and_no_mutation(self):
        module=self.module();b,g,l=self.fixture();before=deepcopy((b,g,l))
        bias,diag=module.fit_bias(b,l,l.copy(),g)
        self.assertTrue(diag['applied']);self.assertTrue(diag['core_mask'].all())
        self.assertEqual(diag['support_counts'].tolist(),[8,8,8,0,0,0])
        self.assertEqual(diag['qualified_classes'].tolist(),[0,1,2])
        for name,value in bias.items():
            self.assertEqual(value.dtype,np.float64);self.assertLessEqual(np.linalg.norm(value),.1+1e-12)
            self.assertLess(value[-1],0.)
        self.assertEqual(bias['reference29'][27],0.)
        adapted=module.apply_bias(b,g,bias)
        for name in b:
            a=g['branches'][name]
            r0=np.linalg.norm((b[name]-a['mean'])/a['scale'],axis=1)
            r1=np.linalg.norm((adapted[name]-a['mean'])/a['scale'],axis=1)
            np.testing.assert_allclose(r0,r1,rtol=2e-6,atol=2e-7)
            self.assertEqual(adapted[name].dtype,np.float32)
            self.assertTrue(np.array_equal(b[name],before[0][name]))
            for key in a:self.assertTrue(np.array_equal(a[key],before[1]['branches'][name][key]))
        self.assertTrue(np.all(adapted['reference29'][:,27]==0.))
        self.assertTrue(np.array_equal(l,before[2]))
        self.assertNotIn('y',inspect.signature(module.fit_bias).parameters)

    def test_core_minimum_classes_support_two_heads_and_votes_identity(self):
        module=self.module()
        for labels,change in ((np.repeat(np.arange(3),7),'none'),(np.repeat(np.arange(2),8),'none'),
                              (np.repeat(np.arange(3),8),'head'),(np.repeat(np.arange(3),8),'confidence'),
                              (np.repeat(np.arange(3),8),'votes'),(np.repeat(np.arange(3),8),'fused_tie')):
            with self.subTest(change=change,rows=len(labels)):
                b,g,l=self.fixture(labels);f0=l.copy()
                if change=='head':f0=np.roll(f0,1,axis=1)
                elif change=='confidence':f0[:]=0.
                elif change=='votes':
                    for name in ('time160','frequency160'):
                        a=g['branches'][name];b[name]=(a['mean']+a['scale']*a['prototypes'][(labels+1)%6]).astype(np.float32)
                elif change=='fused_tie':
                    b['fused160'][:]=0.;b['fused160'][:,0:2]=1.
                bias,diag=module.fit_bias(b,l,f0,g)
                self.assertFalse(diag['applied'])
                self.assertTrue(all(np.count_nonzero(v)==0 for v in bias.values()))
                after=module.apply_bias(b,g,bias)
                for name in b:self.assertTrue(np.array_equal(after[name],b[name]));self.assertIsNot(after[name],b[name])

    def test_source_density_rank_seven_excludes_self_and_LOO_centers(self):
        module=self.module();z,f,y,rx,day=self.density_fixture();before=deepcopy((z,f,y,rx,day))
        result=module.source_density(z,f,y,rx,day)
        self.assertAlmostEqual(result['eps'],7.)
        self.assertEqual(result['anchor_units'].shape,(90,160));self.assertEqual(result['anchor_units'].dtype,np.float32)
        self.assertEqual(result['group_keys'].tolist(),[[a,b,c] for a in range(6) for b in (1,3,4,6,8) for c in (1,2,3)])
        distances=[]
        for index,rows in enumerate(result['group_indices']):
            self.assertEqual(len(rows),8)
            np.testing.assert_array_equal(np.column_stack((y[rows],rx[rows],day[rows])),np.tile(result['group_keys'][index],(8,1)))
            for row in rows:
                center=f[rows[rows!=row]].mean(axis=0);center/=np.linalg.norm(center)
                distances.append(1-np.clip(f[row]@center,-1.,1.))
        self.assertAlmostEqual(result['radius'],np.quantile(distances,.95),places=12)
        for a,b in zip((z,f,y,rx,day),before):self.assertTrue(np.array_equal(a,b))
        self.assertEqual(result['diagnostics']['neighbor_rank'],7)

    def test_source_density_rejects_bad_source_labels_domains_and_small_groups(self):
        module=self.module();z,f,y,rx,day=self.density_fixture()
        for key,bad in (('y',np.full(len(y),6,np.int64)),('rx',np.full(len(y),26,np.int64)),
                        ('day',np.full(len(y),4,np.int64))):
            data=dict(y=y,rx=rx,day=day);data[key]=bad
            with self.subTest(key=key),self.assertRaises(ValueError):module.source_density(z,f,**data)
        with self.assertRaises(ValueError):module.source_density(*self.density_fixture(7))
        floor=module.source_density(np.zeros_like(z),np.tile(np.eye(1,160),(len(z),1)),y,rx,day)
        self.assertEqual(floor['eps'],1e-6);self.assertEqual(floor['radius'],1e-6)

    def test_discovery_estimates_K_strict_threshold_min16_and_stable_anonymous_order(self):
        module=self.module();z,f,q,ids,a,aq=self.discover_fixture();before=deepcopy((z,f,q,ids,a,aq))
        result=module.discover(z,f,q,0.,ids,.1,.2,a,aq,'synthetic')
        self.assertEqual(result['anonymous_ids'].tolist(),['synthetic-anon-0001','synthetic-anon-0002'])
        self.assertEqual(result['prototypes'].shape,(2,160))
        np.testing.assert_array_equal(result['assignments'][:16],np.ones(16,np.int64))
        np.testing.assert_array_equal(result['assignments'][16:36],np.zeros(20,np.int64))
        self.assertTrue(np.all(result['assignments'][36:]==-1))
        self.assertAlmostEqual(result['penalty'],.05)
        permutation=np.random.default_rng(7).permutation(len(ids))
        shuffled=module.discover(z[permutation],f[permutation],q[permutation],0.,ids[permutation],.1,.2,a,aq,'synthetic')
        np.testing.assert_array_equal(shuffled['assignments'],result['assignments'][permutation])
        np.testing.assert_allclose(shuffled['prototypes'],result['prototypes'])
        for x,y in zip((z,f,q,ids,a,aq),before):self.assertTrue(np.array_equal(x,y))

    def test_zero_cluster_is_valid_and_duplicate_ID_cannot_meet_independent_minimum(self):
        module=self.module();z,f,q,ids,a,aq=self.discover_fixture()
        empty=module.discover(z,f,np.zeros_like(q),0.,ids,.1,.2,a,aq,'empty')
        self.assertEqual(empty['prototypes'].shape,(0,160));self.assertEqual(empty['anonymous_ids'].shape,(0,))
        self.assertTrue(np.all(empty['assignments']==-1));self.assertEqual(empty['diagnostics']['candidate_count'],0)
        small=module.discover(z[:15],f[:15],q[:15],0.,ids[:15],.1,.2,a,aq,'small')
        self.assertEqual(len(small['prototypes']),0);self.assertTrue(np.all(small['assignments']==-1))
        duplicate=ids.copy();duplicate[1]=duplicate[0]
        with self.assertRaises(ValueError):module.discover(z,f,q,0.,duplicate,.1,.2,a,aq,'duplicate')

    def test_penalty_protects_all_source_anchors_at_old_margin_scale(self):
        module=self.module();z,f,q,ids,a,aq=self.discover_fixture()
        a=f[[0,16]].copy();aq=np.asarray([.2,-.1])
        result=module.discover(z,f,q,0.,ids,.1,.2,a,aq,'anchors')
        distances=1-np.clip(a@result['prototypes'].T,-1.,1.)
        margins=1-distances/.2-result['penalty']
        self.assertTrue(np.all(margins<=aq[:,None]-.05+1e-12))

    def test_joint_all_classes_REG0_exact_zero_and_ties_accept_old_first(self):
        module=self.module();f=np.zeros((5,160));f[np.arange(5),[6,6,2,9,7]]=1.
        logits=np.zeros((5,6));logits[:,2]=3.;q=np.asarray([.1,-.1,.1,-.1,0.])
        proto=np.eye(8,160)[6:8].astype(np.float32)
        margins=module.joint_margins(logits,f,q,0.,1.,proto,.2,.05)
        self.assertEqual((margins.shape,margins.dtype),((5,8),np.dtype('float64')))
        np.testing.assert_allclose(margins[:,:6],(logits-logits.max(axis=1,keepdims=True))/30+q[:,None])
        np.testing.assert_array_equal(module.joint_predict(logits,f,q,0.,1.,proto,.2,.05),[6,6,2,-1,7])
        empty=np.empty((0,160),np.float32)
        np.testing.assert_array_equal(module.joint_predict(logits,f,q,0.,1.,empty,.2,.05),[2,-1,2,-1,2])
        tie=module.joint_predict(np.zeros((1,6)),f[:1],np.zeros(1),0.,1.,proto[:1],.2,1.)
        self.assertEqual(tie.tolist(),[0])
        fake=np.asarray([[0.,0.,0.,0.,0.,0.,0.],[-1.,-1.,-1.,-1.,-1.,-1.,0.]])
        with patch.object(module,'joint_margins',return_value=fake) as only:
            self.assertEqual(module.joint_predict(None,None,None,None,None,None,None,None).tolist(),[0,6])
            self.assertEqual(only.call_count,1)

    def test_nan_bad_shapes_and_invalid_frozen_scalars_are_rejected(self):
        module=self.module();b,g,l=self.fixture();bad=deepcopy(b);bad['fused160'][0,0]=np.nan
        with self.assertRaises(ValueError):module.embedding(bad,g)
        bad=deepcopy(b);bad['reference29'][0,27]=1.
        with self.assertRaises(ValueError):module.fit_bias(bad,l,l,g)
        with self.assertRaises(ValueError):module.fit_bias(b,l[:,:5],l,g)
        badbias={name:np.zeros(d) for name,d in module.DIMENSIONS.items()};badbias['time160'][0]=np.nan
        with self.assertRaises(ValueError):module.apply_bias(b,g,badbias)
        z,f,q,ids,a,aq=self.discover_fixture()
        for eps,radius in ((0.,.2),(.1,0.),(np.nan,.2)):
            with self.subTest(eps=eps,radius=radius),self.assertRaises(ValueError):module.discover(z,f,q,0.,ids,eps,radius,a,aq,'bad')
        with self.assertRaises(ValueError):module.discover(z,f,q[:-1],0.,ids,.1,.2,a,aq,'bad')
        for sigma in (0.,-1.,np.nan):
            with self.subTest(sigma=sigma),self.assertRaises(ValueError):module.joint_margins(np.zeros((len(f),6)),f,q,0.,sigma,np.empty((0,160)),.2,.05)


if __name__=='__main__':unittest.main()
