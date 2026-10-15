import copy,unittest
import torch
from experiments.cvs_urban_identity50 import design as d,model as model_api,features as joint
from . import branches as b,objective as o


class BranchChecks(unittest.TestCase):
    def model(self):
        m=model_api.build(d.config(d.rows()[0]),'cpu')
        with torch.no_grad():m.id_backbone.encoder.response_gain.fill_(.07)
        return m

    def test_actual_fusion_inputs_include_frequency_stats_and_full_PA(self):
        m=self.model().eval();encoder,core=b.layout(m);parts={};handles=[]
        for name,module in dict(Tproj=core.t_proj,Fproj=core.f_proj,Fstats=core.freq_stats_proj,
                Pproj=core.pa_proj,Pstats=core.pa_stats_proj).items():
            def hook(module,args,out,name=name):parts[name]=out
            handles.append(module.register_forward_hook(hook))
        try:
            out,v=b.forward(m,torch.randn(4,2,256))
            self.assertTrue(torch.equal(v['T'],parts['Tproj']))
            self.assertTrue(torch.equal(v['F'],parts['Fproj']+parts['Fstats']))
            self.assertFalse(torch.allclose(v['F'],parts['Fproj']))
            self.assertTrue(torch.allclose(v['P'],core.cls_head.pa_norm(parts['Pproj']+1.25*parts['Pstats']),atol=1e-6))
            self.assertTrue(torch.equal(v['J'],out['id_feat_joint']))
        finally:
            for handle in handles:handle.remove()

    def test_student_graph_and_selective_upstream_gradients(self):
        for key in (*b.BRANCHES,'J'):
            m=self.model().train();teacher=copy.deepcopy(m).eval().requires_grad_(False)
            x=torch.randn(4,2,256);logits,targets=b.frozen_teacher(teacher,x)
            out,v=b.forward(m,.6*x+.2*torch.randn_like(x));loss=o.direction(v[key],targets[key],torch.ones(4))
            encoder,core=b.layout(m);groups=dict(T=list(core.t2.parameters()),
                F=list(core.f_proj.parameters())+list(core.freq_stats_proj.parameters()),
                P=list(core.pa_proj.parameters())+list(core.pa_stats_proj.parameters()),
                R=list(encoder.response_projection.parameters()),S=list(encoder.sequence_pool.views[0].local_projection.parameters()))
            selected='T' if key=='J' else key
            params=groups[selected]+list(joint.boundary_parameters(m).values())
            if key=='S':params+=groups['T']
            gs=torch.autograd.grad(loss,params,allow_unused=True,retain_graph=True)
            own=gs[:len(groups[selected])]
            self.assertGreater(sum(float(g.square().sum()) for g in own if g is not None),0.,key)
            boundary=gs[len(groups[selected]):len(groups[selected])+4]
            self.assertTrue(all(g is None or float(g.abs().max())<1e-9 for g in boundary),key)
            if key=='S':self.assertGreater(sum(float(g.square().sum()) for g in gs[-len(groups['T']):] if g is not None),0.)
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            self.assertTrue(all(v[k].requires_grad for k in (*b.BRANCHES,'J')))
            torch.nn.functional.cross_entropy(out['tx_logits'],logits.argmax(1)).backward()
            self.assertGreater(float(joint.modules(m)[1].cls_head.weight.grad.norm()),0.)

    def test_replacement_changes_exactly_one_actual_branch(self):
        m=self.model().eval().requires_grad_(False);before={k:v.clone() for k,v in m.state_dict().items()}
        clean=torch.randn(4,2,256);urban=.5*clean+.5*torch.randn_like(clean)
        with torch.no_grad():
            _,cv=b.forward(m,clean);base,uv=b.forward(m,urban)
            for key in b.BRANCHES:
                with b.replace_one(m,cv,key):out,v=b.forward(m,urban)
                self.assertTrue(torch.equal(v[key],cv[key]),key)
                for other in (*b.BRANCHES,'_rho'):
                    if other!=key:self.assertTrue(torch.equal(v[other],uv[other]),(key,other))
                self.assertTrue(torch.equal(v['J'],out['id_feat_joint']))
            again,_=b.forward(m,urban)
            self.assertTrue(torch.equal(again['tx_logits'],base['tx_logits']))
        self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()))
        self.assertFalse(any(module._forward_hooks or module._forward_pre_hooks for module in m.modules()))

    def test_replacement_refuses_training_and_feature_graph_excludes_W(self):
        m=self.model().train();out,v=b.forward(m,torch.randn(3,2,256))
        theta=b.feature_parameters(m,out['id_feat_joint'])
        self.assertFalse(any(p is joint.modules(m)[1].cls_head.weight for _,p in theta))
        self.assertEqual(sum(p.numel() for _,p in theta),195041)
        with self.assertRaises(ValueError):
            with b.replace_one(m,{k:v[k].detach() for k in b.BRANCHES},'T'):pass

    def test_total_mixture_is_explicitly_normalized(self):
        values={k:torch.tensor(2.) for k in b.BRANCHES};coeff={k:.1 for k in b.BRANCHES}
        self.assertAlmostEqual(float(o.mixture(values,coeff,{k:.2 for k in b.BRANCHES})),.2)
        with self.assertRaises(ValueError):o.mixture(values,coeff,{k:1. for k in b.BRANCHES})


if __name__=='__main__':unittest.main()
