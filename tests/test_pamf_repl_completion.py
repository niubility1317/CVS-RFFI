from copy import deepcopy
import unittest
from experiments.pamf_repl_completion.plan import make_plan,validate

def fixture_context():
    from experiments.pamf_repl_completion.plan import PROTOCOLS,SEEDS
    ctx=dict(handles={},protocols={},templates={},sources={},caches={},G_components=[],G_crossed=[dict(seed=s,spec={}) for s in (2026092702,2026092703)],runtime=dict(augmentation_cache='/aug'))
    for pid in PROTOCOLS:
        ctx['handles'][pid]=dict(source_cache='/cache/'+pid,source_contract='/contract/'+pid,classes=list(range(10 if pid=='ManyRx' else 6)),
            tests={t:dict(views_root='/views/'+pid+'/'+t,truth='/truth/'+pid+'/'+t) for t in ('cross_date','cross_rx','cross_both')})
        ctx['protocols'][pid]=dict(source_rxs=[1,3,4,6,8])
        ctx['caches'][pid]={r:dict(path='/restored/'+pid+'/'+r,G={}) for r in ('L','LU')}
    for seed in SEEDS:
        base=dict(model_seed=seed,source_cache='/cache/F0',source_contract='/contract/F0')
        selected=dict(hard_pl=dict(weight=6.,threshold=.95,method='HPLT1U'),hard_pl_no_uas=True,hard_pl_label_smoothing=.01,
            hard_pl_mixstyle=dict(probability_per_level=.5),hard_pl_shared_style_plan=True,hard_pl_gradient_guard={},
            hard_pl_joint_contrast=dict(weight=.05),hard_pl_receiver_extrapolation=dict(alpha=.25))
        ctx['templates'][str(seed)]=dict(base=dict(F21=base),repl=selected)
        ctx['sources'][str(seed)]={k:'/old/'+k+'/s'+str(seed) for k in ('F20','F21','REPL','NO_RXE')}
        for arm in range(7):ctx['G_components'].append(dict(seed=seed,arm='G'+str(arm),spec={}))
    return ctx

class MatrixTest(unittest.TestCase):
    def test_exact_authorized_scope_and_reuse(self):
        p=make_plan();self.assertEqual(201,len(p['rows']));self.assertEqual(189,sum(not r['reuse'] for r in p['rows']))
        self.assertEqual(84,p['shared_E20_prefixes']);self.assertEqual(1924740,p['new_unique_updates'])
        self.assertEqual([0,1,2,3,4,5,6,7],p['allowed_gpus'])
    def test_reject_cross_contract_reuse(self):
        p=make_plan();r=next(r for r in p['rows'] if r['protocol_id']=='ManyRx');r['reuse']=True
        with self.assertRaises(ValueError):validate(p)

    def test_all_source_configs_and_job_dependencies(self):
        import tempfile
        from unittest.mock import patch
        from experiments.pamf_repl_completion.plan import configuration
        from experiments.pamf_repl_completion import prepare
        ctx=fixture_context();matrix=make_plan()
        for row in matrix['rows']:configuration(ctx,row,'/project','commit')
        def read(p):
            p=str(p).replace('\\','/')
            if p=='CONTEXT':return ctx
            if p=='MATRIX':return dict(matrix=matrix)
            if p.startswith('/contract/'):return dict(contract=p)
            if p.endswith('/complete.json'):
                seed=next(s for s in ctx['sources'] if '/s'+s+'/' in p)
                return dict(config=dict(run_id='old',row_id='old-s'+seed,source_contract='/contract/F0'))
            raise AssertionError(p)
        written=[]
        with tempfile.TemporaryDirectory() as tmp,patch.object(prepare,'read',read),patch.object(prepare,'write',lambda path,data:written.append((str(path),data))):
            jobs=prepare.prepare(dict(context='CONTEXT',matrix='MATRIX',run_root=tmp,project='/project',code_commit='commit',frozen_sources={}))
        from collections import Counter
        self.assertEqual(dict(prefix=84,F=189,predict=228,score=228,timing=4),dict(Counter(j['kind'] for j in jobs)))
        self.assertEqual(len(jobs),len({j['id'] for j in jobs}))
        checked=0
        for path,c in written:
            path=path.replace('\\','/')
            if '/score__' in path or '/predict__' in path:
                self.assertEqual([1,2,3],c['source_days']);checked+=1
        self.assertEqual(456,checked)

class PartnerTest(unittest.TestCase):
    def test_actual_loader_mode_preserves_256_U_and_true_supervision(self):
        from experiments.pamf_repl_completion.objectives import loader_config
        self.assertEqual('native_cric',loader_config(dict(U_use=True,regime='matched_sat_plain_pl'))['regime'])
        self.assertEqual('matched_sat_supervised',loader_config(dict(U_use=False,regime='native_cric'))['regime'])
    def test_loader_scope_uses_actual_256_hidden_U_with_partial_batches(self):
        import torch
        from unittest.mock import patch
        from experiments.cric_dura_comparison import f_data,f_train
        from experiments.pamf_repl_completion.objectives import loader_scope
        contract=dict(source_rxs=[1],source_days=[1,2,3],classes=list(range(6)),counts=dict(L_s=1280,U_s=520,V=16))
        opened=[]
        def dataset(root,role,expected):
            opened.append(role);n=contract['counts'][role]
            return torch.utils.data.TensorDataset(torch.arange(n).reshape(-1,1).float(),torch.full((n,),-1 if role=='U_s' else 0),torch.zeros(n,dtype=torch.long))
        c=dict(source_contract='x',source_cache='x',source_rxs=[1],source_days=[1,2,3],model_seed=2026092701,U_use=True,regime='matched_sat_plain_pl',ablation='F21')
        with patch.object(f_data,'read',return_value=contract),patch.object(f_data,'SourceDataset',dataset),loader_scope():
            ctx=f_train.context(c);u=ctx['unlabeled_loader'];self.assertEqual(256,u.batch_size)
            it=iter(u);b=[next(it) for _ in range(3)];self.assertEqual([256,256,8],[len(x[0]) for x in b]);self.assertTrue(all(bool((x[1]==-1).all()) for x in b))
            opened.clear();ctx=f_train.context(dict(c,U_use=False,ablation='F20'))
            self.assertIsNone(ctx['unlabeled_loader']);self.assertEqual(['L_s','V'],opened)
    def test_manyrx_class_9_and_no_partner_fallback(self):
        import torch
        from experiments.pamf_repl_completion.objectives import source_partners
        p=torch.tensor([9,9,3]);a=torch.tensor([True,True,True]);rx=torch.tensor([1,2,1])
        ix,valid=source_partners(p,a,rx,[1,2],10,torch.float32)
        self.assertEqual([True,True,False],valid.tolist());self.assertEqual(2,int(ix[2]))
    def test_hidden_source_receivers_only(self):
        import torch
        from experiments.pamf_repl_completion.objectives import source_partners
        with self.assertRaises(ValueError):source_partners(torch.tensor([0]),torch.tensor([True]),torch.tensor([99]),[1],6,torch.float32)
    def test_permissions_are_not_confounded_as_L_only(self):
        p=make_plan();r=next(r for r in p['rows'] if r['method']=='REPL');r['G_roles']='L'
        with self.assertRaises(ValueError):validate(p)

class OfflineStatisticsTest(unittest.TestCase):
    def test_core_effects_and_corrupt_net_rejection(self):
        import csv,tempfile
        from pathlib import Path
        from itertools import product
        from experiments.pamf_repl_completion.analyze import aggregate
        from experiments.pamf_repl_completion.plan import SEEDS
        records=[]
        views=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
        for bits,seed in product((''.join(x) for x in product('01',repeat=3)),SEEDS):
            base=500+sum(int(x)*w for x,w in zip(bits,(10,20,30)))+SEEDS.index(seed)
            method='REPL' if bits=='111' else 'supervised' if bits=='110' else 'ABC'+bits
            rec=dict(row=dict(row_id=method+str(seed),protocol_id='F0',group='synthetic',method=method,model_seed=seed,classes=6),
                score=dict(model_seed=seed,truth_last=True,target_feedback_forbidden=True,results=[],paired_raw=[]))
            for task,route,view in product(('cross_date','cross_rx','cross_both'),('raw','DURA-L','DURA-LU'),views):
                gain={'raw':0,'DURA-L':50,'DURA-LU':70}[route];n=base+gain
                cm=[[0]*6 for _ in range(6)];cm[0][0]=n;cm[0][1]=1000-n
                row=dict(test=task,readout=route,view=view,dimension='overall',stratum='all',count=1000,confusion=cm,accuracy=n/1000,macro_f1=(2*n/(1000+n))/6)
                rec['score']['results'].append(row)
                if route!='raw':rec['score']['paired_raw'].append(dict(test=task,readout=route,view=view,dimension='overall',stratum='all',count=1000,repairs=gain,damage=0,net=gain))
            records.append(rec)
        with tempfile.TemporaryDirectory() as tmp:
            aggregate(records,Path(tmp))
            with (Path(tmp)/'core_effects.csv').open() as stream:effects=list(csv.DictReader(stream))
            for name,value in [('A',1.),('B',2.),('C',3.),('AB',0.),('AC',0.),('BC',0.),('ABC',0.)]:
                selected=[r for r in effects if r['effect']==name and r['metric']=='accuracy']
                self.assertTrue(selected);self.assertTrue(all(abs(float(r['mean'])-value)<1e-10 for r in selected))
        records[0]['score']['paired_raw'][0]['net']+=1
        with tempfile.TemporaryDirectory() as tmp,self.assertRaises(ValueError):aggregate(records,Path(tmp))

if __name__=='__main__':unittest.main()
