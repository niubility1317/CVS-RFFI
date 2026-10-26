import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from experiments.pamf_channel_pl import u_realization_cache as m


def context():
    return dict(project='/project',python='python',frozen_sources={},G=dict(kind='original_F0_LU',roles='LU'),
        cases=[dict(row_id='F0-ordinary_PL-s'+str(s),protocol='F0',model_seed=s,parent_source='/original/segment20',
            source_config=dict(ablation='F21',num_classes=6),expected_source_contract='/source.json') for s in (2026092701,2026092702,2026092703)])


class URealizationCacheTests(unittest.TestCase):
    def config(self,replica=1):
        ctx=context()
        return dict(run_id=m.RUN,source_context=ctx,frozen_sources=ctx['frozen_sources'],mode='cache',replica=replica,
            output_root='/project/runs/'+m.RUN+'/fresh'+str(replica),code_commit='commit')

    def test_foreign_handles_and_undeclared_realization_rejected(self):
        c=self.config();m.validate_config(c)
        for change in (dict(target_inputs='query'),dict(F_checkpoint='classifier'),dict(output_root='/foreign'),
                       dict(mode='score'),dict(replica=True),dict(replica=0),dict(replica=4)):
            with self.subTest(change=change),self.assertRaises(ValueError):m.validate_config(dict(c,**change))
        for r in (False,0,4,'1'):
            with self.assertRaises(ValueError):m.namespace(r)
        self.assertEqual(len({m.namespace(r) for r in (1,2,3)}),3)
        foreign=copy.deepcopy(c);foreign['source_context']['target_truth']='truth'
        with self.assertRaises(ValueError):m.validate_config(foreign)

    def test_hidden_U_and_source_sessions_required(self):
        contract=dict(source_rxs=[1,3,4,6,8],source_days=[1,2,3])
        meta=dict(rx_i=torch.tensor([1,3]),day_i=torch.tensor([1,2]),eq_i=torch.ones(2,dtype=torch.long),
            tx_label_visible=torch.zeros(2,dtype=torch.bool))
        b=[torch.zeros(2,2,256),torch.tensor([-1,-1]),torch.zeros(2),meta];m.hidden(b,contract)
        visible=copy.deepcopy(b);visible[1][0]=0
        with self.assertRaises(ValueError):m.hidden(visible,contract)
        for key,value in [('tx_label_visible',True),('rx_i',2),('day_i',4),('eq_i',2)]:
            wrong=copy.deepcopy(b);wrong[3][key][0]=value
            with self.subTest(key=key),self.assertRaises(ValueError):m.hidden(wrong,contract)

    def test_shared_smoke_precedes_each_immutable_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ctx=context();ctx['project']=root.as_posix();path=root/'context.json';path.write_text(json.dumps(ctx),encoding='utf-8')
            run=root/'runs'/m.RUN;run.mkdir(parents=True)
            rt=dict(run_root=run.as_posix(),context=str(path),frozen_sources={},code_commit='commit',release_root='/release',python='python')
            jobs=m.prepare(rt)
            self.assertEqual([j['id'] for j in jobs],['smoke','fresh1','fresh2','fresh3'])
            self.assertEqual(jobs[0]['dependencies'],[])
            for j in jobs[1:]:
                self.assertEqual(j['dependencies'],[str(run/'smoke.json')]);self.assertEqual(j['gpu_pool'],list(range(8)))
                cfg=json.loads(Path(j['config']).read_text(encoding='utf-8'));self.assertEqual(Path(cfg['output_root']),run/j['id'])
            with self.assertRaises(FileExistsError):m.prepare(rt)


if __name__=='__main__':unittest.main()
