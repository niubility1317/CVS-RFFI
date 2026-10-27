import copy
import json
from pathlib import Path
import tempfile
import unittest
from experiments.cric_dura_comparison.io import read,write
from experiments.pamf_single_view_paper.plan import RUN,PROTOCOLS,make_plan,configuration
from experiments.pamf_single_view_paper.source import saved_segments
from experiments.pamf_single_view_paper.tail import TAIL,PREDECESSOR,materialize,predecessor_ready,validate_runtime

def context():
    return dict(handles={v[0]:dict(source_cache='source/'+k,source_contract='contract/'+k,classes=list(range(10 if k=='ten_class' else 6))) for k,v in PROTOCOLS.items()},
        protocols={v[0]:dict(source_rxs=[1,3,4,6,8]) for k,v in PROTOCOLS.items()},
        templates={str(s):dict(base=dict(F21={})) for s in (2026092701,2026092702,2026092703)},runtime=dict(augmentation_cache='source_cache'))

def runtime(project):
    q=project/'runs'/TAIL
    return dict(project=str(project),run_id=RUN,continuation_run_id=TAIL,original_root=str(project/'runs'/RUN),
                queue_root=str(q),run_root=str(q/'pipeline'),wait_for=str(project/'runs'/PREDECESSOR/'analysis/complete.json'),
                reuse_completed_jobs=True,max_own_per_GPU=1,max_total_per_GPU=2,code_commit='new-code')

def prefix(c):
    root=Path(c['output_root']);write(root/'resolved_config.json',c)
    f=root/'segment20';cfg=dict(c,output_root=str(f),segment_epochs=20,epoch_offset=0,step_offset=0,parent_source=None)
    write(f/'complete.json',dict(status='SOURCE_FROZEN',config=cfg,updates=4440,target_access=False,U_truth_read=False,head_fit_updates=0,seconds=1))
    (f/'final_ssdg.pth').touch()

class TailTests(unittest.TestCase):
    def test_waits_for_both_groups_and_own_fixed_root(self):
        with tempfile.TemporaryDirectory() as d:
            rt=runtime(Path(d));validate_runtime(rt);self.assertFalse(predecessor_ready(rt))
            write(rt['wait_for'],dict(status='ANALYZED',models=8))
            with self.assertRaises(ValueError):predecessor_ready(rt)
            write(rt['wait_for'],dict(status='ANALYZED',models=16),exclusive=False)
            self.assertTrue(predecessor_ready(rt))
            rt['wait_for']=str(Path(d)/'analysis_three_view/complete.json')
            with self.assertRaises(ValueError):validate_runtime(rt)

    def test_only_exact_own_saved_recipe_can_resume(self):
        with tempfile.TemporaryDirectory() as d:
            row=next(r for r in make_plan()['rows'] if not r['reuse_key'])
            c=configuration(context(),row,str(Path(d)),'old-code');prefix(c)
            resumed=dict(c,output_root=str(Path(d)/'new/source'),code_commit='new-code',resume_source_root=c['output_root'])
            self.assertEqual(len(saved_segments(resumed)),1)
            for key,value in [('model_seed',0),('plain_PL_threshold',.5),('source_contract','foreign')]:
                bad=dict(resumed,**{key:value})
                with self.subTest(key=key),self.assertRaises(ValueError):saved_segments(bad)
            bad=dict(resumed,resume_source_root=str(Path(d)/'foreign/source'))
            with self.assertRaises(ValueError):saved_segments(bad)

    def fixture(self,project):
        rt=runtime(project);old=Path(rt['original_root']);jobs=[];states={};sources={};train_n=0
        for index,row in enumerate(make_plan()['rows']):
            c=configuration(context(),row,str(project),'old-code');source=c['output_root']
            for kind in ([] if row['reuse_key'] else ['train'])+['predict','score']:
                jid=kind+'__'+row['row_id'];folder=Path(source) if kind=='train' else old/'readouts'/row['row_id']/('predictions' if kind=='predict' else 'scores')
                cfg=copy.deepcopy(c) if kind=='train' else dict(row_id=row['row_id'],paper_row=row,output_root=str(folder),source_output=source,prediction_output=str(old/'readouts'/row['row_id']/'predictions'))
                path=old/'configs'/(jid+'.json');write(path,cfg);marker=str(folder/'complete.json')
                status={'train':'SOURCE_FROZEN','predict':'ALL_PREDICTIONS_COMPLETE','score':'SCORED_COMPLETE'}[kind]
                deps=[] if kind=='train' else [source+'/complete.json'] if kind=='predict' else [cfg['prediction_output']+'/complete.json']
                complete=(train_n<65 if kind=='train' else index<36)
                jobs.append(dict(id=jid,kind=kind,config=str(path),completion=marker,successful_statuses=[status],dependencies=deps))
                states[jid]=dict(status='COMPLETE' if complete else 'QUEUED')
                if complete:write(marker,dict(status=status))
                if kind=='train':
                    if train_n==65:prefix(c)
                    train_n+=1
            sources[row['row_id']]=dict(row=row,source_output=source,predictions=str(old/'readouts'/row['row_id']/'predictions'),scores=str(old/'readouts'/row['row_id']/'scores'))
        write(old/'jobs.json',jobs);write(old/'sources.json',sources)
        write(old/'pipeline_state.json',dict(status='PAUSED_BY_USER_PRIORITY',pid=-999999,jobs=states))
        return rt,jobs

    def test_repoints_dependencies_and_reuses_completed_stages(self):
        with tempfile.TemporaryDirectory() as d:
            rt,old=self.fixture(Path(d));jobs,summary=materialize(rt)
            self.assertEqual(summary['remaining_stages'],dict(train=43,predict=84,score=84))
            self.assertEqual(summary['reused_stages'],dict(train=65,predict=36,score=36))
            self.assertEqual(len(summary['saved_segments']),1)
            for prior,j in zip(old,jobs):
                c=read(j['config'])
                if prior['completion']==j['completion']:
                    self.assertEqual(prior,j)
                elif j['kind']=='score':
                    self.assertEqual([Path(x) for x in j['dependencies']],[Path(c['prediction_output'])/'complete.json'])
                    self.assertTrue(c['prediction_output'].startswith(rt['run_root']))
                elif j['kind']=='predict':
                    self.assertEqual([Path(x) for x in j['dependencies']],[Path(c['source_output'])/'complete.json'])

    def test_missing_completed_artifact_blocks_duplicate_retraining(self):
        with tempfile.TemporaryDirectory() as d:
            rt,jobs=self.fixture(Path(d));Path(jobs[0]['completion']).unlink()
            with self.assertRaisesRegex(ValueError,'Previously completed'):materialize(rt,False)

if __name__=='__main__':unittest.main()
