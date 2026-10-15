import copy,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from . import design as d,provenance as p,evaluate as e

class FormalProtocol(unittest.TestCase):
    def row(self):return d.config(d.rows()[0])
    def screen(self,clean=450,none=400):
        def result(errors,ce):return dict(source_health=True,exports=dict(FINAL=dict(count=1260,errors=errors,mean_CE=ce)))
        rows=dict(ELEMENT=result(500,2.),NONE=result(none,1.),CLEAN_SIG=result(clean,.9))
        return dict(status='SOURCE_MATRIX_COMPLETE',actual_optimizer_steps=13320,target_access=False,results=rows,qualification=d.screen.qualify(rows))
    def evidence(self):
        c=self.row();path=Path(c['output_root']);placement=dict(mode=c['arm'],main_t2_dropout_p=0. if c['arm']=='NONE' else .1,
            signature_tap='original t2 output' if c['arm']=='NONE' else 't2.drop input after original pool',signature_gradient_detached=False,
            extra_IQ_forwards=0,extra_rng_draws=0,extra_parameters=0)
        report=dict(t2_noise_placement=placement)
        initial=dict(scratch_only=True,checkpoint_sources=[],ancestors=[],source_roles='EXACT_MATCH',target_contact=False,teacher_origin='own scratch student only',seed=c['model_seed'])
        freeze=dict(status='SOURCE_FROZEN',config=c,checkpoint=str(path/'final_ssdg.pth'),epoch=20,selection='final_only',target_access=False)
        done=dict(status='SOURCE_TRAINED',config=c,epoch=20,logged_steps=4440,optimizer_steps=4440,L_context_steps=4440,
            target_access=False,target_evaluated=False,checkpoint_sources=[],ancestors=[],initialization='scratch',baseline_recipe=c['training_recipe'],
            label_epochs=13,pseudo_epochs=7,native_fitted_probes_disabled=True,native_disabled_probe_calls=1,head_fit_updates=0,
            identity_input_policy=dict(explicit_CFO_identity_input=False),feature_extractor=report,mask_counts=dict(x=0),final_source_metrics=dict(source_val_count=27000))
        resolved=dict(config=c,native_fitted_probes_disabled=True,identity_input_policy=dict(explicit_CFO_identity_input=False),actual_feature_extractor=report)
        expected=dict(role_ids={'L_s':[[1,rx] for rx in (1,3,4,6,8)]},capsule_id='existing',split_id='existing')
        contract=dict(expected,native_role_comparison='EXACT_MATCH')
        return [c,done,initial,freeze,resolved,vars(d.make_args(c,'cpu')),contract,expected]
    def test_selection_requires_complete_matching_source(self):
        with patch.object(d,'ARMS',('NONE',)):
            self.assertEqual(d.source_selection(self.screen())['modes'],['NONE'])
            with self.assertRaises(ValueError):d.source_selection(dict(status='RUNNING'))
            with self.assertRaises(ValueError):d.source_selection(self.screen(clean=399))
        with patch.object(d,'ARMS',('NONE','CLEAN_SIG')):self.assertEqual(d.source_selection(self.screen(clean=399))['modes'],['NONE','CLEAN_SIG'])
        with self.assertRaises(ValueError):d.source_selection(self.screen(clean=550,none=501))
    def test_all_source_native_args_and_budget(self):
        for row in d.rows():
            c=d.config(row);a=d.make_args(c,'cpu');p.validate_args(vars(a),c)
            self.assertEqual(a.seed,row['model_seed']);self.assertEqual((a.epochs,a.label_epochs,a.pseudo_epochs),(20,13,7))
            self.assertEqual(a.baseline_ckpt,'');self.assertEqual(a.teacher_ckpt,'')
            self.assertIsNone(getattr(a,'source_fold_held_rx',None));self.assertEqual(a.wisig_train_rxs,'1,3,4,6,8')
    def test_native_fold_or_recipe_mismatch_rejected(self):
        c=self.row();a=vars(d.make_args(c,'cpu'))
        for key,value in (('source_fold_held_rx',1),('label_smoothing',.01),('wisig_train_rxs','3,4,6,8'),('epochs',40)):
            args=dict(a,**{key:value})
            with self.assertRaises(ValueError):p.validate_args(args,c)
    def test_source_provenance_rejects_partial_fold_and_ancestry(self):
        with patch.object(p.original,'report_validator'):
            self.assertEqual(p.validate_evidence(*self.evidence()),Path(self.row()['output_root']))
            for index,key,value in ((1,'optimizer_steps',4000),(1,'checkpoint_sources',['old-fold']),(2,'ancestors',['old-fold']),(3,'selection','target_best'),(1,'head_fit_updates',1)):
                fixture=self.evidence();fixture[index][key]=value
                with self.assertRaises(ValueError):p.validate_evidence(*fixture)
            fixture=self.evidence();fixture[6]['role_ids']={'L_s':[[1,3],[1,4],[1,6],[1,8]]}
            with self.assertRaises(ValueError):p.validate_evidence(*fixture)
    def test_bad_provenance_blocks_checkpoint_load(self):
        with patch.object(p,'source_proof',side_effect=ValueError('bad source')),patch('torch.load') as load:
            with self.assertRaises(ValueError):p.own_model(self.row(),'cpu')
            load.assert_not_called()
    def test_wrong_checkpoint_blocks_model_build(self):
        from . import model
        with patch.object(p,'source_proof',return_value=Path('own-source')),patch('torch.load',return_value=dict(epoch=20,run_id='other')),patch.object(model,'build') as build:
            with self.assertRaises(ValueError):p.own_model(self.row(),'cpu')
            build.assert_not_called()
    def test_incomplete_prediction_blocks_truth(self):
        from experiments.cvs_report_ssl import evaluate as truth_side
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES=''),patch.object(e,'preflight',side_effect=ValueError('incomplete')),patch.object(truth_side,'truth_arrays') as truth:
            with self.assertRaises(ValueError):e.score(d.rows()[0]['row_id'])
            truth.assert_not_called()
    def test_fresh_CPU_truth_process_required(self):
        with patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'),patch.object(e,'preflight') as preflight:
            with self.assertRaises(ValueError):e.score(d.rows()[0]['row_id'])
            with self.assertRaises(ValueError):e.finish()
            preflight.assert_not_called()
    def test_fixed_configuration_rejects_changed_readout_or_seed(self):
        for key,value in (('model_seed',2026092704),('readouts',['refitted']),('baseline_reuse',True)):
            c=copy.deepcopy(self.row());c[key]=value
            with self.assertRaises(ValueError):d.validate(c)
    def test_failed_first_seed_control_blocks_all_target_predictions(self):
        from . import source
        with tempfile.TemporaryDirectory() as root,patch.object(d,'BASE',Path(root)/'run'),patch.object(d,'ARMS',('NONE','CLEAN_SIG')):
            request=dict(arms=list(d.ARMS),rows=[d.config(row) for row in d.rows()])
            with patch.object(d,'read',return_value=request),patch.object(d,'source_selection',return_value={}),patch.object(d,'write'),patch.object(source,'train') as train,patch.object(e,'source_health',side_effect=[dict(qualified=True),dict(qualified=False)]),patch.object(e,'predict') as predict,patch.object(e.subprocess,'run') as score:
                e.run(Path('request.json'),Path('runner.py'))
                self.assertEqual(train.call_count,2);predict.assert_not_called();score.assert_not_called()
    def test_first_seed_cross_arm_health_precedes_prediction_and_scores_use_CPU(self):
        from . import source
        events=[]
        with tempfile.TemporaryDirectory() as root,patch.object(d,'BASE',Path(root)/'run'),patch.object(d,'ARMS',('NONE','CLEAN_SIG')):
            request=dict(arms=list(d.ARMS),rows=[d.config(row) for row in d.rows()])
            def prediction(row):
                self.assertIn(('train',request['rows'][0]['row_id']),events)
                self.assertIn(('train',request['rows'][1]['row_id']),events)
                events.append(('predict',row))
            def scorer(command,**kwargs):
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],'');events.append(('score',command[-1]))
            with patch.object(d,'read',return_value=request),patch.object(d,'source_selection',return_value={}),patch.object(d,'write'),patch.object(source,'train',side_effect=lambda c:events.append(('train',c['row_id']))),patch.object(e,'source_health',return_value=dict(qualified=True)),patch.object(e,'predict',side_effect=prediction),patch.object(e.subprocess,'run',side_effect=scorer):
                e.run(Path('request.json'),Path('runner.py'))
            self.assertEqual(sum(event[0]=='train' for event in events),6)
            self.assertEqual(sum(event[0]=='predict' for event in events),6)
            self.assertEqual(sum(event[0]=='score' for event in events),7)
            self.assertEqual(events[-1],('score','--finish'))
    def test_unqualified_source_stops_before_training_or_run_root(self):
        from . import source
        with tempfile.TemporaryDirectory() as root,patch.object(d,'BASE',Path(root)/'run'):
            request=dict(arms=list(d.ARMS),rows=[d.config(row) for row in d.rows()])
            with patch.object(d,'read',side_effect=[request,dict(status='RUNNING')]),patch.object(source,'train') as train:
                with self.assertRaises(ValueError):e.run(Path('request.json'),Path('runner.py'))
                train.assert_not_called();self.assertFalse(d.BASE.exists())

if __name__=='__main__':unittest.main(verbosity=2)
