"""Pure source-metadata/logit toys; no Torch, checkpoint, IQ or remote calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from . import readout as r


def metadata():
    return dict(status='SOURCE_READOUT_COMPLETE', run_id=r.RUN, model_seed=r.SEED,
        checkpoint_run=r.PARENT_RUN, checkpoint_row=r.PARENT_ROW, checkpoint_stage='stage20',
        checkpoint_key='model', checkpoint_epoch=20, checkpoint_optimizer_steps=4440,
        checkpoint_selection='final_only', source_role='V', source_unique_V=1800,
        channel_draws=2, scenes=list(r.SCENES), condition='full', condition_index=0,
        NN=12602, source_NN=12600, smoke_NN=2, training_updates=0, head_fit_updates=0,
        teacher_training_updates=0, teacher_bank_created=False, U_truth_read=False,
        U_records_used=0, target_access=False, formal_scores_read=False, query_access=False,
        new_channel_calls=0, model_state_unchanged=True, all_parameters_frozen=True,
        model_parameters=900722, model_state_keys=206, default_head_shape=[6, 160],
        default_head_scale=30., eval_counter=12602, train_counter=0,
        source_role_counts=dict(L_s=6300, U_s=56700, V=27000), checkpoint='toy/As20/model',
        source_contract='toy/source_contract.json', source_panel='toy/sourceV1800',
        source_observations='toy/full_condition0', wall_seconds=1., smoke_wall_seconds=.1,
        source_wall_seconds=.9)


def predictions():
    cells=[(tx,rx,day) for tx in range(6) for rx in r.RX for day in r.DAYS]
    y=np.asarray([x[0] for x in cells],np.int64)
    z=np.eye(6,dtype=np.float32)[y]*30
    return dict(ids=np.asarray(['toy'+str(i) for i in range(90)]),TX=y,
        RX=np.asarray([x[1] for x in cells],np.int64),day=np.asarray([x[2] for x in cells],np.int64),
        clean=z.copy(),satellite=np.broadcast_to(z,(2,3,90,6)).copy())


class Checks(unittest.TestCase):
    def test_exact_student_identity_and_forward_budget(self):
        r.validate_metadata(metadata())
        for change in [dict(checkpoint_key='ema_model'),dict(checkpoint_epoch=30),
                dict(checkpoint_optimizer_steps=2220),dict(checkpoint_row='B00-s2026092701'),
                dict(NN=12600),dict(eval_counter=12600),dict(train_counter=1),
                dict(model_state_keys=207),dict(default_head_scale=1.),
                dict(checkpoint_selection='best_clean')]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                r.validate_metadata(dict(metadata(),**change))

    def test_no_training_teacher_truth_target_or_channel_mutation(self):
        for change in [dict(training_updates=1),dict(head_fit_updates=1),
                dict(teacher_bank_created=True),dict(teacher_training_updates=1),
                dict(U_truth_read=True),dict(U_records_used=1),dict(target_access=True),
                dict(formal_scores_read=True),dict(query_access=True),dict(new_channel_calls=1),
                dict(model_state_unchanged=False),dict(all_parameters_frozen=False),
                dict(wall_seconds=float('nan'))]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                r.validate_metadata(dict(metadata(),**change))

    def test_balanced_prediction_identity_shapes_and_finiteness(self):
        p=predictions();r.validate_predictions(p,90)
        for change in ('duplicate','unbalanced','shape','nonfinite','wrongRX'):
            bad=copy.deepcopy(p)
            if change=='duplicate':bad['ids'][0]=bad['ids'][1]
            elif change=='unbalanced':bad['TX'][0]=1
            elif change=='shape':bad['satellite']=bad['satellite'][:1]
            elif change=='nonfinite':bad['clean'][0,0]=np.nan
            elif change=='wrongRX':bad['RX'][0]=2
            with self.subTest(change=change),self.assertRaises(ValueError):r.validate_predictions(bad,90)

    def test_same_order_is_required_without_silent_alignment(self):
        parent=predictions();bad=copy.deepcopy(parent)
        for key in ('ids','TX','RX','day','clean'):bad[key]=bad[key][::-1]
        bad['satellite']=bad['satellite'][:,:,::-1]
        with self.assertRaises(ValueError):r.cohort_metrics(parent,bad)

    def test_actual_parent_correct_set_is_fixed_for_all_views(self):
        parent=predictions();parent['clean'][0]=np.eye(6,dtype=np.float32)[1]*30
        parent['satellite'][:,:,1]=np.eye(6,dtype=np.float32)[1]*30
        current=copy.deepcopy(parent)
        current['satellite'][:,:,0]=np.eye(6,dtype=np.float32)[0]*30  # Outside parent cohort.
        current['satellite'][:,:,1]=np.eye(6,dtype=np.float32)[0]*30  # One repair.
        current['satellite'][:,:,2]=np.eye(6,dtype=np.float32)[1]*30  # One new error.
        z=r.cohort_metrics(parent,current)
        self.assertEqual(z['parent_clean_correct_count'],89)
        cell=next(x for x in z['records'] if x['scene']==r.SCENES[-1] and x['draw']==0 and x['dimension']=='overall')
        self.assertEqual(cell['parent_clean_correct_count'],89)
        self.assertEqual(cell['repairs_within_parent_cohort'],1)
        self.assertEqual(cell['new_errors_within_parent_cohort'],1)
        self.assertEqual(cell['retention_delta_pp'],0.)
        pooled=next(x for x in z['records'] if x['scene']==r.SCENES[-1] and x['draw']==-1 and x['dimension']=='overall')
        self.assertEqual(pooled['parent_clean_correct_count'],178)
        self.assertEqual(pooled['repairs_within_parent_cohort'],2)
        self.assertEqual(z['NN'],0)

    def test_margin_scale_and_empty_state_cohort_are_explicit(self):
        p=predictions();physics=dict(state=np.zeros((2,3,90),np.int8),lock=np.zeros((2,3,90),np.int8))
        z=r.cohort_metrics(p,p,physics)
        overall=next(x for x in z['records'] if x['scene']==r.SCENES[-1] and x['draw']==0 and x['dimension']=='overall')
        self.assertEqual(overall['current_retention'],1.)
        self.assertEqual(overall['mean_margin_over30_within_parent_cohort'],1.)
        empty=next(x for x in z['records'] if x['scene']==r.SCENES[-1] and x['draw']==0 and x['dimension']=='state' and x['stratum']=='BLOCKED')
        self.assertIsNone(empty['current_retention']);self.assertTrue(empty['empty_cohort'])
        bad=dict(physics,state=np.zeros((2,3,11,90),np.int8))
        with self.assertRaises(ValueError):r.cohort_metrics(p,p,bad)

    def test_output_writer_never_overwrites(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'toy.json';r.write(p,dict(a=1))
            with self.assertRaises(FileExistsError):r.write(p,dict(a=2))
            self.assertEqual(json.loads(p.read_text()),dict(a=1))


def run():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    if not result.wasSuccessful():raise AssertionError('Parent cohort toy checks failed')
    return dict(status='PASS',tests=result.testsRun,NN=0,IQ_reads=0,checkpoint_reads=0,
                real_truth_reads=0,updates=0,teacher_forwards=0,remote_calls=0)


if __name__=='__main__':
    print(json.dumps(run()),flush=True)
