"""Keep native PA/U/EMA and account separately for the reused-logit penalty."""
import gzip
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from . import design as d, model
from .source import parent_payload, segment_train
from experiments.cvs_noise_shape_factorial43.observer import Observer as Original

class Observer(Original):
    def __init__(self, c):
        self.c = d.validate(c)
        self.optimizer = None
        self.out = Path(c['output_root'])
        self.parent = parent_payload(c)

    def initialize(self, m, c, out):
        super().initialize(m, c, out)
        actual = dict(config=c, channel=m._full_channel.audit(),
                      modules={k: str(v.__file__) for k, v in sys.modules.items()
                               if k in ['experiments.cvs_pair_margin61.margin',
                                        'experiments.cvs_noise_shape_factorial43.model',
                                        'SSDG.train_ssdg', 'DataAugmentation',
                                        'leo_practical.channel', 'leo_practical.residual']})
        d.write(self.out / 'pair_margin_actual.json', actual)
        print(json.dumps(dict(stage='ACTUAL_MODULES_AND_PAIRED_MARGIN', **actual)), flush=True)

    def epoch(self, m, ctx, rows, out, device):
        super().epoch(m, ctx, rows, out, device)
        if len(m._pair_margin_records) != len(rows) * d.STEPS:
            raise ValueError('Exactly one paired penalty per original labeled update')
        d.write(out / 'pair_margin_progress.json', self.pair_summary())

    def pair_summary(self):
        records = self.model._pair_margin_records
        sums = {k: sum(r[k] for r in records) for k in (
            'pair_raw', 'pair_weighted', 'pair_count', 'q_sum', 'q_nonzero', 'urban_count',
            'clean_correct_count', 'clean_correct_urban_count', 'active_competitor_count', 'competitor_count')}
        n = len(records)
        return dict(steps=n, **{k + '_sum': v for k, v in sums.items()},
                    pair_raw_mean=sums['pair_raw'] / n if n else 0.,
                    pair_weighted_mean=sums['pair_weighted'] / n if n else 0.,
                    clean_margin_mean=sum(r['clean_margin_mean'] for r in records) / n if n else 0.,
                    satellite_margin_mean=sum(r['satellite_margin_mean'] for r in records) / n if n else 0.,
                    q_mean=sums['q_sum'] / (64 * n) if n else 0.,
                    pair_lambda=.5, rho=.5, m_max=1., head_scale=30.,
                    forward_reuse=True, extra_NN=0, extra_U_NN=0,
                    clean_reference_stopgrad=True, all_satellite_CE_retained=True,
                    low_suburban_q_zero=True, two_pass_mean=True, confidence_threshold=False)

    def finish(self):
        result = super().finish()
        if len(self.model._pair_margin_records) != d.UPDATES:
            raise ValueError('Complete2220 paired-penalty records required')
        with gzip.open(self.out / 'pair_margin_steps.jsonl.gz', 'xt', encoding='utf8', compresslevel=3) as f:
            for r in self.model._pair_margin_records:
                f.write(json.dumps(r, allow_nan=False) + '\n')
        result.update(new_epochs=10, parent_epochs=20, cumulative_epochs=30,
                      new_updates=2220, parent_updates=4440, cumulative_updates=6660,
                      pair_margin_summary=self.pair_summary(), source_RX_holdout=None,
                      TX_RX_scene_state_lock=[dict(TX=k[0], RX=k[1], scene=k[2], state=k[3], lock=k[4], count=v)
                                              for k, v in sorted(self.model._full_coverage.items())])
        if (result['L_NN'] != 1704960 or result['L_As'] != 142080
                or result['Ag_packets'] or result['extra_U_NN'] or result['head_fit_updates'] or result['U_truth_read']):
            raise ValueError('Original sixL/nominalAs/oldU/default-head budget differs')
        native=sys.modules['SSDG.train_ssdg'];scratch_args=SimpleNamespace(**self.parent['args'])
        d.write(self.out/'ordinary_augmentation_actual.json',dict(base=native.build_aug_base_cfg(scratch_args),schedule_frozen_at=13,args={k:v for k,v in self.parent['args'].items() if k.startswith('aug_')},schedule_args_epochs=self.parent['args']['epochs'],source='own scratch20 checkpoint args',no_ordinary_before_practical=True,RDrop_same_IQ=True))
        result['native_NN']=sum(result[k] for k in ('student_train_NN','student_eval_NN','teacher_train_NN','teacher_eval_NN'))
        d.write(self.out / 'result.json', result)
        return result

def train(c):
    observer = Observer(c)
    segment_train(c, design=d, model_api=model, observer=observer)
    return observer.finish()
