from pathlib import Path
import torch
from . import design as d, model
from experiments.cvs_t1_e50.observer import Observer as OriginalObserver
from experiments.cvs_crossview_supcon41.source import segment_train

class Observer(OriginalObserver):
    def epoch(self, m, ctx, rows, out, device):
        super().epoch(m, ctx, rows, out, device)
        if self.c['restored_augmentation']['enabled']:
            summary = m._full_channel.summary()
            if summary['DURA_packets'] != len(rows)*222*32 or summary['G_updates']:
                raise ValueError('Exactly32 of As64 slots require frozen DURA output')
            expected = self.c['restored_augmentation']['crossview_mode']
            records = m._full_records[-222:]
            if any(r['crossview_mode']!=expected or r['raw_As']!=32 or r['DURA_As']!=32 for r in records):
                raise ValueError('Actual mixed views or SupCon membership differs')
            for r in records:
                for detail in r['SupCon_details']:
                    if (detail['total_anchors']!=(192 if expected=='SAT192' else 128)
                            or detail['same_RX_positive_pairs'] or detail['DURA_anchors']!=(32 if expected=='SAT192' else 0)):
                        raise ValueError('Actual crossRX geometry or restored anchors differs')
            d.write(out/'DURA_progress.json', summary)
    def finish(self):
        result = super().finish()
        result['calendar'] = dict(label_epochs=13, pseudo_epochs=37, scratch=True, epochs=50)
        result['augmentation'] = self.c['restored_augmentation']
        if self.c['restored_augmentation']['enabled']:
            result['DURA'] = self.model._full_channel.summary()
        d.write(self.out/'result.json', result)
        return result

def train(c):
    observer = Observer(c)
    segment_train(c, design=d, model_api=model, observer=observer)
    return observer.finish()
