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
            if summary['DURA_packets'] != len(rows)*222*64 or summary['G_updates']:
                raise ValueError('Every As slot requires frozen DURA output')
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
