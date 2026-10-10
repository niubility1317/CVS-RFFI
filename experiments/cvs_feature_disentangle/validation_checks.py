"""Real-adapter dependency probes, full-V accounting and source-only selection."""
import argparse
import copy
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import design as d
from .fishr import ConditionalFishr
from .style import WeakConditionalMixStyle
from .validation import (evaluate_source_stress, source_select, source_readonly,
                         time_contribution_off, _equal)
from experiments.cvs_phase1_overlay.model import native_modules, Phase1IdentityAdapter


class FastClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_down = nn.Identity()
        self.t_proj = nn.Linear(2, 160)
        self.register_buffer('freq', torch.arange(12, dtype=torch.float32).reshape(2, 6)/12)

    def forward(self, x):
        return self.t_proj(self.time_down(x).mean(2))[:, :6] + x.mean(2) @ self.freq


def run(device='cpu'):
    torch.set_num_threads(2)
    native_modules()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7333)
        identity = Phase1IdentityAdapter(SimpleNamespace(num_classes=6, input_len=256)).to(device)
    identity.train()
    style = WeakConditionalMixStyle(343).attach(identity)
    fishr = ConditionalFishr()
    before, style_before = copy.deepcopy(identity.state_dict()), style.state_dict()
    rng = torch.random.get_rng_state().clone()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    generator = torch.Generator().manual_seed(918)
    x = torch.randn(4, 2, 256, generator=generator).to(device)
    captures = {}
    handles = []
    for name, module in identity.named_modules():
        if name.split('.')[-1] in ('f_proj', 'pa_proj', 'response'):
            captures[name] = []
            handles.append(module.register_forward_hook(
                lambda module, inputs, output, key=name: captures[key].append(output.detach().clone())))
    with source_readonly(identity, style, fishr), torch.no_grad():
        full = identity(x)
        with time_contribution_off(identity) as probe:
            off = identity(x)
        assert probe['count'] == 1
        assert (full-off).abs().max() > 0
    for handle in handles:
        handle.remove()
    assert captures and all(len(values) == 2 and torch.equal(*values) for values in captures.values())
    assert any(name.endswith('f_proj') for name in captures)
    assert any(name.endswith('pa_proj') for name in captures)
    assert any(name.endswith('response') for name in captures)
    assert identity.training and _equal(before, identity.state_dict())
    assert style_before == style.state_dict() and not fishr.buckets
    assert torch.equal(rng, torch.random.get_rng_state())
    if cuda is not None:
        assert all(torch.equal(a, b) for a, b in zip(cuda, torch.cuda.get_rng_state_all()))
    style.detach()

    fixture = FastClassifier().train()
    style = WeakConditionalMixStyle(557).attach(fixture)
    def loader(count=27000):
        # Deliberate native RNG use verifies validation restores loader RNG too.
        torch.rand(1); np.random.rand(); random.random()
        private = torch.Generator().manual_seed(883)
        for start in range(0, count, 1000):
            n = min(1000, count-start)
            xx = torch.randn(n, 2, 256, generator=private)
            yy = (torch.arange(n)+start)%6
            meta = dict(rx_i=torch.tensor([1,3,4,6,8]).repeat((n+4)//5)[:n],
                        day_i=(torch.arange(n)+start)%3)
            yield xx, yy, torch.zeros(n, dtype=torch.long), meta
    c = d.config(d.rows()[0])
    with tempfile.TemporaryDirectory(prefix='cvs-feature-validation-') as temporary:
        root = Path(temporary)
        rng, numpy_rng, python_rng = torch.random.get_rng_state(), np.random.get_state(), random.getstate()
        ss = style.state_dict()
        result = evaluate_source_stress(fixture, loader(), 'cpu', c, root, style=style, fishr=fishr)
        assert set(result['views']) == set(d.STRESS['views'])
        assert all(v['count'] == 27000 for v in result['views'].values())
        assert result['time_dependency_probe']['time_off']['count'] == 27000
        assert result['time_dependency_probe']['hook'] == 't_proj'
        for v in result['views'].values():
            for quality in v['quality_strata'].values():
                assert sum(s['count'] for s in quality['bins'].values()) == 27000
            assert all('margin_q10' in g and 'macro_f1' in g and 'error_tail_rate' in g
                       for values in v['group_metrics'].values() for g in values.values())
        assert fixture.training and style.state_dict() == ss and not fishr.buckets
        assert torch.equal(rng, torch.random.get_rng_state())
        assert _equal(numpy_rng, np.random.get_state()) and python_rng == random.getstate()
        try:
            evaluate_source_stress(fixture, loader(10), 'cpu', c, root/'incomplete', style=style)
        except ValueError as error:
            assert 'Incomplete source V' in str(error)
        else:
            raise AssertionError('Incomplete source V accepted')
        assert not (root/'incomplete'/'source_stress.json').exists()

        arms = ['LTR_S','LTR_S_AF3','LTR_S_AF5','LTR_S_AF10']
        rows = [row for row in d.rows() if row['arm'] in arms]
        def record(row, clean=.8, pressure=.81):
            return dict(status='SOURCE_STRESS_COMPLETE', target_access=False, **row, config=d.STRESS,
                        views={name: dict(count=27000, accuracy=clean if name=='clean' else pressure,
                                          worst_rx=(clean if name=='clean' else pressure)-.1)
                               for name in d.STRESS['views']})
        for row in rows:
            d.write(root/row['row_id']/'source_stress.json', record(row, pressure=.82 if row['arm']=='LTR_S_AF5' else .81))
        (root/'target_scores.json').write_text('POISON: forbidden target access', encoding='utf-8')
        original_read = d.read
        opened = []
        def guarded_read(path):
            path = Path(path)
            assert path.name in ('source_stress.json', 'source_selection.json')
            opened.append(str(path))
            return original_read(path)
        with patch.object(d, 'config', lambda row: dict(output_root=str(root/row['row_id']))), \
             patch.object(d, 'BASE', root/'selection'), patch.object(d, 'read', guarded_read):
            selected = source_select()
            assert selected['selected_ratio'] == .05 and selected['selected_companion'] == 'LTR_S_F5'
            assert selected['target_test_rows'] == 36 and not selected['target_scores_consumed']
            assert source_select() == selected
            expected_score = ((.8+.7)/2 + 7*(.82+.72)/2)/8
            assert abs(next(v['score'] for v in selected['records'] if v['ratio']==.05)-expected_score) < 1e-12
            with patch.object(d, 'rows', lambda: rows[:-1]):
                try: source_select()
                except ValueError as error: assert 'Incomplete' in str(error)
                else: raise AssertionError('Incomplete seed selection accepted')
            changed = next(row for row in rows if row['arm']=='LTR_S_AF3')
            d.write(root/changed['row_id']/'source_stress.json', record(changed, pressure=.95))
            try: source_select()
            except ValueError as error: assert 'Immutable' in str(error)
            else: raise AssertionError('Frozen selection overwritten')
        for row in rows:
            d.write(root/row['row_id']/'source_stress.json', record(row))
        with patch.object(d, 'config', lambda row: dict(output_root=str(root/row['row_id']))), \
             patch.object(d, 'BASE', root/'ties'):
            assert source_select()['selected_ratio'] == .03
        for row in rows:
            clean = .8 if row['arm']=='LTR_S' else (.79 if row['arm']=='LTR_S_AF10' else .78)
            d.write(root/row['row_id']/'source_stress.json', record(row, clean=clean))
        with patch.object(d, 'config', lambda row: dict(output_root=str(root/row['row_id']))), \
             patch.object(d, 'BASE', root/'fallback'):
            fallback = source_select()
            assert fallback['selected_ratio']==.1 and fallback['no_clean_noninferior_candidate']
    style.detach()
    return dict(status='PASS', device=device, actual_adapter_time_vector_probe=True,
                frequency_PA_reference_outputs_unchanged=True, source_V_count=27000,
                pressure_views=sorted(d.STRESS['views']), clean_quality_quartiles_all_packets=True,
                grouped_macroF1_margin_errorTail=True, modes_RNG_style_Fishr_state_unchanged=True,
                incomplete_V_and_seeds_rejected=True, immutable_source_only_selection=True,
                all_eight_views_score=True, exact_tie_lower_ratio=True, no_feasible_fallback=True,
                selected_training_rows=48, selected_test_rows=36)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    result=run(args.device)
    path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
