"""Materialize reviewed CORE90 source sections; never execute a shell launcher."""
from pathlib import Path
import ast
import json
import re
import shlex
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('E:/type10-7/code/snapshots/phase1_adv3_mechanism32_queue_20260701')
DEST = ROOT / 'code/cvsrffi/game_tracking/legacy'
DEST.mkdir(parents=True, exist_ok=True)

def write(name, text):
    path = DEST / name
    assert not path.exists(), path
    path.write_text(text, encoding='utf-8', newline='\n')

write('__init__.py', '"""CORE90 historical architecture/loss definitions, isolated from newer methods."""\n')
for name in ('model.py', 'model_dual_cvsincnet.py'):
    raw = subprocess.check_output(['git', 'show', '766c2ea8:code/' + name], cwd=ROOT).decode('utf-8')
    if name == 'model_dual_cvsincnet.py':
        start = raw.index('_CODE_ROOT =')
        end = raw.index('\ntry:\n    from baselines', start)
        raw = raw[:start] + 'from .model import build_model as build_single_model\n' + raw[end:]
    write(name, '# Architecture source: git 766c2ea8 (2026-06-26). Import isolation only.\n' + raw)
write('losses.py', (SOURCE / 'losses.py').read_text(encoding='utf-8'))
src = (SOURCE / 'train_ssdg.py').read_text(encoding='utf-8')
tree = ast.parse(src)
functions = {node.name: ast.get_source_segment(src, node) for node in tree.body if isinstance(node, ast.FunctionDef)}
names = ['build_arg_parser', '_loss_weights', '_stage_gate_scale', '_threshold_mask',
         '_as_plain_list', '_update_ema_model']
header = '''from __future__ import annotations
import argparse
import math
from typing import Any, Dict, Mapping, Optional, List, Sequence, Tuple
import torch
from post_stage_cli import add_common_data_args, add_sat_eval_args, str2bool
'''
write('options.py', header + '\n\n'.join(functions[n] for n in names) + '\n')
start = src.index('                domain_stats =', src.index('def train(args)'))
end = src.index('                use_sat_train =', start)
block = textwrap.dedent(src[start:end])
block = textwrap.indent(block, '    ')
loss_header = '''from __future__ import annotations
import math
from typing import Dict
import torch
import torch.nn.functional as F
from .losses import *
from .options import _stage_gate_scale

def labeled_terms(out_l, y_l, d_l, args, epoch, batch_idx, cur_w, proto_bank):
'''
tail = '''    terms = {
        "tx": loss_tx_l, "dom": loss_dom_l, "adv": loss_adv_l,
        "orth": loss_orth_l, "cons": loss_cons_l, "group_ce": loss_group_ce_l,
        "fishr": loss_fishr_l, "proto": loss_proto_l,
        "open_world_feat": loss_open_world_feat_l * ow_feat_stage_scale,
        "zid_compact": loss_zid_compact_l * zid_warm,
        "proxy_unknown": loss_proxy_unknown_l * proxy_stage_scale,
        "soft_unknown_mixup": loss_soft_unknown_mixup_l * soft_unknown_mixup_stage_scale,
        "source_episode": loss_source_episode_l * source_episode_stage_scale,
    }
    total = terms["tx"] + sum(cur_w[k] * v for k, v in terms.items() if k != "tx")
    return total, terms
'''
write('objective.py', loss_header + block + tail)
# Parse the known shell syntax as data; only simple literal assignments and substitutions.
launch = (SOURCE / 'launch_phase1_adv3_mechanism32_queue_20260701.sh').read_text(encoding='utf-8')
defaults = launch.split('set_candidate_defaults() {', 1)[1].split('\n}', 1)[0]
values = {}
for line in defaults.splitlines():
    match = re.fullmatch(r'\s*(\w+)=(.*)', line)
    if match:
        values[match[1]] = shlex.split(match[2])[0]
values.update(proxy_core_q='0.90', proxy_accept_q='0.85', proxy_cvar_alpha='0.30', proxy_core_w='0.45',
              pseudo_epochs='70', ROOT='.', WISIG_PKL='Dataset_WigSig/ManySig.pkl', RUNS_ROOT='runs',
              RUN_ID='core90_game', cid='B0', seed='392002')
command = launch.split('    --wisig_pkl', 1)[1].split(')\n}', 1)[0]
command = '    --wisig_pkl' + command
command = re.sub(r'\$\{(\w+)\}', lambda m: values[m.group(1)], command)
tokens = shlex.split(command)
(ROOT / 'code/configs/core90_game_historical_argv.json').write_text(json.dumps(tokens, indent=2), encoding='utf-8')
manifest = {'architecture_git': '766c2ea8', 'trainer_snapshot': str(SOURCE),
            'extracted_functions': names, 'labeled_loss_lines': [src[:start].count('\n')+1, src[:end].count('\n')],
            'changes': ['isolated imports', 'labeled terms returned before optimizer', 'no legacy data or checkpoint loading']}
write('source_manifest.json', json.dumps(manifest, indent=2))
print('Materialized legacy architecture, losses, labeled objective and historical argv')
