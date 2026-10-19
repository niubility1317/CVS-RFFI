"""Source-only training, frozen seven-view prediction, independent truth scoring.

Executed as a standalone script. Runtime paths are local-only; method recipes
and the row matrix are committed in matrix.json. No target results select rows.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))

VIEWS = ('clean', 'practical_high', 'practical_mid', 'practical_low_suburban',
         'practical_high_urban', 'practical_mid_urban', 'practical_low_urban')
SEEDS = (2026092701, 2026092702, 2026092703)
COUNTS = {'L_s': 6300, 'U_s': 56700, 'V': 27000}
CLASSES = ['14-10', '14-7', '20-15', '20-19', '6-15', '8-20']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    write(tmp, value, exclusive=False)
    os.replace(tmp, path)


def safe_row(c):
    if any(c.get(k) for k in ('truth', 'target_truth', 'target_inputs', 'capsule',
                              'resume', 'teacher', 'checkpoint', 'initial_checkpoint')):
        raise ValueError('Source-only scratch config rejects target/model inputs')
    if c['model_seed'] not in SEEDS or c['variant'] not in ('full', 'ce'):
        raise ValueError('Unregistered row')
    expected = {'riei': (100, 64, .0001), 'drift': (100, 256, .0001),
                'ibrff': (200, 128, .001), 'sdg': (200, 32, .001)}
    if c['family'] not in expected:
        raise ValueError('Unknown method')
    if (c['epochs'], c['batch_size'], c['lr']) != expected[c['family']]:
        raise ValueError('Method budget differs from registered recipe')
    if c.get('weight_decay') != 0 or c.get('split_seed') != 392005:
        raise ValueError('Source or optimizer contract mismatch')
    return c


def seed(value):
    import numpy as np
    import torch
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def prepare_source(rt):
    """Reuse the existing physical roles; construct no target dataset."""
    import numpy as np
    sys.path.insert(0, str(Path(rt['data_code_root']) / 'code'))
    from dataset_wisig import load_wisig_compact_pkl, WiSigCompactDataset
    contract = read(rt['source_contract'])
    if (contract['source_rxs'] != [1, 3, 4, 6, 8] or
        contract['source_days'] != [1, 2, 3] or contract['counts'] != COUNTS or
        contract['split_seed'] != 392005 or contract['ratios'] != [.07, .63, .30]):
        raise ValueError('Source contract mismatch')
    raw = load_wisig_compact_pkl(rt['dataset'])
    if [str(c) for c in raw['tx_list']] != CLASSES:
        raise ValueError('Source class mapping differs from registered mapping')
    base = WiSigCompactDataset(raw, out_len=256, crop_mode='center', normalize=True,
        equalized=1, rx_keep=contract['source_rxs'], day_keep=contract['source_days'], domain='rx_day')
    # Same ID formula as the previously verified contract adapter. U_s is never
    # indexed for training or exported; its TX labels remain unavailable here.
    owners = {sid: role for role, ids in contract['role_ids'].items() for sid in ids}
    out = Path(rt['source_cache']); out.mkdir(parents=True, exist_ok=False)
    roles = {r: [] for r in ('L_s', 'V')}
    for index, item in enumerate(base.index):
        sid = hashlib.sha256(('%s/%s/%s/%s/%s' %
            (item.tx_i, item.rx_i, item.day_i, item.eq_i, item.sig_i)).encode()).hexdigest()[:24]
        role = owners.get(sid)
        if role is None:
            raise ValueError('Source physical ID not present in existing contract')
        if role in roles:
            roles[role].append((index, sid))
    for role, records in roles.items():
        if len(records) != COUNTS[role]:
            raise ValueError('Source role count mismatch')
        x = np.empty((len(records), 2, 256), dtype=np.float32)
        y = np.empty(len(records), dtype=np.int64)
        rx = np.empty(len(records), dtype=np.int64)
        ids = []
        for i, (index, sid) in enumerate(records):
            iq, label, _, meta = base[index]
            x[i] = np.asarray(iq.tolist(), dtype=np.float32)
            y[i] = int(label); rx[i] = contract['source_rxs'].index(int(meta['rx_i']))
            ids.append(sid)
        if not np.isfinite(x).all() or y.min() < 0 or y.max() >= 6:
            raise ValueError('Invalid source input')
        np.save(out / (role + '_x.npy'), x, allow_pickle=False)
        np.savez(out / (role + '_index.npz'), y=y, rx=rx, ids=np.asarray(ids))
    write(out / 'metadata.json', dict(source_contract=contract,
        classes=[str(c) for c in raw['tx_list']], equalized=1, normalize=True,
        out_len=256, counts=COUNTS, U_s_exported=False, target_constructed=False))
    print('SOURCE_CACHE_COMPLETE L_s=6300 V=27000 U_s_exported=false', flush=True)


def array_loader(c, role, batch_size=None):
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    cache = Path(c['source_cache'])
    x = np.load(cache / (role + '_x.npy'), allow_pickle=False)
    with np.load(cache / (role + '_index.npz'), allow_pickle=False) as ix:
        y, rx = ix['y'].copy(), ix['rx'].copy()
    ds = TensorDataset(torch.tensor(x), torch.tensor(y), torch.tensor(rx))
    return DataLoader(ds, batch_size=batch_size or c['batch_size'], shuffle=role == 'L_s',
                      drop_last=False, num_workers=0)


def author_sdg(c):
    sys.path.insert(0, c['sdg_model_root'])
    import main
    return main


def build(c):
    if c['family'] == 'riei':
        from baselines.riei_fd.model import RIEIModel
        return RIEIModel(6, 5, fed_variant='short_stem1d')
    if c['family'] == 'drift':
        from baselines.drift.model import DRIFTModel
        return DRIFTModel(6, 5, domain_discriminator_layers=2)
    if c['family'] == 'ibrff':
        sys.path.insert(0, c['ibrff_model_root'])
        from backbones.FAN import FAN
        return FAN(seq_len=256, in_chans=2, patch_size=32, embed_dim=128,
                   mlp_ratio=2., num_classes=6, bottleneck_type='DSCNN', use_SAF=True, use_FAF=True)
    if c['family'] == 'sdg':
        author = author_sdg(c)
        # The author module seeds to 2023 on import. Undo that import side effect
        # before initializing any registered model seed.
        seed(c['model_seed'])
        return author.MACNN(in_channels=2, channels=8, num_classes=6)
    raise ValueError('Unknown model')


def forward(net, c, x):
    if c['family'] == 'riei':
        return net.ec(net.fed(x)[:, :net.emitter_feature_dim])
    if c['family'] == 'drift':
        return net.tx_classifier(net.encoder(x)[:, :net.split_dim])
    return net(x)[1]


def finite(net, loss):
    import torch
    if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all()
                                      for p in net.parameters() if p.grad is not None):
        raise FloatingPointError('Nonfinite source loss/gradient')


def ordinary_step(net, c, batch, optimizer, fed_optimizer=None):
    import torch.nn.functional as F
    x, y, rx = [v.to(next(net.parameters()).device) for v in batch]
    if c['variant'] == 'ce':
        optimizer.zero_grad(); loss = F.cross_entropy(forward(net, c, x), y)
        loss.backward(); finite(net, loss); optimizer.step()
        return float(loss.detach())
    if c['family'] == 'riei':
        from riei_step import alternating_training_step
        value = alternating_training_step(net, {'iq': x, 'label': y, 'receiver_target': rx},
            optimizer, fed_optimizer, lambda_mi=1.2, lambda_ie=1.2,
            device=x.device, grad_clip_norm=0., disentangle_steps=1)
        if not math.isfinite(value['loss']):
            raise FloatingPointError('Nonfinite RIEI source objective')
        return value['loss']
    from baselines.drift.losses import compute_drift_loss
    optimizer.zero_grad()
    loss = compute_drift_loss(net(x, grl_lambda=1.), y, rx, lambda_grl=1.,
        lambda_center=.01, lambda_mse=.02, mse_reduction='sum', mse_cap=0.,
        normalize_features_for_mse=False, center_mode='batch')['loss']
    loss.backward(); finite(net, loss); optimizer.step()
    return float(loss.detach())


def validation(net, c, loader, device):
    import torch
    import torch.nn.functional as F
    net.eval(); total = correct = count = 0
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = forward(net, c, x)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite source validation')
            total += float(F.cross_entropy(logits, y, reduction='sum'))
            correct += int((logits.argmax(1) == y).sum()); count += len(y)
    return total / count, correct / count


def save(path, net, c, contract, epoch, stage, val_loss):
    import torch
    torch.save(dict(state_dict=net.state_dict(), source_contract=contract, config=c,
        model_seed=c['model_seed'], epoch=epoch, stage=stage, source_val_loss=val_loss,
        scratch_only=True, ancestors=[], default_head_joint_training=True,
        separate_head_training=False, target_access=False), path)


def load(path, c, contract, device='cpu'):
    import torch
    payload = torch.load(path, map_location=device)
    if (payload['source_contract'] != contract or payload['config'] != c or
        payload['model_seed'] != c['model_seed'] or not payload['scratch_only'] or
        payload['ancestors'] != [] or payload['target_access'] is not False):
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    return payload


def train(c):
    import numpy as np
    import torch
    import torch.nn.functional as F
    safe_row(c)
    out = Path(c['output_root']); out.mkdir(parents=True, exist_ok=False)
    meta = read(Path(c['source_cache']) / 'metadata.json')
    if meta['counts'] != COUNTS or meta['U_s_exported'] or meta['target_constructed']:
        raise ValueError('Invalid source cache')
    contract = meta['source_contract']
    write(out / 'resolved_config.json', dict(c, pid=os.getpid(), cwd=os.getcwd(),
        python=sys.executable, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        source_counts=COUNTS, U_s_use=False, target_access=False,
        default_head_joint_training=True, separate_head_training=False))
    device = torch.device('cuda:0')
    completed = []
    stages = 2 if c['family'] == 'ibrff' else 1
    for stage in range(1, stages + 1):
        seed(c['model_seed']); net = build(c).to(device)
        folder = out / ('stage' + str(stage)); folder.mkdir()
        if stage == 2:
            parent = load(out / 'stage1/best.pt', c, contract, device)
            if parent['stage'] != 1:
                raise ValueError('Wrong same-run stage1 parent')
            net.load_state_dict(parent['state_dict'], strict=True)
        if c['family'] == 'riei':
            params = list(net.ec.parameters()) if c['variant'] == 'full' else list(net.ec.parameters()) + list(net.fed.parameters())
            if c['variant'] == 'full': params += list(net.rc.parameters())
            optimizer = torch.optim.SGD(params, lr=c['lr'], momentum=0.)
            fed_optimizer = torch.optim.SGD(net.fed.parameters(), lr=c['lr'], momentum=0.) if c['variant'] == 'full' else None
        else:
            optimizer = torch.optim.Adam(net.parameters(), lr=c['lr'], weight_decay=0.)
            fed_optimizer = None
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
            factor=.1, patience=5, min_lr=1e-6) if c['family'] == 'ibrff' else None
        train_loader, val_loader = array_loader(c, 'L_s'), array_loader(c, 'V', 256)
        sdg = None
        if c['family'] == 'sdg':
            sdg = author_sdg(c)
            # Importing author main seeds at module import. Restore the registered
            # model seed before the retained author summary forward and training.
            if c['variant'] == 'full':
                x = np.load(Path(c['source_cache']) / 'L_s_x.npy', allow_pickle=False)
                with np.load(Path(c['source_cache']) / 'L_s_index.npz') as ix: y = ix['y'].copy()
                train_loader = torch.utils.data.DataLoader(sdg.AugDataset(x, y,
                    sdg.transforms.ToTensor(), [2], [1]), batch_size=32, shuffle=True, num_workers=0)
            sdg.summary(net, (2, 256), device='cuda')
        alpha = .04 if c['family'] == 'ibrff' and stage == 2 and c['variant'] == 'full' else 0.
        hsic = None
        if alpha:
            from utils.hsic import hsic_normalized_cca
            hsic = hsic_normalized_cca
        best = float('inf'); stale = 0; updates = 0
        started = time.monotonic()
        with (folder / 'epochs.jsonl').open('x', encoding='utf-8') as log:
            for epoch in range(1, c['epochs'] + 1):
                net.train(); tic = time.monotonic(); losses = []
                if sdg and c['variant'] == 'full':
                    buffer = io.StringIO()
                    conf = argparse.Namespace(main_aug_depth=[2], aux_aug_depth=[1], lambda_con=[1., 100.])
                    with contextlib.redirect_stdout(buffer):
                        sdg.train(net, [torch.nn.NLLLoss().to(device), sdg.SupConLoss().to(device)],
                                  train_loader, optimizer, epoch, conf)
                    text = buffer.getvalue(); print(text, flush=True)
                    import re
                    match = re.search(r'Loss:\s*(\S+)', text)
                    if not match: raise ValueError('Cannot read author source loss')
                    losses = [float(match.group(1).rstrip(','))]
                    if not math.isfinite(losses[0]): raise FloatingPointError('Nonfinite SDG loss')
                    updates += len(train_loader)
                else:
                    for batch in train_loader:
                        if c['family'] == 'ibrff':
                            x, y, _ = [v.to(device) for v in batch]
                            optimizer.zero_grad(); hidden, logits = net(x)
                            ce = F.nll_loss(F.log_softmax(logits, dim=1), y)
                            loss = ce + alpha * hsic(hidden.flatten(1), x.flatten(1)) if hsic else ce
                            loss.backward(); finite(net, loss); optimizer.step()
                            losses.append(float(loss.detach()))
                        else:
                            losses.append(ordinary_step(net, c, batch, optimizer, fed_optimizer))
                        updates += 1
                val_loss, val_acc = validation(net, c, val_loader, device)
                if scheduler: scheduler.step(val_loss)
                improved = val_loss < best
                if improved:
                    best = val_loss; stale = 0
                    if c['family'] in ('ibrff', 'sdg'):
                        save(folder / 'best.pt', net, c, contract, epoch, stage, val_loss)
                else:
                    stale += 1
                rec = dict(stage=stage, epoch=epoch, loss=sum(losses)/len(losses),
                    source_val_loss=val_loss, source_val_accuracy=val_acc, alpha=alpha,
                    lr=optimizer.param_groups[0]['lr'], updates=updates,
                    seconds=time.monotonic()-tic, target_access=False)
                rec.update(batch_steps_total=updates,
                    optimizer_steps_total=updates*(3 if c['family']=='riei' and c['variant']=='full' else 1),
                    physical_source_exposure_total=epoch*COUNTS['L_s'],
                    train_forward_examples_per_epoch=COUNTS['L_s']*(2 if c['variant']=='full' and c['family'] in ('riei','sdg') else 1))
                log.write(json.dumps(rec, allow_nan=False)+'\n'); log.flush()
                print('EPOCH '+json.dumps(rec), flush=True)
                if c['family'] == 'ibrff' and stale >= 10: break
        save(folder / 'last.pt', net, c, contract, epoch, stage, val_loss)
        completed.append(dict(stage=stage, epochs=epoch, updates=updates,
                              seconds=time.monotonic()-started))
        del optimizer, net; torch.cuda.empty_cache()
    selected = out / ('stage%d' % stages) / ('best.pt' if c['family'] in ('ibrff', 'sdg') else 'last.pt')
    frozen = dict(status='SOURCE_TRAINED_FROZEN', checkpoint=str(selected), stages=completed,
        selection='source_validation_nll' if c['family'] in ('ibrff','sdg') else 'fixed_final_E100',
        target_access=False, target_score_used=False, source_contract=contract, config=c)
    write(out / 'freeze.json', frozen)
    print('SOURCE_TRAINED_FROZEN '+str(selected), flush=True)


def frozen_model(c, device):
    import torch
    source = read(Path(c['source_output']) / 'freeze.json')
    sc = source['config']; safe_row(sc)
    if source['status'] != 'SOURCE_TRAINED_FROZEN' or source['target_score_used']:
        raise ValueError('Training is not frozen')
    expected = read(c['expected_source_contract'])
    if source['source_contract'] != expected:
        raise ValueError('CHECKPOINT_DATA_CONTRACT_MISMATCH')
    payload = load(source['checkpoint'], sc, expected, device)
    net = build(sc).to(device); net.load_state_dict(payload['state_dict'], strict=True)
    net.eval()
    with torch.no_grad(): smoke = forward(net, sc, torch.zeros(4, 2, 256, device=device))
    if smoke.shape != (4, 6) or not torch.isfinite(smoke).all():
        raise ValueError('Frozen real-checkpoint no-query smoke failed')
    return net, sc


def predict(c):
    import numpy as np
    import torch
    if any(c.get(k) for k in ('truth', 'target_truth', 'labels')):
        raise ValueError('Predictor rejects truth')
    seed(c['model_seed']); device = torch.device('cuda:0')
    net, sc = frozen_model(c, device)
    views = Path(c['views_root']); manifest = read(views / 'manifest.json')
    if (manifest['status'] != 'VALIDATED_ONCE' or manifest['count'] != 168000 or
        manifest['scenes'] != list(VIEWS[1:]) or manifest['classes'] != CLASSES or
        manifest.get('source_target_disjoint') is not True):
        raise ValueError('Existing seven-view capsule mismatch')
    with np.load(views / 'index.npz', allow_pickle=False) as ix: ids = ix['ids'].copy()
    out = Path(c['output_root']); out.mkdir(parents=True, exist_ok=False)
    values = {}; times = {}
    with torch.no_grad():
        for view in VIEWS:
            path = manifest['clean_ref'] if view == 'clean' else views / (view + '.npy')
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            if array.shape != (len(ids), 2, 256): raise ValueError('Query shape mismatch')
            start_time = time.monotonic(); predicted = []
            for offset in range(0, len(ids), 128):
                data = np.ascontiguousarray(array[offset:offset+128], dtype=np.float32)
                x = torch.frombuffer(bytearray(data.tobytes()), dtype=torch.float32).reshape(data.shape).to(device)
                logits = forward(net, sc, x)
                if logits.shape != (len(x), 6) or not torch.isfinite(logits).all():
                    raise FloatingPointError('Invalid frozen predictions')
                predicted.extend(logits.argmax(1).cpu().tolist())
            values[view] = np.asarray(predicted, dtype=np.int64)
            times[view] = time.monotonic() - start_time
            print('PREDICT '+view+' '+str(len(predicted)), flush=True)
    np.savez(out / 'predictions.npz', ids=ids, **values)
    write(out / 'complete.json', dict(status='PREDICTIONS_COMPLETE', count=len(ids),
        views=list(VIEWS), truth_read=False, query_fit=False, seconds=times))


def score_preflight(c):
    import numpy as np
    folder = Path(c['prediction_output']); done = read(folder / 'complete.json')
    if done.get('status') != 'PREDICTIONS_COMPLETE' or done.get('views') != list(VIEWS) or done.get('truth_read') is not False or done.get('query_fit') is not False:
        raise ValueError('Truth closed: incomplete prediction')
    with np.load(Path(c['views_root']) / 'index.npz', allow_pickle=False) as ix: expected = ix['ids'].copy()
    with np.load(folder / 'predictions.npz', allow_pickle=False) as p:
        if set(p.files) != {'ids', *VIEWS} or not np.array_equal(p['ids'], expected):
            raise ValueError('Truth closed: prediction identity mismatch')
        values = {v: p[v].copy() for v in VIEWS}
    for value in values.values():
        if value.shape != expected.shape or value.dtype.kind not in 'iu' or value.min() < 0 or value.max() >= 6:
            raise ValueError('Truth closed: invalid classes')
    if len(expected) != done['count'] or len(set(expected.tolist())) != len(expected):
        raise ValueError('Truth closed: coverage mismatch')
    return expected, values


def metrics(y, p):
    import numpy as np
    cm = np.bincount(6*y+p, minlength=36).reshape(6, 6)
    den = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2*np.diag(cm), den, out=np.zeros(6), where=den != 0)
    return dict(count=len(y), accuracy=float(np.trace(cm)/len(y)), macro_f1=float(f1.mean()), confusion=cm.tolist())


def score(c):
    import numpy as np
    ids, values = score_preflight(c)  # Truth path is first consumed below.
    truth = read(c['truth'])
    y = np.asarray([truth[s]['label'] for s in ids], dtype=np.int64)
    rx = np.asarray([str(truth[s]['receiver']) for s in ids])
    if y.min() < 0 or y.max() >= 6 or len(set(rx)) != 7: raise ValueError('Wrong truth join')
    results = []
    for view, predicted in values.items():
        for dimension, strata in [('overall', ['ALL']), ('receiver', sorted(set(rx))), ('transmitter', range(6))]:
            for stratum in strata:
                mask = np.ones(len(y), dtype=bool) if dimension == 'overall' else rx == stratum if dimension == 'receiver' else y == stratum
                results.append(dict(view=view, dimension=dimension, stratum=stratum,
                                    **metrics(y[mask], predicted[mask])))
    out = Path(c['output_root']); out.mkdir(parents=True, exist_ok=False)
    write(out / 'scores.json', dict(status='SCORED', row_id=c['row_id'], results=results,
                                  truth_last=True, target_feedback_forbidden=True))
    write(out / 'complete.json', dict(status='SCORED_COMPLETE', views=list(VIEWS), count=len(ids)))


def worker(c):
    """Separate source/predict/scorer processes, one GPU lease for this row."""
    row = c['row']; root = Path(c['row_root']); root.mkdir(parents=True, exist_ok=False)
    write(root / 'source_config.json', row)
    subprocess.run([sys.executable, '-u', __file__, 'train', '--config', str(root/'source_config.json')], check=True)
    pc = dict(source_output=row['output_root'], expected_source_contract=c['source_contract'],
        model_seed=row['model_seed'], views_root=c['views_root'], output_root=str(root/'prediction'))
    write(root / 'prediction_config.json', pc)
    subprocess.run([sys.executable, '-u', __file__, 'predict', '--config', str(root/'prediction_config.json')], check=True)
    sc = dict(prediction_output=pc['output_root'], views_root=c['views_root'], truth=c['truth'],
              output_root=str(root/'scores'), row_id=row['row_id'])
    write(root / 'score_config.json', sc)
    subprocess.run([sys.executable, '-u', __file__, 'score', '--config', str(root/'score_config.json')],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    write(root / 'complete.json', dict(status='ANALYZED', row_id=row['row_id'], views=list(VIEWS)))


def smoke(rt):
    """One real L_s checkpoint roundtrip per method; no query access."""
    import torch
    out=Path(rt['run_root'])/'smoke';out.mkdir()
    meta=read(Path(rt['source_cache'])/'metadata.json')
    results=[]
    for raw in read(rt['matrix'])['rows']:
        if raw['model_seed']!=SEEDS[0]:continue
        c=dict(raw,source_cache=rt['source_cache'],ibrff_model_root=rt['ibrff_model_root'],sdg_model_root=rt['sdg_model_root'])
        safe_row(c);seed(c['model_seed']);net=build(c).cuda()
        batch=next(iter(array_loader(c,'L_s',12)))
        x,y,_=[v.cuda() for v in batch]
        optimizer=torch.optim.Adam(net.parameters(),lr=c['lr'])
        if c['family']=='drift' and c['variant']=='full':
            from baselines.drift.losses import compute_drift_loss
            loss=compute_drift_loss(net(x),y,batch[2].cuda(),mse_reduction='sum')['loss']
        elif c['family']=='ibrff' and c['variant']=='full':
            sys.path.insert(0,c['ibrff_model_root'])
            from utils.hsic import hsic_normalized_cca
            hidden,logits=net(x)
            loss=torch.nn.functional.cross_entropy(logits,y)+.04*hsic_normalized_cca(hidden.flatten(1),x.flatten(1))
        else:loss=torch.nn.functional.cross_entropy(forward(net,c,x),y)
        optimizer.zero_grad();loss.backward();finite(net,loss);optimizer.step()
        path=out/(c['row_id']+'.pt');save(path,net,c,meta['source_contract'],0,0,0.)
        payload=load(path,c,meta['source_contract'],'cuda:0')
        copy=build(c).cuda();copy.load_state_dict(payload['state_dict'],strict=True)
        net.eval();copy.eval()
        with torch.no_grad():
            a,b=forward(net,c,x),forward(copy,c,x)
        if not torch.equal(a,b):raise ValueError('Real source checkpoint roundtrip differs')
        results.append(dict(row=c['row_id'],status='PASS',real_source_samples=12,checkpoint_roundtrip=True,query_access=False))
        del net,copy,optimizer;torch.cuda.empty_cache()
    write(out/'result.json',dict(status='PASS',rows=results,smoke_weights_discarded=True,query_access=False))


def summarize(rt,state):
    import numpy as np
    root=Path(rt['run_root']);groups={};paired=[]
    for rid,row_state in state['rows'].items():
        if row_state['status']!='ANALYZED':continue
        config=read(root/(rid+'.json'))['row']
        scores=read(root/rid/'scores/scores.json')['results']
        for view in VIEWS:
            overall=next(r for r in scores if r['view']==view and r['dimension']=='overall')
            worst=min(r['accuracy'] for r in scores if r['view']==view and r['dimension']=='receiver')
            key=(config['family'],config['variant'],view)
            groups.setdefault(key,[]).append(dict(seed=config['model_seed'],accuracy=overall['accuracy'],macro_f1=overall['macro_f1'],worst_rx_accuracy=worst))
    summary=[]
    for (family,variant,view),rows in groups.items():
        result=dict(family=family,variant=variant,view=view,actual_seeds=len(rows),rows=rows)
        for field in ('accuracy','macro_f1','worst_rx_accuracy'):
            values=[r[field] for r in rows];result[field+'_mean']=float(np.mean(values))
            result[field+'_sample_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        summary.append(result)
    for family in ('riei','drift','ibrff','sdg'):
        for view in VIEWS:
            full={r['seed']:r for r in groups.get((family,'full',view),[])}
            ce={r['seed']:r for r in groups.get((family,'ce',view),[])}
            shared=sorted(set(full)&set(ce));delta=[full[s]['accuracy']-ce[s]['accuracy'] for s in shared]
            if delta:paired.append(dict(family=family,view=view,actual_pairs=len(shared),seeds=shared,
                full_minus_ce=delta,mean=float(np.mean(delta)),sample_sd=float(np.std(delta,ddof=1)) if len(delta)>1 else None,positive_pairs=sum(x>0 for x in delta)))
    write(root/'summary.json',dict(status=state['status'],summary=summary,paired=paired,expected_model_seeds=list(SEEDS),
                                  target_feedback_forbidden=True))
    lines=['# Phase1 四种域泛化方法与自身 CE','',
        '状态：'+state['status']+'。全部预登记七视图预测固定后独立评分，缺项保留。',
        '', '| 方法 | 版本 | 实际种子 | clean Accuracy（%） | clean Macro-F1（%） |','|---|---|---:|---:|---:|']
    for r in summary:
        if r['view']=='clean':lines.append('| %s | %s | %d | %.4f | %.4f |'%(r['family'],r['variant'],r['actual_seeds'],100*r['accuracy_mean'],100*r['macro_f1_mean']))
    lines+=['','完整种子 SD、逐场景、最差 RX 和配对差见 summary.json；不把不完整三种子当完整矩阵。',
            'PAMF 的既有 E200 clean 对应种子结果另列，不能把不同训练成本隐去。']
    (root/'final_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def dispatch(rt):
    root = Path(rt['run_root']); root.mkdir(parents=True, exist_ok=False)
    logs = Path(rt['log_root']); logs.mkdir(parents=True, exist_ok=False)
    if rt['allowed_gpus'] != [0, 1]: raise ValueError('Exactly registered two GPUs required')
    matrix = read(rt['matrix'])
    if len(matrix['rows']) != 24: raise ValueError('Wrong three-seed matrix')
    for row in matrix['rows']: safe_row(row)
    write(root / 'dispatcher.json', dict(pid=os.getpid(), cwd=os.getcwd(), argv=sys.argv,
        allowed_gpus=rt['allowed_gpus'], launch_owner=rt['launch_owner']))
    prepare_source(rt)
    smoke_gpu=None
    while smoke_gpu is None:
        for gpu in rt['allowed_gpus']:
            current=subprocess.check_output(['nvidia-smi','-i',str(gpu),
                '--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if not current:smoke_gpu=gpu;break
        if smoke_gpu is None:time.sleep(10)
    smoke_env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(smoke_gpu))
    with (root/'smoke.stdout.log').open('x') as log:
        subprocess.run([sys.executable,'-u',__file__,'smoke','--config',rt['runtime_file']],
                       env=smoke_env,stdout=log,stderr=subprocess.STDOUT,check=True)
    state = dict(status='RUNNING', pid=os.getpid(), allowed_gpus=[0, 1], rows={})
    pending = list(matrix['rows']); active = {}; failures = []
    while pending or active:
        for gpu, item in list(active.items()):
            code = item['process'].poll()
            if code is None: continue
            rid = item['row_id']; completed = root/rid/'complete.json'
            state['rows'][rid]['status'] = 'ANALYZED' if code == 0 and completed.exists() else 'FAILED'
            if state['rows'][rid]['status'] == 'FAILED': failures.append(rid)
            item['log'].close(); del active[gpu]
        # Two pre-prediction technical row failures pause unstarted rows. Existing
        # healthy rows finish normally. Scores never enter this decision.
        if len(failures) >= 2 and pending:
            state['unstarted_rows'] = [x['row_id'] for x in pending]; pending.clear()
            state['stop_reason'] = 'two_technical_row_failures'
        for gpu in rt['allowed_gpus']:
            if gpu in active or not pending: continue
            current = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
            if current: continue  # Never displace another owner's workload.
            row = dict(pending.pop(0), source_cache=rt['source_cache'],
                ibrff_model_root=rt['ibrff_model_root'], sdg_model_root=rt['sdg_model_root'])
            rid = row['row_id']; row['output_root'] = str(root/rid/'source')
            config = dict(row=row, row_root=str(root/rid), source_contract=rt['source_contract'],
                          views_root=rt['views_root'], truth=rt['truth'])
            path = root/(rid+'.json'); write(path, config)
            logpath = logs/(rid+'.log'); log = logpath.open('x')
            command = [sys.executable, '-u', __file__, 'worker', '--config', str(path)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='2',
                       MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            child = subprocess.Popen(command, cwd=rt['release_root'], env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active[gpu] = dict(process=child, row_id=rid, log=log)
            state['rows'][rid] = dict(status='RUNNING', pid=child.pid, gpu=gpu,
                                     argv=command, cwd=rt['release_root'], log=str(logpath))
        atomic(root/'pipeline_state.json', state)
        if active or pending: time.sleep(10)
    state['status'] = 'PARTIAL' if failures or state.get('unstarted_rows') else 'ANALYZED'
    state['failures'] = failures
    atomic(root/'pipeline_state.json', state)
    summarize(rt,state)
    write(root/'completion.json', dict(status=state['status'], analyzed=sum(
        r['status'] == 'ANALYZED' for r in state['rows'].values()), expected_rows=24,
        failures=failures, target_feedback_forbidden=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'predict', 'score', 'worker', 'dispatch', 'smoke'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    globals()[args.action](read(args.config))
