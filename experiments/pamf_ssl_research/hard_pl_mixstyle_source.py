"""Frozen V identity-pair falsification; V never updates any model state."""
from pathlib import Path
import gc
import numpy as np
import torch
from experiments.cric_dura_comparison.io import read,write
from experiments.cric_dura_comparison.f_data import SourceDataset
from experiments.cric_dura_comparison.frozen_inference import reader_from_payload
from .hard_pl_train import source_parents
from .hard_pl_mixstyle import validate
from .hard_pl_mixstyle_objective import style_hooks
from .u_mixstyle_objective import protected_partners


def run(rt,device='cuda:0'):
    rows=[]
    for row in rt['rows']:
        c=read(row['train_config']);parents,contract=source_parents(c,validate_config=validate)
        parent=parents[-1];payload=torch.load(Path(parent['folder'])/'final_ssdg.pth',map_location='cpu',weights_only=False)
        teacher=reader_from_payload(dict(parent['config'],parent_source=None),payload,device)
        data=SourceDataset(c['source_cache'],'V',contract);m=data.metadata;ids=[]
        for cls in range(6):
            for rx in c['source_rxs']:
                for day in c['source_days']:
                    ids.extend(np.flatnonzero((m['y']==cls)&(m['rx_i']==rx)&(m['day_i']==day))[:8].tolist())
        if len(ids)!=720:raise ValueError('Fixed balanced source V720 required')
        ids=np.asarray(ids)[np.random.default_rng(2026102401).permutation(720)]
        torch.manual_seed(c['model_seed']+20261024)
        accepted_count=pseudo_correct=eligible=pair_correct=styled_correct=changed_predictions=0
        applied={name:0 for name in c['hard_pl_mixstyle']['levels']}
        with torch.inference_mode():
            for begin in range(0,720,256):
                ix=ids[begin:begin+256].tolist();x=data.x[ix].to(device);rx=torch.tensor(m['rx_i'][ix].tolist(),device=device)
                truth=torch.tensor(m['y'][ix].tolist(),device=device)
                q=teacher.classify_features(teacher.features(x)).softmax(1);confidence,pseudo=q.max(1)
                accepted=confidence>=.95
                partner,valid=protected_partners(q,accepted.to(q.dtype),rx,c['source_rxs'],1.)
                if bool((valid&(~accepted|~accepted[partner]|(pseudo!=pseudo[partner])|(rx==rx[partner]))).any()):
                    raise ValueError('Source style plan violates accepted-only pseudo/RX permissions')
                alpha=torch.tensor(.1,device=device);lam=torch.distributions.Beta(alpha,alpha).sample((len(x),1,1))
                with style_hooks(teacher,'core.id_backbone.',c['hard_pl_mixstyle'],partner,valid,lam,force=True) as actual:
                    mixed=teacher.classify_features(teacher.features(x)).softmax(1)
                if mixed.shape!=q.shape or not bool(torch.isfinite(mixed).all()):raise ValueError('Nonfinite mixed default-head prediction')
                accepted_count+=int(accepted.sum());pseudo_correct+=int((accepted&(pseudo==truth)).sum())
                eligible+=int(valid.sum());pair_correct+=int((valid&(truth==truth[partner])).sum())
                styled_correct+=int((accepted&(mixed.argmax(1)==truth)).sum())
                changed_predictions+=int((accepted&(mixed.argmax(1)!=pseudo)).sum())
                for name,n in actual.items():applied[name]+=n
        if teacher.training or any(p.requires_grad for p in teacher.parameters()):raise ValueError('Frozen source model changed')
        passed=(accepted_count>0 and eligible>0 and 100*pseudo_correct>=99*accepted_count
                and 100*pair_correct>=99*eligible and all(v>0 for v in applied.values()))
        rows.append(dict(model_seed=c['model_seed'],samples=720,accepted=accepted_count,
            accepted_raw_pseudo_correct=pseudo_correct,accepted_raw_precision_percent=100*pseudo_correct/accepted_count,
            style_eligible=eligible,same_true_TX_pairs=pair_correct,true_TX_pair_agreement_percent=100*pair_correct/eligible,
            accepted_styled_prediction_correct=styled_correct,accepted_styled_prediction_accuracy_percent=100*styled_correct/accepted_count,
            accepted_prediction_changed=changed_predictions,style_applied=applied,scientific_rule_pass=passed))
        del teacher,payload,data;gc.collect();torch.cuda.empty_cache()
    result=dict(status='PASS' if all(v['scientific_rule_pass'] for v in rows) else 'SCIENTIFIC_FALSIFIED',rows=rows,
        rule='Each fixed seed: accepted raw V pseudo precision>=99%,true TX agreement of eligible same-pseudo cross-RX pairs>=99%,each actual style level applied>0;no threshold on styled classifier accuracy.',
        model_updates=0,normalization_updates=0,EMA_updates=0,head_fit_updates=0,U_truth_reads=0,target_reads=0,
        selection_seed=2026102401,style_seed='model_seed+20261024',both_style_levels_forced_in_diagnostic=True,
        interpretation='Source pair identity diagnostic only.Frozen styled prediction errors do not prove wrong raw pseudo labels or invalid augmentation;no target accuracy claim.')
    write(Path(rt['run_root'])/'source_style_identity.json',result)
    return result
