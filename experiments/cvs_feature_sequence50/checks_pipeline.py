"""Focused protocol negatives, default-head closure and source-only gates."""
import ast,json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import design as d

def check():
 from . import predict,score,dispatch
 from experiments.cvs_report_ssl import evaluate
 with tempfile.TemporaryDirectory(prefix='sequence_noquery_') as td:
  root=Path(td);ids=np.arange(168000,dtype=np.int64);row=d.rows()[1];rowid=row['row_id']
  with patch.object(d,'BASE',root),patch.object(predict,'source_proof',lambda c:root/c['row_id']/'source'),patch.object(evaluate,'physical_ids',lambda:ids):
   c=d.config(row);out=root/rowid/'prediction';out.mkdir(parents=True)
   provenance=dict(status='VERIFIED',config=c,checkpoint=str(root/rowid/'source/final_ssdg.pth'),baseline_native_reused=False,default_classifier_only=True,separate_head_training=False,backbone_frozen=True,truth_read=False,query_fit=False,historical_benchmark_exposed=True)
   done=dict(status='PREDICTIONS_COMPLETE',config=c,count=168000,views=['clean'],readouts=['NATIVE_HEAD'],truth_read=False,query_fit=False,head_fit_updates=0)
   d.write(out/'provenance.json',provenance);d.write(out/'complete.json',done)
   np.savez(out/'predictions.npz',ids=ids,NATIVE_HEAD=np.zeros(168000,dtype=np.int64))
   predict.preflight(rowid)
   mutations=[dict(count=167999),dict(readouts=['REFIT_COSINE']),dict(truth_read=True),dict(query_fit=True),dict(head_fit_updates=1)]
   for mutation in mutations:
    d.write(out/'complete.json',dict(done,**mutation))
    with patch.object(evaluate,'truth_arrays') as truth:
     try:score.score(rowid)
     except ValueError:pass
     else:raise AssertionError('Invalid predictions reached scoring')
     assert not truth.called
   d.write(out/'complete.json',done)
   np.savez(out/'predictions.npz',ids=ids[::-1],NATIVE_HEAD=np.zeros(168000,dtype=np.int64))
   try:predict.preflight(rowid)
   except ValueError:pass
   else:raise AssertionError('Physical ID mismatch accepted')
  from . import provenance
  with patch.object(d,'BASE',root),patch.object(provenance,'source_proof',lambda c:root/c['row_id']/'source'):
   later=d.rows()[8]
   assert not dispatch.eligible(later)
   for rr in d.rows()[:8]:
    cc=d.config(rr);p=root/cc['row_id']/'source';p.mkdir(parents=True,exist_ok=True)
    d.write(p/'completion.json',dict(final_training_L_CE=.2,final_source_metrics=dict(source_val_accuracy=.98,source_val_worst_rx=.94)))
   # Eligibility's existence check includes the reused historical baseline.
   with patch.object(Path,'is_file',lambda p:True):
    assert dispatch.eligible(later)
    p=root/d.rows()[1]['row_id']/'source/completion.json';bad=d.read(p);bad['final_training_L_CE']=.6;d.write(p,bad)
    try:dispatch.eligible(later)
    except ValueError:pass
    else:raise AssertionError('Source health failure expanded seeds')
 # Native training closes by calling this function even when its gate is off.
 # The scoped replacement must never enter the original fitted-probe path.
 from .source import native_default_head_only
 from .model import source_domain_count
 from experiments.cvs_phase1_stack.runtime import installed
 c=d.config(d.rows()[1]);args=d.make_args(c,'cpu');args.num_domains=source_domain_count(args);args.input_len=256
 with installed(c) as native:
  original=native._evaluate_zid_leakage_probes
  with patch.object(native,'frozen_ridge_linear_probe',side_effect=AssertionError('Separate classifier fitting attempted')) as fit:
   with native_default_head_only(native) as evidence:
    disabled=native._evaluate_zid_leakage_probes(None,None,None,args)
    decision=native._assess_zid_leakage_probe(disabled,args)
    assert not decision['fired'] and evidence['disabled_calls']==1 and evidence['head_fit_updates']==0
    assert not fit.called
   assert native._evaluate_zid_leakage_probes is original
 source_tree=ast.parse((Path(__file__).parent/'source.py').read_text())
 train=next(n for n in source_tree.body if isinstance(n,ast.FunctionDef) and n.name=='train')
 assert any(isinstance(n,ast.With) and any(isinstance(it.context_expr,ast.Call) and isinstance(it.context_expr.func,ast.Name) and it.context_expr.func.id=='native_default_head_only' for it in n.items) for n in ast.walk(train))
 # No fitting routine or probe import is on the default prediction path.
 for name in ('predict.py','diagnostics.py','dispatch.py'):
  text=(Path(__file__).parent/name).read_text();tree=ast.parse(text)
  assert 'CosineProbe' not in text and 'REFIT_COSINE' not in text and '.probe import' not in text
  assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('backward','step') for n in ast.walk(tree))
 return dict(status='PASS',truth_closed_until_complete=True,wrong_ids_rejected=True,extra_readouts_and_head_updates_rejected=True,source_only_expansion=True,no_diagnostic_or_prediction_training=True,native_fitted_RX_day_channel_probes_disabled=True)
if __name__=='__main__':print(json.dumps(check()))
