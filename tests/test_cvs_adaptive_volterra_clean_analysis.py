"""Public synthetic confusion matrices; never reads a real score or query."""
import json,statistics
import numpy as np
from comparison_suite.score import metrics

def test_all256_confusions_and_28000_per_tx_report(tmp_path,monkeypatch):
    from experiments.cvs_adaptive_volterra_clean import analyze as a
    from experiments.cvs_equivariant_identity.precision import FULL_FP32_POLICY
    candidate='adaptive_volterra_lag1';seeds=range(2026092701,2026092705)
    methods=['native','cvcnn','real_cnn','resnet1d','residual_fusion','energy_equivariant','coupled_lag4',candidate]
    receivers=['ALL',*map(str,range(7))];records=[];summary=[];pairs=[]
    # Explicit 28000 per TX and 24000 per RX; all-perfect public fixture.
    for m in methods:
        for rx in receivers:
            count=28000 if rx=='ALL' else 4000
            cm=np.eye(6,dtype=int)*count
            metric=dict(accuracy=1.,macro_accuracy=1.,macro_f1=1.,query_count=count*6,confusion=cm.tolist())
            records.extend(dict(method=m,receiver=rx,model_seed=s,**metric) for s in seeds)
            summary.append(dict(method=m,receiver=rx,accuracy_mean=1.,accuracy_seed_sd=0.,macro_accuracy_mean=1.,macro_accuracy_seed_sd=0.,macro_f1_mean=1.,macro_f1_seed_sd=0.))
    for rx in receivers:
        pairs.extend(dict(candidate=candidate,baseline=m,receiver=rx,accuracy_delta_pp_by_seed=[0.]*4,accuracy_delta_pp_mean=0.,accuracy_delta_pp_seed_sd=0.,positive_seeds=0) for m in methods[:-1])
    profile=dict(total_parameters=202555,trainable_parameters=202555,gradient_used_parameters=202555,resident_state_bytes=810212,
                 conv_linear_macs_per_sample=1,inference_batch1_ms=1.,training_batch128_ms=1.,hardware='public CPU fixture',torch_version='fixture',
                 mac_scope='fixture',fft_calls_per_sample=1,adaptive_contract={'fixture':True})
    source=dict(rows=[dict(resolved=dict(variant=candidate,model_seed=s),profile=profile,epochs=[dict(peak_cuda_allocated_bytes=0)],
                          physical_diagnostics=dict(phase_audit=[dict(received_cfo_hz=0.,whole_logit_max_abs_error=0.,whole_unit_embedding_max_distance=0.)],normalization={},adaptive_input=dict(coefficients=[.1,-.2]))) for s in seeds])
    d=dict(classes=list('abcdef'),rows=[dict(resolved=dict(variant=candidate,numerical_policy=FULL_FP32_POLICY,backend_flags=FULL_FP32_POLICY),completion=dict(prediction_seconds=1.,peak_cuda_allocated_bytes=0)) for s in seeds],
           marker=dict(status='SCORED_COMPLETE',rows=32,query_count=168000),results=dict(results=records),summary=dict(summary=summary,paired=pairs))
    selection=dict(selected_variant=candidate,new_candidate_selected=True)
    previous=dict(results=dict(results=[r for r in records if r['method']!=candidate]))
    def read(path):
        p=str(path)
        if p.endswith('performance_selection.json'):return selection
        if p.endswith('source_research_complete.json'):return source
        if a.OLD_RUN in p:return previous
        if p.endswith('final_readback.json'):return d
        raise AssertionError('Unexpected read '+p)
    monkeypatch.setattr(a,'read',read)
    report=tmp_path/'automation_reports/CV-SincNet'/a.RUN;(report/'evidence').mkdir(parents=True)
    a.analyze(tmp_path)
    result=json.loads((report/'evidence/analysis_validation.json').read_text(encoding='utf-8'))
    assert result['all256_confusions_recomputed'] and result['old224_metrics_exactly_unchanged']
    assert result['genuine_RFF_physics_aware_goal_achieved'] is False
    text=(report/'report.md').read_text(encoding='utf-8');assert '28000' in text and '统一32行' in text and '新增2参数' in text and '冻结后实际残差系数' in text
