"""Complete source results and full training cost; no target-score inputs."""
from pathlib import Path
from . import design as d
from experiments.cvs_urban_semantic50._audit_worker import parse_steps


def collect():
    result=d.read(d.BASE/'comparison/results.json')
    if not result['source_only'] or result['query_read'] or result['U_truth_read']:raise ValueError('Source-only teacher comparison required')
    costs=[];curves={}
    for row in d.rows():
        source=Path(d.config(row)['output_root']);done=d.read(source/'completion.json');mechanism=d.read(source/'training_mechanism_complete.json')
        parsed=parse_steps(source/'step_metrics.jsonl.gz')
        if (done['epoch']!=50 or done['optimizer_steps']!=11100 or done['logged_steps']!=11100 or done['target_access']
                or done['head_fit_updates'] or mechanism['U_forward_packets'] or mechanism['EMA_updates']
                or mechanism['unique_train_samples']!=4464 or mechanism['forward_packets']['train']!=1420800):
            raise ValueError('Actual equal-budget L-only teacher training differs')
        costs.append(dict(arm=row['arm'],parent_training_epochs=0,training_epochs=50,updates=11100,
            training_seconds=done['elapsed_seconds'],actual_model_forward_packets=mechanism['forward_packets'],
            source_evaluation=d.read(source.parent/'source_evaluation/complete.json'),
            unique_training_samples=4464,train_receivers=mechanism['train_receivers'],clipping=parsed['clipping'],
            full_step_log_read=parsed['complete_steps'],U_forward_packets=0,EMA_updates=0,head_fit_updates=0))
        curves[row['arm']]=parsed['curves']
    d.write(d.BASE/'resources.json',dict(rows=costs,initial_reference_smoke_packets=2,comparison_forward_packets=1440,
        comparison_models_trained=0,teacher_training_updates=22200,student_models_trained=0))
    d.write(d.BASE/'training_curves.json',curves)
    lines=['# Fixed RX1: equal-budget source teacher comparison','',
        'Two independent scratch E50 SIG_TIME/default heads;4464 labels each;0 U/EMA/augmentation. No target predictions or KD student.','',
        '|source view|region|n|SAMPLE %|EXCLUDE_RX1 %|repairs|new errors|net|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in result['records']:
        if r['region'] not in ('RX1','RX1_old_gate_rejected'):continue
        a='N/A' if r['sample_accuracy'] is None else f"{100*r['sample_accuracy']:.4f}"
        b='N/A' if r['excluded_accuracy'] is None else f"{100*r['excluded_accuracy']:.4f}"
        lines.append(f"|{r['view']}|{r['region']}|{r['count']}|{a}|{b}|{r['repairs']}|{r['new_errors']}|{r['net']}|")
    lines+=['','Source complementarity rule passed: '+str(result['source_decision']['passed']),
        'Native old-gate replay uses source L_dev batching and cloned ephemeral train buffers; it is not historical U acceptance.',
        'No automatic bank/student expansion, cross-seed or formal target claim. All teacher training and source forward costs are retained.']
    (d.BASE/'analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result
