"""Analyze complete frozen source artifacts; never consume target scores."""
import argparse
import csv
import json
import statistics
from pathlib import Path
from experiments.cvs_moment_residual_identity.prepare import RUN
from experiments.cvs_moment_residual_identity.dispatch import CONTROL, CONTROL_RUN
from experiments.cvs_moment_residual_identity.collect import validate_completed
from experiments.cvs_moment_residual_identity.model import VARIANTS


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')


def table(path, rows):
    if not rows:
        raise ValueError('Empty source evidence table')
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def input_record(lift, variant, seed, epoch=None):
    rec = lift['records'][0]
    flat = {key: value for key, value in rec.items() if not isinstance(value, (list, dict))}
    flat.update(dict(variant=variant, seed=seed, epoch=epoch))
    for i in range(4):
        flat['moment_mean_delay'+str(i)] = rec['moment_mean_by_delay'][i]
        flat['moment_variance_delay'+str(i)] = rec['moment_variance_by_delay'][i]
    for prefix, values in [('mix_raw', lift['raw_parameters']), ('mix_coefficient', lift['coefficients']),
                           ('projection_raw', lift['projection_raw_parameters']),
                           ('projection_coefficient', lift['projection_coefficients'])]:
        flat.update({prefix+str(order): value for order, value in zip((3, 5), values)})
    return flat


def analyze(root):
    report = root/'automation_reports/CV-SincNet'/RUN
    evidence = report/'evidence'
    data = read(evidence/'source_research_complete.json')
    selection = validate_completed(data)
    controls = [row for row in read(root/'automation_reports/CV-SincNet'/CONTROL_RUN/'evidence/source_research_complete.json')['rows']
                if row['resolved']['variant'] == CONTROL]
    if len(controls) != 4:
        raise ValueError('Missing four complete source control curves')
    for row in controls:
        actual = next(record for record in data['source_controls'] if record['seed'] == row['resolved']['model_seed'])
        final = row['completion']['final_source_metrics']
        if actual['accuracy'] != final['source_val_accuracy'] or actual['worst_rx'] != final['source_val_worst_rx']:
            raise ValueError('Control curves differ from independently verified final source metadata')
    tables = {key: [] for key in ('source_curves', 'source_final', 'source_RX', 'source_resources',
        'source_all720_cells', 'source_geometry', 'source_paired_control', 'source_gates_all1600',
        'source_input_all1600', 'source_moments_all6400', 'public_input_all8', 'public_moments_all32',
        'source_normalization_all9600', 'public_normalization_all48', 'source_phase_all24',
        'public_cascade_all240', 'public_isolated_TX_all32')}
    for row in controls+data['rows']:
        cfg = row['resolved']; variant = cfg['variant']; seed = cfg['model_seed']
        epochs = row['epochs']; profile = row['profile']
        if [rec['epoch'] for rec in epochs] != list(range(1, 201)):
            raise ValueError('Incomplete full source curve')
        for rec in epochs:
            tables['source_curves'].append(dict(variant=variant, seed=seed, epoch=rec['epoch'],
                V=rec['source_val_accuracy'], worst_RX=rec['source_val_worst_rx'], V_CE=rec['source_val_ce'],
                train_CE=rec['clean_ce'], gradient_norm=rec['gradient_norm'], learning_rate=rec['learning_rate'],
                elapsed_seconds=rec['elapsed_seconds']))
            if variant in VARIANTS:
                diag = rec['moment_diagnostics']
                flat = input_record(diag['moment_input'], variant, seed, rec['epoch'])
                tables['source_input_all1600'].append(flat)
                tables['source_moments_all6400'].extend(dict(variant=variant, seed=seed, epoch=rec['epoch'],
                    delay=i, mean=flat['moment_mean_delay'+str(i)], variance=flat['moment_variance_delay'+str(i)]) for i in range(4))
                tables['source_gates_all1600'].append(dict(variant=variant, seed=seed, epoch=rec['epoch'],
                    **{key: value for key, value in rec.items() if key.startswith(('mix_', 'projection_'))
                       or key in ('mixture_gradient_used_parameters', 'projection_gradient_used_parameters')}))
                tables['source_normalization_all9600'].extend(dict(variant=variant, seed=seed, epoch=rec['epoch'], **item)
                    for item in diag['normalization']['records'])
        final = epochs[-1]; best = max(epochs, key=lambda rec: rec['source_val_accuracy'])
        tables['source_final'].append(dict(variant=variant, seed=seed, V=final['source_val_accuracy'],
            worst_RX=final['source_val_worst_rx'], CE=final['clean_ce'], best_V=best['source_val_accuracy'],
            best_epoch=best['epoch'], best_epoch_selected=False))
        tables['source_RX'].extend(dict(variant=variant, seed=seed, receiver=rx, accuracy=value)
            for rx, value in final['source_val_rx_accuracy'].items())
        tables['source_resources'].append(dict(variant=variant, seed=seed, parameters=profile['total_parameters'],
            trainable_parameters=profile['trainable_parameters'], gradient_used_parameters=profile['gradient_used_parameters'],
            resident_state_bytes=profile['resident_state_bytes'], conv_linear_macs=profile['conv_linear_macs_per_sample'],
            fft_calls=profile['fft_calls_per_sample'], inference_batch1_ms=profile['inference_batch1_ms'],
            inference_batch128_ms=profile['inference_batch128_ms'], training_batch128_ms=profile['training_batch128_ms'],
            source_peak_bytes=max(rec['peak_cuda_allocated_bytes'] for rec in epochs),
            profile_clone_peak_bytes=profile['benchmark_peak_cuda_allocated_bytes'], hardware=profile['hardware'],
            torch_version=profile['torch_version'], elapsed_seconds=row['completion']['elapsed_seconds']))
        if variant in VARIANTS:
            diag = row['source_diagnostics']; phys = row['physical_diagnostics']
            tables['source_all720_cells'].extend(dict(variant=variant, seed=seed,
                **{key: value for key, value in item.items() if not isinstance(value, (list, dict))}) for item in diag['groups'])
            tables['source_geometry'].append(dict(variant=variant, seed=seed,
                between_TX=diag['mean_between_tx_centroid_squared_distance'],
                within_TX_RX=diag['mean_within_tx_rx_centroid_squared_distance']))
            flat = input_record(phys['moment_input'], variant, seed)
            tables['public_input_all8'].append(flat)
            tables['public_moments_all32'].extend(dict(variant=variant, seed=seed, delay=i,
                mean=flat['moment_mean_delay'+str(i)], variance=flat['moment_variance_delay'+str(i)]) for i in range(4))
            tables['public_normalization_all48'].extend(dict(variant=variant, seed=seed, **item) for item in phys['normalization']['records'])
            tables['source_phase_all24'].extend(dict(variant=variant, seed=seed, **item) for item in phys['phase_audit'])
            tables['public_cascade_all240'].extend(dict(variant=variant, seed=seed,
                **{key: value for key, value in item.items() if not isinstance(value, (list, dict))}) for item in phys['records'])
            tables['public_isolated_TX_all32'].extend(dict(variant=variant, seed=seed, **item) for item in phys['isolated_tx_changes'])
            control = next(record for record in data['source_controls'] if record['seed'] == seed)
            tables['source_paired_control'].append(dict(variant=variant, seed=seed,
                V_delta_pp=100*(final['source_val_accuracy']-control['accuracy']),
                worst_RX_delta_pp=100*(final['source_val_worst_rx']-control['worst_rx'])))
    counts = dict(source_curves=2400, source_final=12, source_RX=60, source_resources=12, source_all720_cells=720,
        source_geometry=8, source_paired_control=8, source_gates_all1600=1600, source_input_all1600=1600,
        source_moments_all6400=6400, public_input_all8=8, public_moments_all32=32, source_normalization_all9600=9600,
        public_normalization_all48=48, source_phase_all24=24, public_cascade_all240=240, public_isolated_TX_all32=32)
    if any(len(tables[key]) != number for key, number in counts.items()):
        raise ValueError('Incomplete source analysis evidence dimensions')
    for name, rows in tables.items():
        table(evidence/(name+'.csv'), rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout='constrained')
    for variant in selection['candidate_universe']:
        rows = [row for row in controls+data['rows'] if row['resolved']['variant'] == variant]
        for ax, key, label, factor in zip(axes, ('source_val_accuracy', 'source_val_worst_rx', 'clean_ce'),
            ('Source V accuracy (%)', 'Worst source RX accuracy (%)', 'Training CE'), (100, 100, 1)):
            values = np.asarray([[rec[key] for rec in row['epochs']] for row in rows])*factor
            mean = values.mean(0); sd = values.std(0, ddof=1)
            ax.plot(range(1, 201), mean, label=variant)
            ax.fill_between(range(1, 201), mean-sd, mean+sd, alpha=.12)
            ax.set_xlabel('Epoch'); ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0].legend(fontsize=7); axes[2].set_yscale('log')
    fig.savefig(evidence/'source_curves.png', dpi=180); fig.savefig(evidence/'source_curves.pdf'); plt.close(fig)
    inputs = tables['source_input_all1600']; phases = tables['source_phase_all24']
    validation = dict(status='VERIFIED', new_rows=8, new_epochs=1600, new_steps=80000, control_rows=4,
        table_counts=counts, source_selection_recomputed=True, selected_variant=selection['selected_variant'],
        new_candidate_selected=selection['new_candidate_selected'], target_access=False, target_scores_used=False,
        goal_achieved=False, raw_IQ_max_error=max(rec['raw_order1_max_abs_error'] for rec in inputs),
        input_formula_max_abs_error=max(rec['input_formula_max_abs_error'] for rec in inputs),
        whole_constant_phase_max_logit_error=max(rec['whole_logit_max_abs_error'] for rec in phases if rec['received_cfo_hz'] == 0.),
        whole_affine_phase_max_logit_response=max(rec['whole_logit_max_abs_error'] for rec in phases if rec['received_cfo_hz'] != 0.),
        numerical_measurements_scope='Last28 source packets per epoch;full27000V90cells geometry per model;30public packets/model',
        scope='Four CE-trained global coefficients and actual packet-moment residual; no unique TX hardware,TX/RX separation,whole-CFO/RX/LTI invariance claim')
    write(evidence/'source_analysis_validation.json', validation)
    text = '# CVS可学习包内矩残差：完整源实验报告\n\n'
    text += '8个新scratch模型全部完成E200×50；完整审计80000步、1600轮、详细stdout及完整/紧凑JSONL/CSV。实际仅CE、无增强、身份骨干、原L6300/V27000/U56700unused、完整FP32，无权重继承。四个原adaptive控制的完整源曲线终值与本轮独立读回元数据一致。\n\n'
    text += '固定源规则选择 `'+selection['selected_variant']+'`。'+('新候选胜出，冻结后默认4新clean＋32固定控制，共36行独立truth-last评分；本报告不含新测试成绩。' if selection['new_candidate_selected'] else '原控制保留；未选两种新候选clean均为N/A，不追加query，按预登记复用控制已完成的测试。')+'总体目标仍未证明完成。\n\n'
    text += '|候选|四seed源V（%）|最差源RX（%）|固定性能分数（%）|参数|\n|---|---:|---:|---:|---:|\n'
    for variant, summary in selection['source_summaries'].items():
        text += f"|{variant}|{100*summary['source_accuracy']:.4f}|{100*summary['worst_rx_accuracy']:.4f}|{100*summary['score']:.4f}|{summary['parameters']}|\n"
    text += '\n按四seed mean(0.5V+0.5最差源RX)最高选择，完全并列后比较V、最差RX与成本；仅E200权重参加选择，最好轮只作诊断。\n\n|模型|seed|E200 V（%）|最差RX（%）|末轮CE|最好V（%，未选）|最好轮|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for rec in tables['source_final']:
        text += f"|{rec['variant']}|{rec['seed']}|{100*rec['V']:.4f}|{100*rec['worst_RX']:.4f}|{rec['CE']:.6g}|{100*rec['best_V']:.4f}|{rec['best_epoch']}|\n"
    text += '\n![全部2400条源曲线](evidence/source_curves.png)\n\n阴影为四seed样本标准差。全部720个新模型TX/RX/day单元覆盖源V，源RX分层不替代未知RX测试。\n\n'
    text += '## 实际学习与输入修正\n\n保留原包络B、原相位记忆A；I=A+tanh(beta)(O-B)。两个原alpha与两个新增beta均从零开始、只由源CE学习；beta=0保留原adaptive函数。O来自同包加权矩，不是已识别的TX PA系数，不保证最终混合输入正交。\n\n|模型|seed|末轮alpha3|alpha5|beta3|beta5|三阶矩修正相对量|五阶矩修正相对量|公式误差|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for rec in inputs:
        if rec['epoch'] == 200:
            text += f"|{rec['variant']}|{rec['seed']}|{rec['mix_coefficient3']:.8g}|{rec['mix_coefficient5']:.8g}|{rec['projection_coefficient3']:.8g}|{rec['projection_coefficient5']:.8g}|{rec['degree3_moment_residual_relative_change_mean']:.8g}|{rec['degree5_moment_residual_relative_change_mean']:.8g}|{rec['input_formula_max_abs_error']:.8g}|\n"
    text += f"\n完整80000步核对四个系数的真实梯度、raw/tanh更新前后值与连续性；1600轮保留状态和梯度均值、实际末batch28包输入，6400行保留四delay矩。原一阶IQ最大误差{validation['raw_IQ_max_error']:.8g}，实际输入公式最大误差{validation['input_formula_max_abs_error']:.8g}。相对修正量不是识别收益，不用于选模。\n\n"
    text += '## 通信、物理与RFF边界\n\n25MHz下记忆4为160ns、相位项最大历史8为320ns；它们是设计间隔，不是器件测量。包内矩只用同一received256点IQ，无标签、RX/TX标识、跨包或持久拟合状态；完整包矩不宣称流式因果性。输入lift保留相位协变，整网仍可响应CFO。\n\n'
    text += f"公共链覆盖5TX×6RX、240条完整观测、32条孤立TX干预与24条相位测量。八模型最大常相位logit误差{validation['whole_constant_phase_max_logit_error']:.8g}，附加±80kHz最大logit响应{validation['whole_affine_phase_max_logit_response']:.8g}；后者不是CFO不变性误差。保留TX/RX同波形反例，不宣称唯一TX硬件恢复、TX/RX因果分离、任意RX/LTI不变、完整Volterra识别或真实在轨验证。\n\n"
    text += '## 实际成本\n\n|模型|参数|模型常驻bytes|Conv/Linear MAC/包|batch1推理ms均值|batch128训练ms均值|源峰值bytes最大|\n|---|---:|---:|---:|---:|---:|---:|\n'
    for variant in selection['candidate_universe']:
        records = [rec for rec in tables['source_resources'] if rec['variant'] == variant]; rec = records[0]
        text += f"|{variant}|{rec['parameters']}|{rec['resident_state_bytes']}|{rec['conv_linear_macs']}|{statistics.mean(item['inference_batch1_ms'] for item in records):.4f}|{statistics.mean(item['training_batch128_ms'] for item in records):.4f}|{max(item['source_peak_bytes'] for item in records)}|\n"
    text += '\nRTX3090/Torch2.1/完整FP32。新候选202557参数，控制202555，仅新增2，但包内矩和逐元素计算不计入Conv/Linear MAC；实际时延、显存包含执行且受并发影响，不作独占硬件结论。CPU峰值、星载成本及新增传输未测量，记N/A；无SFT/support/新增类。\n\n'
    text += ' · '.join(f'[{name}](evidence/{name}.csv)' for name in tables)
    text += '\n\n[前瞻设计](../../../docs/CVS_LEARNED_MOMENT_RESIDUAL_20261002.md) · [完整日志审计](evidence/source_completion_validation.json) · [源冻结](evidence/source_selection.json) · [分析核对](evidence/source_analysis_validation.json)。只测clean；目标结果不回流结构、矩定义、系数、epoch、seed或选择性重跑，全部负结果保留。\n'
    (report/'report.md').write_text(text, encoding='utf-8')
    print(json.dumps(validation))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    analyze(parser.parse_args().root)
