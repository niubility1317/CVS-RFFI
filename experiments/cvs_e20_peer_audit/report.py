"""Human-readable full source diagnostic, no source or target model selection."""
from . import design as d


def pct(value):
    return 'N/A' if value is None else f'{100*value:.4f}%'


def number(value):
    return 'N/A' if value is None else f'{value:.6f}'


def build(value, completion):
    assert value['status'] == completion['status'] == 'SOURCE_DIAGNOSTIC_COMPLETE'
    assert value['target_access'] is False and value['fitted_parameters'] == 0
    lines=['# E20 独立种子教师的源域互补性', '',
           '复用三个原 E20 的 99,900 条源样本模型读出；没有新模型前向、训练、U 身份标签或目标访问。LOO2 排除被比较学生，采用其他两个模型的均匀概率均值。ALL3 是一个三模型教师；其三个学生比较不能当作三个独立重复。', '',
           '所有置信阈值均为预登记的 0.95；一致门控要求两位独立教师预测同类且各自置信度均至少 0.95。L 是已用于训练的有标签样本；V 只作固定读出，全部源 RX 已被父模型看到。下述结果不能替代未知 RX 测试。']
    for role, values in value['roles'].items():
        lines += ['', f'## {role}：{values["count"]} 个物理样本', '',
                  '|学生种子|自身准确率|LOO2 准确率|自身 NLL／ECE15|LOO2 NLL／ECE15|修复／破坏|自身高置信错误／被修复|',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for seed, row in values['students'].items():
            own=row['own_P0']['overall']; teacher=row['LOO2']['readout']['overall']
            pair=row['LOO2']['paired']['overall']['overall']
            lines.append(f'|{seed}|{pct(own["accuracy"])}|{pct(teacher["accuracy"])}|{number(own["NLL"])}／{number(own["ECE15"])}|{number(teacher["NLL"])}／{number(teacher["ECE15"])}|{pair["fixed"]}／{pair["broken"]}|{pair["high_confidence_original_errors"]}／{pair["high_confidence_original_errors_fixed"]}|')
        all3=values['ALL3']['readout']['overall']
        overlap=values['error_overlap']['overall']
        lines += ['', f'单个 ALL3 教师准确率 **{pct(all3["accuracy"])}**，NLL={number(all3["NLL"])}，ECE15={number(all3["ECE15"])}。三个模型都错误的物理样本有 {overlap["all_three_wrong"]} 个；“至少一个模型判断正确”的不可部署 oracle 上限为 {pct(overlap["joint_oracle_accuracy_upper_bound"])}。', '',
                  '### 接受与改判风险', '',
                  '覆盖率分母为本角色全部物理样本；修复／破坏只统计当前接受子集。改判包括错类变另一错类，不能把改判全部当作纠错。', '',
                  '|学生种子|门控|接受数／覆盖率|其中改判数|教师改判正确率|学生原判断正确率（改判子集）|修复／破坏|接受错误率|',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for seed, row in values['students'].items():
            for selection in ('all','teacher_q095','peer_consensus'):
                p=row['LOO2']['paired']['overall']['selections'][selection]
                lines.append(f'|{seed}|{selection}|{p["count"]}／{pct(p["coverage"])}|{p["changed_count"]}|{pct(p["teacher_accuracy_when_changed"])}|{pct(p["student_accuracy_when_changed"])}|{p["fixed"]}／{p["broken"]}|{pct(p["selection_risk"])}|')
        lines += ['', '### 学生高置信错误中的可纠正部分', '',
                  '|学生种子|门控|被接受的高置信错误|修复数|仍错误数|', '|---|---|---:|---:|---:|']
        for seed,row in values['students'].items():
            cohort=row['LOO2']['paired']['overall']['student_initial_cohorts']['high_error']
            for name in ('all','teacher_q095','peer_consensus'):
                p=cohort['selections'][name]
                lines.append(f'|{seed}|{name}|{p["count"]}|{p["fixed"]}|{p["persistent_error"]}|')
        lines += ['', '### 原模型错误重叠', '',
                  '|模型种子对|双方均错|一方或双方错|错误 Jaccard|左错右对／右错左对|', '|---|---:|---:|---:|---:|']
        for p in overlap['pairwise']:
            lines.append(f'|{p["left_seed"]}／{p["right_seed"]}|{p["both_wrong"]}|{p["either_wrong"]}|{pct(p["error_jaccard"])}|{p["left_wrong_right_correct"]}／{p["right_wrong_left_correct"]}|')
        lines += ['', '### 接收机、日期和身份分层', '',
                  '|维度／层|学生2701|LOO2 对2701|学生2702|LOO2 对2702|学生2703|LOO2 对2703|ALL3|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for dimension,strata in values['ALL3']['readout']['by_stratum'].items():
            for label,stat in strata.items():
                cells=[]
                for seed in d.SEEDS:
                    student=values['students'][str(seed)]
                    cells += [pct(student['own_P0']['by_stratum'][dimension][label]['accuracy']),pct(student['LOO2']['readout']['by_stratum'][dimension][label]['accuracy'])]
                lines.append('|'+dimension+'/'+label+'|'+'|'.join(cells+[pct(stat['accuracy'])])+'|')
    lines += ['', '## 复算证据和限制', '',
              f'CPU 聚合耗时 {completion["seconds"]:.2f} 秒。三份源预测的实际角色 PID 与原契约相同，跨模型顺序、标签、接收机和日期一致，L/V 不交。JSON 保留全部置信分箱、四种学生初始分组及各层的固定门控细目；空子集指标为 null。', '',
              'E20 来自原始从零纯 CE、旧门控与 cosine 日程的固定 final20。用户在已暴露轮次探索后指定 E20 为后续基线，后续目标测试不能宣称为新的盲测。本诊断只提供合法源证据，不借目标排名选种子。', '',
              'LOO2 三个配对共用物理样本且共享教师，不能当作独立三次试验；ALL3 只计一套教师输出。教师在 L 上的结果包含训练内乐观性，V 更适合检查固定监督质量，但仍不是未知 RX。oracle 使用真值只描述互补潜力，禁止作为伪标签生成规则。', '',
              '本诊断没有产生新身份模型，没有新的 clean 成绩，84% 目标仍未完成。下一步若训练独立教师变体，须保持 U 真值隐藏，单独记录教师计算成本并按预登记固定方案完整测试。']
    return '\n'.join(lines)+'\n'


if __name__ == '__main__':
    result=d.read(d.BASE/'analysis.json'); completion=d.read(d.BASE/'completion.json')
    (d.REPORT/'analysis.md').write_text(build(result,completion),encoding='utf-8')
    print('VERIFIED source-only peer report')
