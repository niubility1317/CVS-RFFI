# 正交空间约束FSCIL-SEI论文一致性优化追踪

来源PDF：`C:\Users\lh594\Downloads\正交空间约束的特定辐射源小样本类增量识别方法.pdf`

本记录只覆盖论文忠实复现层，不引入CVSStage2、satellite/LEO、unknown/open-set声明。根目录`E:\type10-7`不是Git仓库，完成后需镜像到`E:\type10-7\github_publish\CVS-RFFI-repo`。

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|OSC-01|式(6)|扰动伪目标为`\tilde t_i=t_i+\epsilon_i`，`\epsilon_i \sim U(-\lambda,\lambda)`；默认实现不应额外归一化后再声称严格公式一致。|`pseudo_targets.py`、`tests/test_orthogonal_incremental_sei.py`|verified|`pytest tests/test_orthogonal_incremental_sei.py -q`：13 passed；smoke dry-run通过|默认已改为严格加性扰动；如需归一化，必须显式`renormalize=True`。|
|OSC-02|式(38)、式(41)|平均遗忘率按论文`F_t=max A_i^k-A_t^k`和`\bar F=1/T\sum F_t`处理；默认按增量任务数`T`作分母。|`metrics.py`、`tests/test_orthogonal_incremental_sei.py`|verified|`pytest tests/test_orthogonal_incremental_sei.py -q`：13 passed；smoke dry-run通过|默认`forgetting_denominator="incremental_sessions"`；总session分母需显式指定。|
|OSC-03|算法1/算法2超参|dry-run配置应消费论文命名的`tau_s`、`tau_c`、`q`，避免只支持实现别名导致配置与论文断开。|`train.py`、`configs/*.json`、`tests/test_orthogonal_incremental_sei.py`|verified|`pytest tests/test_orthogonal_incremental_sei.py -q`：13 passed；smoke dry-run通过|主配置已使用`tau_s/tau_c/q`；兼容旧别名。|
|OSC-05|固定映射函数`h:C->{1,...,N}`|类别到伪目标的映射应可按论文数据/session给定顺序固定，不能被数值排序静默重排。|`pseudo_targets.py`、`losses.py`、`tests/test_orthogonal_incremental_sei.py`|verified|`pytest tests/test_orthogonal_incremental_sei.py -q`：13 passed；smoke dry-run通过|`assign_base_targets`默认保持输入顺序，`sort_labels=True`仅作显式兼容选项。|
|OSC-06|式(19)|样本锚点正集按论文集合包含同类样本特征；实现不再隐式排除anchor自身。|`losses.py`、`tests/test_orthogonal_incremental_sei.py`|verified|`pytest tests/test_orthogonal_incremental_sei.py -q`：13 passed；smoke dry-run通过|`Ls`分母保持论文负集口径；代码/测试审计提出“正样本也进分母”建议，因与PDF公式不符，已拒绝。|
|OSC-07|式(36)|配置应支持论文符号`lambda_a`，避免敏感性实验写`lambda_a`时被忽略。|`train.py`、`configs/*.json`、`tests/test_orthogonal_incremental_sei.py`|verified|`pytest tests/test_orthogonal_incremental_sei.py -q`：13 passed；smoke dry-run通过|保留`lambda_align`旧别名兼容，主配置使用`lambda_a`。|
|OSC-08|算法2冻结特征提取器|增量阶段应显式冻结特征提取器，只优化新类权重。|`train.py`、`tests/test_orthogonal_incremental_sei.py`|verified|`pytest tests/test_orthogonal_incremental_sei.py -q`：13 passed；dry-run返回`encoder_trainable_after_increment=0`|dry-run仍不是正式session训练。|
|OSC-09|正式增量校准输入合法性|`new_features/new_labels/new_class_ids/prototypes/weights`必须shape和类别一致，避免静默跳过样本。|`losses.py`、`tests/test_orthogonal_incremental_sei.py`|verified|`pytest tests/test_orthogonal_incremental_sei.py -q`：13 passed|补充长度、rank、feature dim和新类集合校验。|
|OSC-10|余弦分类器正式评估可用性|`weight_override`需对齐feature device/dtype，支持增量拼接权重评估。|`model.py`、`tests/test_orthogonal_incremental_sei.py`|verified|`pytest tests/test_orthogonal_incremental_sei.py -q`：13 passed|CPU/CUDA混用时会迁移override到features所在设备。|
|OSC-11|式(38)-(41)|遗忘率按论文差值口径，不做非负截断；`F_bar`默认按增量任务数分母。|`metrics.py`、`tests/test_orthogonal_incremental_sei.py`|verified|`pytest tests/test_orthogonal_incremental_sei.py -q`：13 passed|保留`total_sessions`作为显式替代口径。|
|OSC-04|表2-表7、图8/图9|正式ADS-B/WiFi loader、多session训练、消融、计时和混淆矩阵仍缺失。|`paper_checklist.md`、`findings.md`|deferred|不在本轮实现|需要真实数据路径和GPU正式运行，不能用synthetic dry-run替代。|
