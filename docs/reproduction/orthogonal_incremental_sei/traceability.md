# 正交空间约束FSCIL-SEI论文一致性优化追踪

来源PDF：`C:\Users\lh594\Downloads\正交空间约束的特定辐射源小样本类增量识别方法.pdf`

本记录只覆盖论文忠实复现层，不引入CVSStage2、satellite/LEO、unknown/open-set声明。根目录`E:\type10-7`不是Git仓库，完成后需镜像到`E:\type10-7\github_publish\CVS-RFFI-repo`。

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|OSC-01|式(6)|扰动伪目标为`\tilde t_i=t_i+\epsilon_i`，`\epsilon_i \sim U(-\lambda,\lambda)`；默认实现不应额外归一化后再声称严格公式一致。|`pseudo_targets.py`、`tests/test_orthogonal_incremental_sei.py`|verified|`pytest tests/test_orthogonal_incremental_sei.py -q`：12 passed；smoke dry-run通过|默认已改为严格加性扰动；如需归一化，必须显式`renormalize=True`。|
|OSC-02|式(38)、式(41)|平均遗忘率按论文`F_t=max A_i^k-A_t^k`和`\bar F=1/T\sum F_t`处理；当`T`按总session数传入时应把基类`F_0=0`纳入分母。|`metrics.py`、`tests/test_orthogonal_incremental_sei.py`|verified|`pytest tests/test_orthogonal_incremental_sei.py -q`：12 passed；smoke dry-run通过|默认`forgetting_denominator="total_sessions"`；增量阶段均值需显式指定。|
|OSC-03|算法1/算法2超参|dry-run配置应消费论文命名的`tau_s`、`tau_c`、`q`，避免只支持实现别名导致配置与论文断开。|`train.py`、`configs/*.json`、`tests/test_orthogonal_incremental_sei.py`|verified|`pytest tests/test_orthogonal_incremental_sei.py -q`：12 passed；smoke dry-run通过|主配置已使用`tau_s/tau_c/q`；兼容旧别名。|
|OSC-04|表2-表7、图8/图9|正式ADS-B/WiFi loader、多session训练、消融、计时和混淆矩阵仍缺失。|`paper_checklist.md`、`findings.md`|deferred|不在本轮实现|需要真实数据路径和GPU正式运行，不能用synthetic dry-run替代。|
