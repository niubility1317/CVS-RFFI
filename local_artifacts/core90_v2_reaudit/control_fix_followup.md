# V2控制修复独立复核

结论：VERIFIED。先前A01(P1)和A02(P2)在当前工作区代码中均已修复，本次窄范围复核未发现新的P0/P1问题。这里只验证控制语义与配置边界，不构成全量回归、GPU训练效果或正式实验准入结论。

## A01：HIGH/LOW证据不能被数值阈值重新解释

- `RELIABLE_HIGH_GAP`且gap=0.2，校准阈值enter=1.0、exit=0.5，完整有效梯度/能力证据和连续两次新观测仍返回NORMAL，不再误入CORRECT。
- `RELIABLE_LOW_GAP`且gap=2.0不会进入CATCHUP。
- 正向对照：LOW+gap=0.2连续确认后进入CORRECT；HIGH+gap=2.0连续确认后进入CATCHUP，证明检查没有通过禁用全部动作取得。
- 实际CPU ProbeV2对2样本线性头执行40步拟合，得到HIGH且stationary=false、plateau=false；沿用原先可触发缺陷的校准数据，两次决策均为NORMAL，原缺陷未再现。拒绝理由为`no_trusted_action_condition`，不是缺失证据/陈旧证据。

## A02：零额外head预算保留correction

- `--game_control correction --game_max_extra_head 0`可正常解析；3份有效source lag/能力证据仍生成V2控制器，catchup_steps=0。
- 该校准控制器对LOW证据连续确认后返回CORRECT。
- 零预算下V1/V2控制器对HIGH证据均不返回CATCHUP。

## 配置边界

- V2的`--game_no_audit`分别与controller、capability curriculum、response tracking、Jacobian interval组合，均明确拒绝为requires source audits。
- B8的`head_lookahead + no_audit + control off`仍可解析。
- 负数`game_max_extra_head`明确拒绝。

## 复核方法与产物

使用已验证的`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8`执行`local_artifacts/core90_v2_reaudit/control_fix_followup.py`，退出码0，脚本断言全部通过，JSON状态VERIFIED。仅运行此独立小型CPU检查，未重复父任务的完整测试；未修改正式源码，未覆盖原始发现JSON。

- 可复跑脚本：`control_fix_followup.py`
- 完整动作/配置拒绝/实际probe证据：`control_fix_followup.json`
- 原始发现仍保存在同目录`control_review.md`及对应原始repro JSON。
