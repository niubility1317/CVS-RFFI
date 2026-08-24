# CCOI-PA-V2修复与最小矩阵执行计划

> 执行日期：2026-08-25
> 基线：冻结`ADV3B02_CORE90_SOFT_E200`，沿用CCOI-PA的C0–C4同容量矩阵
> 原则：不改变Phase1数据权限、source-role划分、checkpoint、场景、seed和训练预算；只修复V1已由日志和artifact证实的问题。

## 任务1：把已知问题固化为失败测试

目标文件：

- `code/tests/test_phase1_ccoi_pa_runner.py`
- `code/tests/test_phase1_ccoi_pa_scorer.py`
- `code/tests/test_ccoi_pa_views_and_challenge.py`

验收：测试能够分别暴露嵌套接收机元数据丢失、无效receiver被评分、融合公式不符设计、码本集中缺少直接约束。

## 任务2：实现V2最小修复

目标文件：

- `code/train_phase1_ccoi_pa.py`
- `code/cvsrffi/ccoi_pa.py`
- `code/score_phase1_ccoi_pa.py`

实现：

1. 递归读取WiSig批次`extra=(domain,meta)`中的原始`rx_i`，写prediction时对负值硬失败。
2. 算子头以自身分类目标训练；仅在`V_cal`上估计去中心RMS尺度，并在包含`alpha=0`的冻结网格中选择凸融合系数；并列时选更小alpha。
3. 用“最小有效码比例+最大单码均值概率”铰链正则替代强制全均匀目标，保留码本利用率诊断。
4. sidecar和manifest记录V2 schema、`fusion_alpha`和`fusion_scale`。

## 任务3：本地验证和一次正确性检查

在`ssr-gpu`中串行运行全部CCOI聚焦测试、语法编译、dry-run和真实checkpoint无query smoke。随后只检查会直接导致协议越权、覆盖输出、不能启动或不能产生合法prediction的P0/P1问题。

## 任务4：版本化、发布和N607验证

只stage本次文件，提交并自动push，独立比较远端分支OID与本地`HEAD`。建立不可覆盖run ID和最小预登记报告；N607只传一个release归档并比较一次本地/远端SHA，完成远端编译后让launcher的smoke通过即继续C0–C4。

## 任务5：监控、评分和判断

启动后检查一次PID/CWD/cmdline/GPU/log增长。运行期间只因协议、安全或确定性技术失败停止，不因低性能停止。prediction完整后由独立scorer连接truth，报告clean、三类LEO、receiver-floor、码本、融合和holdout诊断；按V1预注册门槛判断是否晋级。
