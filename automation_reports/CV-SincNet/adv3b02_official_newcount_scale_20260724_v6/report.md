# ADV3B02官方对比方法新类数量实验v6

- 实验ID：`adv3b02_official_newcount_scale_20260724_v6`
- 日期：2026-07-24
- N607唯一运行所有者：`no_leo_n607_release`
- 状态：`LOCAL_VERIFIED / INDEPENDENT_REVIEW_APPROVED`
- 目标：官方ADV3B02、CSIL、MoPC-HR完整方法；只改变新类数；只有新类样本叠加LEO。

## 矩阵

| 方法 | 新类数 | 说明 |
|---|---|---|
| CSIL | 1、3 | 本轮规模诊断 |
| CSIL | 20 | 论文新增20类，联合既有完整结果 |
| MoPC-HR | 1、3、5、10、25 | 3/5/10/25来自官方代码 |

5接收机×5seed×4个K×3场景；CSIL 200 cells/600行，MoPC-HR 500 cells/1500行，
共700 cells/2100行。

## 冻结边界和验证

- `target_old`：ManyTx原始物理接收IQ，overlay=false，view=`unmodified_received_iq`。
- `target_new`：叠加LEO弱信道，overlay=true，view=`rx_base`。
- policy=`target_old_received_iq_target_new_leo_weak`，core和comparison loader均fail closed。
- parity仅锁定前20新类；每类每场景50条，runner严格要求1000行/场景。
- 25份spec的文件字节SHA和规范化JSON SHA分别25/25通过。
- 真实builder→current v2 comparison loader闭环通过；policy篡改负向测试通过。
- 相关测试`50 passed`；编译和`git diff --check`通过。
- 独立复审：`P0=0,P1=0,P2=0 / APPROVE`。

## N607计划

- env/CWD：`ssr-gpu`；`/home/szu2070436088/2510044040/CV-SincNet`
- run/log：`runs/adv3b02_official_newcount_scale_20260724_v6`；
  `logs/adv3b02_official_newcount_scale_20260724_v6`
- 顺序：哈希/import→真实cache smoke→25 cache→25 parity→base26/base31→两方法smoke→
  700-cell完整矩阵→回收分析。
- 只按P0或两个不同row预测前同一确定性异常停止，不按性能停止。

| candidate_id | 方法 | 新类数 | coverage | 结论 |
|---|---|---:|---|---|
| 待运行 | — | — | 0/700 | `WAITING_REVIEW` |
