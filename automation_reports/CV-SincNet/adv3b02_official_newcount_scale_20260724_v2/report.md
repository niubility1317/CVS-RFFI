# ADV3B02官方CSIL/MoPC-HR新类数量扩展实验v2

## 基本信息

- 实验ID：`adv3b02_official_newcount_scale_20260724_v2`
- 日期：2026-07-24
- 操作者：Codex主代理；N607发布子代理`no_leo_n607_release`
- 当前状态：`LOCAL_VERIFIED / PREREGISTERED / NOT_LANDED`
- 目标：减少新类数量并覆盖论文增量数量，正式新类support/query均叠加LEO弱星地信道。
- 前序失败run：`adv3b02_official_newcount_scale_20260724_v1`

## v1失败继承与唯一修复

v1已封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
25份spec哈希通过，但12次可观测缓存尝试均因冻结ManyTx路径不存在而失败，
cache成功0，parity/base/smoke/full均未启动。

只读服务器证据确认实际文件为：

`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManyTx.pkl`

v2仅修正以下内容：

- run/log/cache/parity/plan路径中的run ID由v1改为v2；
- `manytx_pkl`补上`CV-SincNet/`路径段。

方法实现、类集合/顺序、receiver、seed、K、LEO场景、CSIL base26、
MoPC-HR base31、优化参数、官方`drop_last`和矩阵规模均不改变。

## 冻结矩阵

| 分支 | 方法 | 新类数 | 论文口径 | base容量 | cell | 场景行 |
|---|---|---|---|---:|---:|---:|
| CSIL-reduced | CSIL官方仓库语义 | `1,3` | 既有正式`new20`覆盖论文每增量20类 | 26 | 200 | 600 |
| MoPC-paper-scale | MoPC-HR官方仓库语义 | `1,3,5,10,25` | `3,5,10,25`对应官方增量数量 | 31 | 500 | 1500 |
| 合计 | 两方法 | — | — | — | 700 | 2100 |

- receiver：`20-1,3-19,7-14,7-7,8-8`
- seed：`713101-713105`
- K：`1,5,10,20`
- 场景：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`
- CSIL smoke：`new1/K1,new3/K20`
- MoPC-HR smoke：`new1/K1,new25/K20`
- 不运行无LEO矩阵。

## 固定类列表

- 旧6类：`14-10,14-7,20-15,20-19,6-15,8-20`
- 新25类：
  `1-16,1-18,18-10,14-11,8-3,18-8,10-10,16-19,20-12,4-10,13-14,2-5,1-8,19-13,19-9,3-8,19-8,11-19,2-16,19-6,13-19,18-14,20-4,20-16,11-10`
- 所有较小新类集合使用严格嵌套前缀。

## v2本地规格

- release目录：
  `paper_reproduction/configs/adv3b02_official_newcount_scale_20260724_v2_release`
- cache spec：25份
- cache build命令：25条
- parity命令：25条
- scope：`external_comparison_registered`
- ManyTx：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManyTx.pkl`
- target cache root：
  `runs/adv3b02_official_newcount_scale_20260724_v2/target_cache_new25`
- parity root：
  `runs/adv3b02_official_newcount_scale_20260724_v2/cache_parity`
- reference root：
  `runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/target`

## 发布硬门槛

1. 25份spec规范化哈希全部匹配；
2. 25个cache set全部生成；
3. 25个parity收据全部PASS，旧6+前20逐项sample ID和信道后IQ哈希一致；
4. CSIL base严格为容量26；MoPC-HR重建base严格为容量31和classifier`[31,160]`；
5. 两方法smoke的prediction/scorer/cell receipt全部完整；
6. 真实smoke收据生成formal authority后才允许700cell矩阵；
7. P0单次，或两个不同row在prediction前同一确定性异常指纹时，
   停止dispatch并终止身份匹配的run-owned进程组；
8. 不因低准确率或其他性能值提前停止；
9. fresh-run自动重试未授权。

## 预期路径

- N607 cwd：`/home/szu2070436088/2510044040/CV-SincNet`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- run root：`runs/adv3b02_official_newcount_scale_20260724_v2`
- log root：`logs/adv3b02_official_newcount_scale_20260724_v2`
- base31：`runs/adv3b02_official_newcount_scale_20260724_v2/base31/official_repo_base_state.pt`
- CSIL输出：`runs/adv3b02_official_newcount_scale_20260724_v2/csil_reduced_leo`
- MoPC-HR输出：`runs/adv3b02_official_newcount_scale_20260724_v2/mopc_paper_scale_leo`

## 本地验证和审查

沿用v1同一release实现：

- release实现commit：`3b8e9988f2213df1435b4a63e5e88b0e7b77a8ff`
- 报告commit：`5d0a7fd0efbc99bd1d3c9f30fc566a3140f14861`
- 相关测试：`41 passed`
- Python编译：PASS
- `git diff --check`：PASS
- 独立release审查：`P0=0,P1=0,P2=0 / APPROVE`

v2需对修正后的25份spec重新完成规范化哈希、schema、ManyTx实际路径和
“除run ID/ManyTx路径外无语义漂移”检查，并单独Git提交后方可发布。

## 结果表占位

| candidate_id | 方法 | receiver | 新类数 | K | seed | 场景 | old_acc | seen_new_acc | H_old_new | forgetting/per-class old | loss摘要 | coverage | 结论 |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---|---|---|---|
| 待运行 | — | — | — | — | — | — | — | — | — | — | — | — | `NOT_ANALYZED` |
