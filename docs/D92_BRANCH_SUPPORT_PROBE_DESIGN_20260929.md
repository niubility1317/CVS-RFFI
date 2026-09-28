# D92 冻结分支的 support-only 增量探查

日期：2026-09-29。该设计只判断冻结分支在合法 support 上是否具有数值非冗余和留出物理样本的预测增量，不产生 query 预测，不启动正式方法比较。设计与参数未读取任何历史或当前 query 成绩。实现范围为纯数学探针及合成验证；真实探查由独立入口管理。

## 1. 已核实输入与固定比较

架构证据是 `E:/type10-7/local_artifacts/d92_upgrade_20260928/fixed_phase1_branch_metadata_20260929.json`。四个 checkpoint 均经 exact loader、完整来源与 SHA 核验，状态 VERIFIED：`lite_d/no_dac/feat_joint`，time/freq/PA/stats 启用，DAC 关闭，`active_defects=['pa']`；t/f/embedding 均为 160 维，joint projection 为 320→160；频率输入 raw_fft、PA 输入 raw_iq；stability/CRRA 关闭，BN 数量为 0。核验没有 forward 或数据读取。实际提取仍必须 eval、冻结所有参数，关闭 dropout/MixStyle 更新。

只使用原始、完整的同一 received IQ，不添加相位、裁剪或重新模拟视图。固定辅助组是融合前 `t_emb/f_emb/pa_local` 三个实际启用出口。禁用 DAC 不作为特征；不再比较 base/feat_cls/feat_pa/feat_imp 等层级重复出口，不按接收机或模型选择分支。一次原生 `return_aux` forward 同时取得 z_id 和三个出口。

定义 `u(v)=v/max(||v||₂,10^-12)`。每个物理样本 i 的固定向量为

\[
A_i=[u(t_i),u(f_i),u(p_i)]/\sqrt3,
\quad B_i^{z}=u(z_i),
\quad B_i^{zf}=u([u(z_i),4u(FFT96_i)]).
\]

FFT96 完全沿用现有 received descriptor。两种背景各比较三个臂：B、[B,B]、[B,A]，机器名称依次为 `z/z_duplicate/z_aux` 与 `zfft/zfft_duplicate/zfft_aux`；维度为 160/320/640 与 256/512/736。拼接后不再次整体归一化，保留原背景尺度。每个臂用同一 ridge=1，不搜索参数。

重复背景是必要的正则控制：在相同截距处理下，[B,B] 加 ridge=1 与 B 加 ridge=0.5 的预测严格等价。因此增加辅助特征的单独 OOF 提高不能排除等效正则变化。在全部块归一化为单位向量时，重复臂和辅助臂具有相同总能量；零或低于 norm floor 的块可能破坏等能量，必须记录实际均值能量差，不以辅助范数重新缩放重复背景。后者会把辅助信息带进控制臂。

## 2. 唯一解析探针

当前 row 有 C 类，每类 K 个独立物理 support。固定目标为 `Y_i=onehot(y_i)-1/C`。每个臂只求解

\[
\min_{W,b}\frac12\sum_{i\in train}\|X_iW+b-Y_i\|_2^2+\frac12\|W\|_F^2.
\]

这是物理样本之和，样本权重均为 1；没有除以 N 后仍保留 ridge=1 的隐式改法。截距不惩罚。所有训练类物理 K 相同，所以所有类物理权重相同。旧/新 membership 只用于诊断，不参与拟合。

仅以 trainfold 计算 X̄、Ȳ、Xc、Yc：`W=(XcᵀXc+I)^-1 XcᵀYc`，`b=Ȳ-X̄W`。纯 float64 Cholesky；D≤Ntrain 用 primal，否则用等价 dual `W=Xcᵀ(XcXcᵀ+I)^-1Yc`。此维度选择是精确线性代数优化，不是科学参数。矩阵最小特征值至少为 1，条件数上界 `1+||Xc||F²`。记录正常方程残差和含截距的真实梯度残差；optimizer steps=0、epoch/LR=null，不伪造迭代日志。

## 3. 分开判断两种增量

数值诊断记录原始各块范数、零块数量、固定特征能量、中心化 stable rank 和数值 rank。数值 rank 阈值为 `max(N,D)*eps64*最大奇异值`；全零 rank=0。这只是算术精度判据，不是信号显著性阈值。小 N 的高维背景往往能插值任意辅助值，因此训练残差小或 rank 没增加都不能证明信息冗余。

条件线性可重构性使用同一个固定 ridge 探针从每种背景 B 预测 A。训练均值和映射只用 trainfold。baseline 分类和 A 重构可以使用联合 RHS `[Y,A]` 共用一次分解；平方损失对输出列可分，所以分类预测与独立分类解完全相同。记录每个 held physical 的 `||Ahat-A||²`，对照 trainfold A 均值预测的误差。合并 OOF 后定义 `R²_linear=1-SSE/SST_trainmean`；分母为零则 null 并给原因。它只说明该固定线性探针的可重构性，不是互信息、非线性不可重构性或 TX 因果证据。辅助分类臂直接用 A，不使用重构残差改造特征。

预测增量记录全部六臂 OOF accuracy、等类 macro accuracy、旧/新 accuracy 及 H，以及同一物理记录的三项正确率差：aux-base、duplicate-base、aux-duplicate。固定 softmax(score) NLL 仅为未校准线性分数诊断，不将 ridge 输出称为后验，不用 NLL 选参数或跨方法晋级。数值非冗余、held 条件重构和 held 分类增量需分别报告；即使共同支持仍不保证 query 改善，更不证明新增 TX 因果信息。

## 4. 物理隔离与 K1

K≥2 时 F=min(K,3)，每类物理 ID 排序后用位置 mod F 分折。全部臂使用相同折；每个 held 物理样本及其所有出口整体移除。每折独立重算所有中心、Gram、cross moment、重构映射和分类头，不借其他 row 的标签、特征统计或拟合状态。折中每类 train n 一致，K/F 不整除时用每折真实 n。

K1 不拟造独立留出：只计算输入和数值结构诊断。folds 为空、OOF/reconstruction/paired 为 null，原因 `K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT`，不拟合全 support 分类器以训练正确率替代证据。K≥2 也只保留 OOF 诊断，不拟合用于部署的全 support 头。

覆盖原矩阵 4 model×4 RX×3 scene×4 K×5 new-count settings×5 support seeds，共 4800 row；实际 new-count 和注册类从既有 support 请求继承，不猜数值。K=1 的 1200 row 仅数值诊断；其余 3600 row 各 3 folds、每 fold 6 次分解，共 64800 次分解。聚合报告所有预定 strata 及缺失/失败数，不按表现挑接收机、分支或 row。

## 5. 输入边界与缓存

已实现入口（实际路径与模型种子由逐行配置提供）：

```text
python tools/export_d92_branch_support_features.py --checkpoint-root SOURCE_ROOT --native-code DIR --source-contract CONTRACT --source-receivers JSON_LIST --seed MODEL_SEED --capsule CAPSULE --output NEW_CACHE --expected-checkpoint-sha256 SHA --expected-capsule-id ID --device cuda:0 --batch-size 32
python tools/evaluate_d92_branch_support_probe.py --support-features NEW_CACHE --capsule CAPSULE --output NEW_PROBE --config COHORT_CONFIG --expected-checkpoint-sha256 SHA --expected-capsule-id ID --expected-model-seed MODEL_SEED
```

manifest 只含 row/model/checkpoint SHA/capsule/split/receiver/scenario/K/support seed/registered classes/old classes 及合法 support ID、class ID。不接受 query、source、ground、teacher 输入，不提供分支选择或正则网格。既有 checkpoint 来源硬检查继续生效，复用 VALIDATED_ONCE capsule，不新增数据重验或签名链。

received reader 只能读取请求的 support IQ。允许有界索引或 mmap gather；不能为了提取 support 先解压/读取包含 query 的整个 IQ 数组。如现有存储无法实现 support-only 读取，报告具体存储边界并修合法 reader，不扩大授权。每 row 的 fit 严格限制于本 row support；某物理记录在其他 row 是 support，不会因此获得进入当前 row fit 的权限。

缓存 schema=`d92_branch_support_features_v1`；`support_branch_features.npz` 包含 ids、物理 class 字符串 labels、z_id/t_emb/f_emb/pa_local 各 [U,160] float32、fft [U,96] float32、checkpoint_sha256、capsule_id、feature_contract_json。缓存 labels 只用于逐 row 一致性检查，不取代 row 支持集合。`support_splits.json` 只保存 split 和 support 角色字段。缓存键为 checkpoint SHA、capsule、physical ID、原始单 view schema；不同 row 可复用冻结出口，不能复用拟合状态。无 query 数组/预测、source 样本或 source feature bank。

## 6. 核心 API 与输出

```python
probe_branch_support(*, z_id, fft, t_emb, f_emb, pa_local,
                     support_labels, support_ids, classes, old_classes=()) -> dict
```

输入仅当前 row 的数组和整数 registry labels；返回纯 JSON 可序列化诊断，不返回可部署 state 或 score 接口。classes 按物理 class ID canonical 排序，预测平局用物理 class ID，support 按物理 ID 排序，避免输入排列影响结果。

顶层字段包括 config、classes、old_classes、k、support_count、fold_count、physical_fold_assignment、numerical、folds、oof、reconstruction、paired、fit_seconds、factorization_count、optimizer_steps=0、persistent_state_bytes=0。folds 每项记录训练/held IDs、train_k 及 stages。每个 stage 记录臂/背景、设计维度、输出维度、实际物理样本数、权重、ridge、分类损失/惩罚/总目标、目标范数、梯度与正常方程残差、实际 solver/分解维度/次数/条件数上界、center/Gram/solve/objective/score 耗时、W/b/临时状态字节；baseline 另记录分开的重构目标。OOF 保存每物理 ID、类、fold、预测、correct、NLL；重构保存每物理平方误差和均值对照误差；paired 保存三组配对正确率差。零分母给 null 和明确原因；非有限计算报错，不输出 NaN。

外围保存不可覆盖的 config/manifest、完整逐 row JSONL、每 fold/stage 紧凑 JSONL/CSV、每物理 support OOF JSONL 及完整分层 aggregate。所有成绩明确标为 `SUPPORT_OOF_DIAGNOSTIC_NOT_QUERY_EVALUATION`。不连接正式 scorer，不生成正式 query prediction，不自动晋级任何臂。

## 7. 成本与实现验证

当前 identity+FFT cache 没有辅助出口，补提取需要每个唯一 `(checkpoint, capsule, physical ID)` 重新执行一次冻结模型原始 forward；不能报本轮零新增计算。未来同次 forward 输出多个分支不增加 backbone forward 数，但仍有拷贝、存储和序列化成本。四个 160 维块加 FFT96 是每唯一物理样本 2944 数值字节；另列 metadata、压缩文件实长和已有模型包实长。新增 source 传输为 0。无需地面摘要。

每 fold 六个解析求解，无参数搜索。primal/dual 分解阶数为 min(Ntrain,D)，求解需保存 O(ND+min(N,D)²+D·output_dim) 临时数组；baseline 联合输出维度 C+480，其他臂 C。分类 W/b 实际字节为 `8*(D+1)*C`，重构联合头按 C+480 计；丢弃临时头不意味着零拟合内存。记录真实唯一读取 ID、IQ bytes、forward physical count/batch calls、export/cache-write 耗时、每阶段拟合/打分/写日志耗时、分解次数和阶数、峰值 RSS、实际 CPU/BLAS/GPU。当前设计不填写未测秒数。

合成验证覆盖：展开平方目标与解的梯度；primal/dual 等价；联合 RHS 分类列与单独求解一致；duplicate 与背景 ridge=0.5 等价；held physical 扰动不影响该折 train 状态；类/样本排列等变；K1 所有 held 指标为 null；零/近零/finite/shape/labels 负测；真实临时字节与 JSON `allow_nan=False`。常量辅助或重复辅助的训练可拟合性不作为新增信息的测试结论。

实施状态：`code/cvsrffi/d92_branch_support_probe.py`、固定 JSON 配置及 21 项合成测试已完成，测试全部通过。固定配置包含六臂，不接受外部科学参数覆盖。函数返回的是完整诊断字典，不保留拟合头。

本机 `ssr-gpu` Python、2 BLAS threads 的一次纯合成 C=26、K=20 测量：外层 0.344360 s，核心 0.343165 s，其中五次数值谱诊断 0.111570 s；18 次 Cholesky，最大分类梯度残差 1.55×10^-13，最大临时联合头 1,040,336 B，持久分类状态 0 B，完整 JSON 878,635 B。输入由 seed=20260929 的标准正态独立生成；这是单次实现成本测量，不是实际数据成本或性能证据，未计真实 frozen forward。未启动真实探查，未读取 query 成绩。
