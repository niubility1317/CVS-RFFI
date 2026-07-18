# D59 full/block SPD几何中点追溯与预注册

## 1.问题与单一机制

D43证明full auto-shrinkage协方差保护最低新类与rain旧类，而3-block协方差提高聚合old/new/H、旧类floor和混淆，但硬置零全部z160/FFT96/RF32跨块项会损伤最低新类。D44–D46在score域融合两套head；D55–D58又证明类别截距、混淆流和类别幅度校准不能稳定外推。D59不再修改类别幅度、类别截距或类别权重，只在所有类别共享的协方差域插值一次。

令`F`为D42的完整等先验auto-shrinkage协方差，`B=blockdiag(F)`为z160/FFT96/RF32三块协方差。D59使用二者的SPD仿射不变几何中点：

`G=B^(1/2)·(B^(-1/2)·F·B^(-1/2))^(1/2)·B^(1/2)`。

最终仍为等先验LDA：`w_c=G^(-1)μ_c`，`b_c=-0.5μ_c^T G^(-1)μ_c+log(1/C)`；随后仅删除对所有类别相同的公共仿射项。几何中点无可调权重，保持SPD，交换`B/F`时得到同一几何位置；所有类别共享同一个`G`，因此不会引入D58式类别logit比例漂移。

## 2.协议与边界

- 固定receiver`20-1`、seed`713101`、K10/new5、clear/low-elev/rain×5个physical-rank held折；实际outer fit K8。
- 复用匹配`VALIDATED_ONCE/p2_min_v1`的同一received-IQ capsule，不重建、不重验数据。
- support-only拟合；outer-held只在state冻结后评价；query/truth/role/count/quota/global reassignment全部不可达。
- `F`与`B`仅由当前合法support和support label估计；不读取receiver、场景、匿名handle、old/new角色或outer-held结果。
- K1、rank0或零残差严格回退D42单位协方差；不构造额外物理样本。
- 不扫描geodesic位置、ridge、floor、block、threshold、rank、temperature或任何类别/场景参数。
- query部署面仍是一套int8 residual coefficient＋FP16 intercept线性state，对全registry独立argmax。

## 3.实现不变量

1. `F`、`B`、标准化矩阵`A=B^(-1/2)FB^(-1/2)`与`G`必须对称、有限且严格正定。
2. `G`必须满足几何中点Riccati闭包`G·B^(-1)·G≈F`；报告相对Frobenius残差。
3. `B`与`F`相同时`G`逐数值容差退化为同一矩阵；类别置换不改变共享协方差且只置换输出行。
4. 公共仿射删除前后FP32 support argmax一致；输出shape、finite、共享尺度与单state闭包必须通过。
5. 记录full/block/midpoint的特征值范围、条件数、跨块Frobenius能量、midpoint距两端的仿射不变距离及额外资源。

## 4.预注册判门

D59是development probe，强制identity selection、禁止selected-only full-K10、禁止正式性能声明。相对当前最强合法development点D46，进入下一阶段必须同时满足：

1. 105/105行、query0、lifecycle/source/ground/state/resource/artifact闭包全部通过；
2. before/final int8-FP32 argmax变化与pairwise margin翻转均为0；
3. 聚合before-old≥92.22%、after-old≥81.67%、seen-new≥84.67%、同rowH≥82.33%，forgetting≤10.56pp；
4. mean joint floor≥23.33%，最低before≥80.00%、最低after≥53.33%、最低new≥73.33%，且`最低after/最低new/joint`至少一项严格改善；
5. clear、low-elev、rain各场景的before/after/new/H/joint不低于D46且forgetting不高于D46；
6. final old→new/new→old/new-new不超过D46的25/8/15；
7. 15个outer held预测至少1个不同于D46，否则判为无新信息。

即便上述门全部通过，D59也只能进入另行正式候选与封闭开发验证，不能直接运行125或声明达标。若失败，停止D59，不做几何位置或ridge扫描；D60必须换成新的共享几何证据机制。报告必须完整给出7候选、3场景、11类、15fold、混淆、量化、资源和artifact，不得只写缺陷。

## 5.文件与验证

- 实现：`code/scripts/probe_d59_full_block_spd_geodesic_midpoint.py`
- 单测：`tests/test_probe_d59_full_block_spd_geodesic_midpoint.py`
- 输出：`automation_reports/CV-SincNet/d59_full_block_spd_geodesic_midpoint_probe_20260719/full_block_spd_geodesic_midpoint`
- 环境：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，本地串行，`device=auto`。
- Git：只暂存D59精确文件；执行前建立detached clean worktree并记录脚本SHA。N607本轮不访问。
