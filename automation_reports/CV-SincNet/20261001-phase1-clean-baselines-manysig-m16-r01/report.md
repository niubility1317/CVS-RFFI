# 无信道增强基准的 clean 测试报告

状态：LOCAL_VERIFIED，正式测试尚未启动。

## 已冻结范围与授权

用户于 2026-10-01 明确要求“进行基准的测试集测试，给出报告”。立即测试四个已固定基准：原 CVS 身份骨干（native）、CVCNN、普通 1D CNN（real_cnn）、轻量 1D ResNet（resnet1d），各 4 个 model seed（2026092701 至 2026092704），共 16 行。源训练 run `20261001-phase1-cvs-clean-architecture-manysig-m32-r01` 的 32 行已经 E200 完成，基准测试使用其中 16 个固定 E200 scratch 权重。

最新授权调整基准测试时间：不再等待新 CVS 候选完成。新 CVS 的六候选源域联合选择规则保持固定，本次基准评分不能进入候选排序、调参或选择性重训。

## 数据与公平性

同 ManySig 物理源划分：RX 1、3、4、6、8，day 1、2、3；L_s=6300、U_s=56700（未使用）、V=27000，split_seed=392005。源训练固定 200 轮，每轮 50 步，共 10000 步；batch 128，AdamW lr=2e-4、wd=1e-4、cosine 最低 lr=1e-6，FP32，无梯度裁剪。同 model seed 的 batch 顺序一致。仅身份交叉熵，无数据/信道增强、域骨干、伪标签、teacher/EMA 或额外损失。输入 equalized=1、256 点 I/Q、逐包 RMS 归一化。

复用原 VALIDATED_ONCE 的 Phase1 clean 测试 capsule，目标 RX 0、2、5、7、9、10、11，day 0 至 3，全部 6 类统一竞争。预测器只打开 clean.npy 和 opaque ID；不打开 satellite.npy 或 truth。16 行全部预测固化后，独立 scorer 才连接真实标签，报告 overall、各 RX、4seed 均值及样本标准差、Macro-F1、混淆矩阵和原 CVS 对三个对比网络的配对差值。

目标 benchmark 历史上已暴露，属于固定研究评估，不声称全新盲测。目标结果不回流当前 CVS 研发。Phase1 闭集测试没有目标 support 适应或新增类；D92 三阶段及 K×新增类表不适用。

原 CVS 保留原时域/频域/PA 身份网络、cosine scale=30 和原 dropout，域网络不启用；常见网络保留 Linear 分类头。resnet1d 是 6 个 basic block、宽度 32/64/96 的项目轻量基准，不能写成作者原版 ResNet18。对比结果是完整网络比较，不能单独归因于复数卷积或某一物理分支。

## 权重与执行边界

每行预测前核对 completion、initialization、完整物理 source contract、resolved config 与 last.pt 内部字段：E200/10000、同 physical role IDs、同类映射、scratch 无祖先、无 target 接触。来源不一致、未知或受污染即阻断该行。checkpoint provenance 检查不触发重复数据验证。

独占新 run/release，输出防覆盖，launch owner 唯一。仅在实际无进程且至少 12 GB 可用的 GPU 启动一个评估进程；不启动训练，不停止、重启或改变健康 CVS 任务。部分预测失败时不打开 truth，不自动重试覆盖。

## 本地验证

13 个协议检查通过，涵盖固定 16 行与兼容 20 行、缺预测/ID 不匹配关闭 truth、拒绝未选候选/卫星 view/来源污染/物理 contract 不符、已知合成指标与 seed 配对、外部 GPU 占用及 cold process 入口。合成夹具明确标注 SYNTHETIC_TEST_ONLY，绝不作正式实验结果。登记 launch-ready 字段检查通过。

正式结果及独立远端读回在完成后追加。本报告不以源域 V 准确率替代测试集结果。

独立 P0/P1 审查发现原始角色 contract 不含 runtime 输入扩展字段的问题，已在发布前修复：五个物理角色字段严格匹配、原始 num_classes=6、运行输入字段单独硬检查、可选原始扩展存在时严格匹配；payload 对 source 契约、初始化、resolved 的全量相等检查保留。新增原始 schema 回归用例通过。正式 16 行源 metadata 独立只读核实 E200、scratch、相同导出 contract；详见 evidence/source_input_readback.json。
