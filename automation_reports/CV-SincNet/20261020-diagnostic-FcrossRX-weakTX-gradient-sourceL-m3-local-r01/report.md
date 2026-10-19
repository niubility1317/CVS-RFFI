# 原 F：固定弱 TX 的源 L 特征梯度诊断

状态：ANALYZED。三个固定 BASE 的源 L 诊断均完成，结果见 [results.md](results.md) 与 results_summary.json；此项只为下一训练机制取得源域证据，不能当作性能提升结果。

复用原始 F＋CROSS_RX50 的 BASE-s2026092701/02/03，确切模型实现为 31467086c3ced30237f3ac86e91eae68645425d7；来自既定 scratch50、默认头联合训练和 final-only 冻结，同一 source contract，绝不继承为新训练初始化。固定弱 TX 仍为源 V 先前确定的 14-7、20-19（索引 1、3）。

## 方法与范围

每个模型只读取已验证源 L_s 的全部 6300 个 clean IQ 一遍；局部种子 2026102007 固定打乱，同一顺序供三个模型使用，共 50 批（49×128＋28）。不读取 V、U、目标 IQ 或 truth，不拟合任何头，不进行参数或优化器更新。各 TX 源 L 恰为 1050 个。原有默认头只用于读取 CE 信号。

同一次 eval 前向计算 11 个目标的实际部署身份特征参数梯度：六个 TX 的 LS0.01 CE、原浅层/中层 CROSS_RX SupCon、新增条件 MEAN/TAIL 风险、实际 joint160 上两个弱 TX 的跨 RX 对比风险。后两项 joint160 仅是诊断损失，不是拟合分类器。每个类 CE 和 joint160 锚点和除以 1050，批次对比目标按批量/6300 加权。三组 CE_OTHER4、CE_WEAK2、CE_ALL6 由对应类梯度线性组合，不增加 VJP。

全部部署 id_backbone 参数纳入，排除实际默认 cls_head；包含内层卷积、Sinc、参考响应、三个独立 SIG_TIME 局部投影及融合投影。与部署身份输出无关的 domain 辅助骨干不纳入。当前共 231233 个特征参数；无梯度的闲置参数明确保留为零并列出，不掩盖。

报告两种坐标度量：CLIP_PHYSICAL 为 g_p=g_q/25000，与基底全局 clip 的物理 Hz 度量一致；NORMALIZED_HZ 为对 p/Fs 的导数 g_q×1000（Fs=25 MHz、s=25 kHz）。其他参数不变。坐标关系不是叠乘两个优化倍率，此处根本不执行更新。

完整有限池梯度的 cosine、每批 cosine 和逐模块 cosine 分开报告，零范数记为未定义。单一冻结点上的 clean eval 梯度不等同于真实随机 RDrop＋U＋AdamW 位移，也不能证明因果、未见 RX 泛化或目标性能。源码复用原 SupCon 和已提交 tail loss 公式；当前诊断没有两次随机 RDrop，也没有跨批对比对。

## 执行与预算

固定三行 BASE-s2026092701/02/03，每行 0 训练 epoch、6300 源 L 前向、550 次 feature VJP；总 18900 前向、1650 VJP、0 更新。两个物理坐标度量共享相同 VJP。输入复用 source_cache READY/EXACT_MATCH 和既定 source_contract；不重验或重建 physical split。

CWD：LOCAL_RELEASE/cvs_F_weakTX_gradient50_local_20261020_r01；环境 native Windows Python3.10/PyTorch2.10+cu128、FULL_FP32_NO_TF32、threads2。物理 GPU0 一个诊断 worker，顺序处理三个模型；总 GPU worker 含其他 owner 上限 3，派发前空闲 RAM 至少 4.5 GiB，保留健康任务。新建不可覆盖 root LOCAL_RUNS/20261020-diagnostic-FcrossRX-weakTX-gradient-sourceL-m3-local-r01。

启动命令：

```text
LOCAL_PYTHON -X utf8 -u experiments/cvs_f_weaktx_gradient50/runner.py --profile LOCAL_PROFILE/profile.json --source-run LOCAL_RUNS/20261019-phase1-FcrossRX-TXrisk-geometry-e50-clean-local-m4-r01 --source-release LOCAL_RELEASE/cvs_F_crossRX50_weakTX_local_20261019_r01 --output LOCAL_RUNS/20261020-diagnostic-FcrossRX-weakTX-gradient-sourceL-m3-local-r01 --commit RELEASE_COMMIT
```

停止规则：Wrong source role/checkpoint/reference/output, nonfinite derivative, incomplete source budget or any parameter/buffer/gradient/RNG/mode/hook mutation: fail and preserve artifacts. No automatic/performance retry or stopping other tasks.

代码身份为实际 code/config Git commit（发布后记入 release_commit.txt）。预期 execution.json、每行 gradient.json、result.json、completion.json；公开报告只保留聚合梯度统计和完整科学边界。原始 IQ、梯度向量、模型权重、本机进程和缓存资料不提交 Git。

## 已完成检查

7 项权限/标签/来源负例、已知向量 cosine 及零范数 oracle、两种单位映射和 CE 线性组合检查 PASS。匹配真实 checkpoint 的一个合成 no-query 检查覆盖两次前向和重复 VJP：36 个合成样本，全部真实源/目标 IQ 读取为 0；辅助默认 W 梯度为 None，新对比风险有非零 Sinc 梯度；参数、buffer、grad、模式和 RNG 不变。

一次独立 P0/P1 审查发现旧 SourceRoleDataset 无 samples_read 属性，已由诊断侧 CountingRole 独立记录实际 __getitem__ 调用并核对 6300/模型、18900 总计；聚焦计数测试及唯一一次定点复核 PASS。无额外整体或报告审查。checks.json 与 review.json 保存证据。

REJECTED_EXTRA_GATE：不增加通用设计审批、全项目测试、成员 SHA、签名或报告完整性发布门槛。此只读诊断由现有目标授权；下一训练机制仍须独立固定配方、合法源域判断及冻结后的 clean 评分，不能把此处当作目标达成。

执行代码 `83535d3c37226f8b40883f2cad72f60c07e513db` 已推送并核对远端；发布包一次复制 SHA 核对、编译和实际 PID/CWD/GPU/log 增长核验通过。completion_verification.json 确认全部三模型完成、18900 次源 L 读取、1650 次 VJP、零模型/头/优化器更新及零目标访问。进程已结束，完整终态标记和产物均已保存。
