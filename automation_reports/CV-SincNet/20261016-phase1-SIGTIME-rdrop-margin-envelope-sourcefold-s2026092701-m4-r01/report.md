# SIG_TIME＋身份间隔包络 RDrop：源域预登记

状态 RUNNING，四行启动 VERIFIED。本轮只改变 RDrop 的一致性训练目标，4 个模型均从零训练 E20，暂不访问目标测试集。上一版互补掩码完整 E50 主条件失败，已关闭；原版 SIG_TIME＋RDrop E20/E50 三种子正收益及所有负结果保留，见[完整结果](../20261015-phase1-SIGTIME-rdrop-paired-masks-e20e50-clean-manysig-m12-r01/report.md)。

待验证假设：随机子模型可能分别保留对不同竞争身份有用的证据。直接对齐两个输出会让较弱的证据影响较强的一侧。把每个竞争身份对应的较强间隔组合为共同软目标，可能让两个视图学习对方保留的判别证据。它也可能只增加自信或放大噪声，以下对照专门检验这些解释。

对同一个实际增强后的 L 样本，原生独立 dropout 得到 logits z1、z2，合法标签 y。令 a_vc=z_vc−z_vy，包络目标 r=softmax(min(a_1,a_2))，计算后停止梯度。因此 log(r_y/r_c)=max(z1_y−z1_c,z2_y−z2_c)，r_y≥max(p1_y,p2_y)。这是目标分布的数学性质；共享参数反传、AdamW 更新、训练 CE 或泛化性能没有同样的保证。

| 条件 | 一致性目标 |
|---|---|
| FULL | 原版双向对称 KL，两个概率参数均反传 |
| MEAN | 两次概率的算术均值 r，停止梯度；平均 KL(r∥p1)、KL(r∥p2) |
| CONFIDENCE | 使用包络 r_y，竞争类条件分布来自 MEAN；停止梯度后用同样平均 KL |
| ENVELOPE | 竞争身份逐项最大间隔形成 r，停止梯度后用同样平均 KL |

四行均为原双 CE 均值＋一致性损失（alpha=1）＋原 U 损失。CONFIDENCE 与 ENVELOPE 的目标真类概率逐包相同，区别在竞争类之间的关系，避免把单纯提升置信度当作关系学习。该对照不保证梯度范数或优化位移相同。所有分支都只复用同一实际增强 L128，不重抽增强、不改变 dropout 掩码分布，也不增加模型前向次数。

旧 SIG_TIME、RAW_HZ、普通增强、LS=.01、默认 6×160 cosine 分类头 scale30/eps1e−4、AdamW lr.0002/wd.0001、cosine、旧 U 门控及自身 EMA 保持。骨干和网络默认分类头按原流程联合训练；不冻结、替换、重训头，不另训诊断分类器。损失梯度实际进入浅层 t1、中层 t2、SIG_TIME 与默认 W。部署网络与默认推理不新增参数或额外视图。

模型 seed 固定为第一预定义值 2026092701，四行独立 scratch E20=13L＋7U、222 步/轮。训练前真排除源 RX1，fit RX3/4/6/8，day1/2/3 和六 TX 不变；实际 L5040/U45360hidden/V21600，U 真值不可读，V 只监测。固定 final20 用 checkpoint 内默认头预测 1260 个合法 held L，保留逐包 CE、margin、真类概率；RX1 之前已作开发，不能称新盲测。144 个 fit L anchor 由共用 helper 读取但未拟合或参与训练，公开计入成本。

科学资格在首次训练前固定：四行 final L CE≤.5；ENVELOPE 相对其余三行均严格少错且 mean CE 不高；来自两视图不同竞争类的混合证据包比例≥.1；实际 teacher 归一化误差≤2e−6、置信度对照差≤1e−6、真类质量支配违规为 0。PASS 只进入另行预登记的固定 RX3 排除源确认，四臂/seed/E20 不变，确认前不发主测试。FAIL 关闭此配方，不扫 alpha、温度、层、种子或轮次救跑。目标成绩不回流任何选择。

成本 4×20=80 model-epoch／17760 更新；每行 568320 次额外 L 身份前向、396368 次循环 U 观察、1260 次 held L 决策，四行合计 2273280、1585472、5040。训练只继承代码，无任何 checkpoint/teacher/EMA/optimizer 祖先；初始 CP 每行 2 个合成包 smoke 后立即继续。CPU/CUDA 实现检查各 1280 合成包，均为 0 真实 IQ、0 CP 加载、0 优化器步、0 头拟合、0 target。

CPU/CUDA 各 56 个协议负例、5 个科学否决用例通过。包络数学、教师停梯度、标签置换/独立 logit 平移不变性、同预测时零梯度、原 KL 全梯度逐位一致通过。四臂实际 198 个初始张量相同，CE/logits/原生 RNG 相同，FULL 的实际原版参数梯度在浮点容差内一致。一次独立下一运行 P0/P1 审查 PASS，无 P0/P1；证据见 local_verification.json、review.json。本地初次导入命名错误在任何训练前修正，不算模型性能结果。

执行环境 ordinary N607 CVS-RFFI，CWD PROJECT/releases/cvs_SIGTIME_rdrop_margin_envelope_source20_20261016_r01；命令 `CVS-RFFI/bin/python -u experiments/cvs_sigtime_rdrop_envelope20/runner.py experiments/cvs_sigtime_rdrop_envelope20/configs/matrix.json`；唯一 owner `codex/root/SIGTIME-RDrop-margin-envelope-20261016`；不可覆盖 root `PROJECT/runs/20261016-phase1-SIGTIME-rdrop-margin-envelope-sourcefold-s2026092701-m4-r01`。实际 Git 身份写 release_commit.txt，模型代码依赖 b3ac442e2d33a5b315fb3251c30e0ad7083566bc，共用训练器显式指向 FOLD release 的 default-head-only 版本。预期产物见 experiment.json。发布仅一次 ordinary preflight、单 archive 本地/远端 SHA 比较、remote compile，随后验证 PID/CWD/argv/GPU/日志增长；所有 owner 和 preCUDA 进程计入每 GPU≤3，CPU2、loader0。协议、输入、非有限损失、输出碰撞或 worker 技术错误保留产物并停止 pending，不自动重发或停止健康/无关进程；低分不会触发技术重跑。

文献边界：[R-Drop](https://arxiv.org/abs/2106.14448) 已有同样本双随机子模型一致性；[CA-MKD](https://arxiv.org/abs/2201.00007) 已用真值评估教师可靠性；[最大 LLR 共蒸馏](https://www.nature.com/articles/s41592-026-03050-9) 已逐项选择较强证据并蒸馏回模型。最大相对证据合成本身不能宣称首创。本项目的待证点是将它用于同包随机子模型的身份一致性训练，并通过真类置信度匹配对照检验竞争类关系的独立作用。有限检索不能证明不存在相同方法，也不能从蛋白任务的证据推出 RFFI 收益。

<!-- LAUNCH_VERIFIED -->
执行提交 `7827a9a6b04764e690ecaf90457666435017c824` 已推送，远端 OID 独立匹配。单次发布 archive 比较和 remote compile VERIFIED。四行均通过 own checkpoint 无 query smoke；独立 PID/CWD/argv/GPU、实际角色和后续日志增长已核实。首轮实际包络／置信度匹配／归一化审计通过。训练与最终源评分尚未完成，未访问 target。见 launch_verification.json。
