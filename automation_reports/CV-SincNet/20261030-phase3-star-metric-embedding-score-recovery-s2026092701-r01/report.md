# PAMF 度量学习独立评分恢复

## 预登记（2026-10-30）

Run `20261030-phase3-star-metric-embedding-score-recovery-s2026092701-r01`，实际修复code `4211e8981e85fa26e0ce354267043ecb2116530b`。只恢复原 `20261030-phase3-star-metric-embedding-s2026092701-r01` 的 compact_oe 缺失独立评分；原三臂模型/2000训练步/全部predictions/聚合/12τ不动，identity及mainstructure两个已完成score直接引用。原pipeline终态FAILED_PRESERVED和失败日志保留，不把恢复说成旧pipeline成功。原prediction代码efb82a4177e9b59533b7e71408ded6b63fc8733a，原科学主compact_oe_structure::predicted_identity_mahalanobis不变，完整goalACTIVE。

### 已定位故障与唯一修复

旧score在公有模型重建、私有truth之前失败。原compactOE known27000的CPU FP32 vs savedGPU FP32有7行超rtol1e-5/atol2e-6，maxabs2.98768e-6；4tensor/feature均FP32且finite，残差范数最小0.0534883，坏点范数0.0596906至0.0920954，不支持近零塌缩或模型损坏。层参考支持FP32累加差异经残差归一化放大。相同权重转CPU FP64参考后known27000全过原tol、maxabs1.17427e-6，公有novel3150全过原tol、maxabs9.53043e-7；诊断没有读取真TX/scene/阈值或性能。

只将独立参考net和input计算改FP64，checkpoint仍严格4个FP32tensor，产出FP32单位向量比原GPU值。原unit容差1e-5/2e-6及FP64raw/margin容差1e-10均不变，不引入CUDA/回退/宽松阈值。真正部署与saved predictions保持FP32，同一model/aggregate/τ和原已知身份预测不变。score结果run_id/code_commit继续指原prediction，scorer_run_id/scorer_code_commit分别指本次恢复ID/code；verification_precision显式CPU_FP64_reference_from_FP32_weights。

### 数据、顺序和执行

输入仅原compactOE已冻结prediction_root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-metric-embedding-s2026092701-r01/rows/compact_oe/predict`（known27000/novel3150），同G10MIX01 P1/源5已见RX/三日及历史novel，不新增物理/跨新RX/匿名注册证据。既有L6300/OE6134/sourcecontract392005/原F0与DURA祖先按原数据角色复用；L metadata与完整1000step trace只复核原训练权限，无新model/aggregate/阈值拟合。source V/novel features只用于独立复算，private source truth路径 `/home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-representation-s2026092701-r01-release/score_truth/source_novel.npz` 仅全部公有验证后连接；targetU/Q/正式RX24-18/隐藏Us/OE真TX/K读取0。

新独占root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-metric-embedding-score-recovery-s2026092701-r01`，release/CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-metric-embedding-score-recovery-s2026092701-r01-release`；Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，唯一owner `codex-phase3-star-metric-score-recovery-20261030`。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_metric_embedding_score --prediction-root /home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-metric-embedding-s2026092701-r01/rows/compact_oe/predict --novel-truth /home/szu2070436088/2510044040/CV-SincNet/releases/20261029-phase3-star-representation-s2026092701-r01-release/score_truth/source_novel.npz --output /home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-metric-embedding-score-recovery-s2026092701-r01/score --scorer-run-id 20261030-phase3-star-metric-embedding-score-recovery-s2026092701-r01 --scorer-code-commit 4211e8981e85fa26e0ce354267043ecb2116530b --smoke-output /home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-metric-embedding-score-recovery-s2026092701-r01/smoke.json`。CPU-only，CUDA_VISIBLE_DEVICES空，新增GPU worker0，foreign不干预。旧prediction/score目录只读，新smoke/log/score写本run，不覆盖或重发旧run。启动结果UNKNOWN先读回原PID/CWD/cmd/log，不因连接失败再提交。

实际部署g checkpoint加载后、任何known/novel NPZ特征打开前，2条固定合成单位输入做noquery FP64参考smoke，写本root/smoke.json PASS立即继续。所有source公有模型/来源/预算/训练/部署状态/τ/defaulthead/两score与margin复核后才truth-last。newg reference30150+synthetic2=30152，F/G/真实IQ0、neural/model/prediction/τ更新0；CPU参考时延不是部署时延。预期launch.json/logs/score.log/smoke.json/score(results,complete)/observations及runtime；原identity/main complete路径按runtime引用不重算。

仅这一次根因修复；任何失败保留partial，相同数值指纹若重复则停止恢复重试，不扩容差或重训。恢复只补齐原固定矩阵，指标和晋级规则不变，不能依据控制分数换主/调参或访问target。AN/HOW=N/A、完整Phase3 ACTIVE。

验证：8项受影响score测试/23子用例PASS10.50s、2AST PASS；实际冻结checkpoint30150条公有数值回归已证实旧FP32失败/FP64过原tol。一项直接P0/P1问题的唯一针对性复核PASS，不重原core/train/launch suite或全审查；一次新run preflight核原prediction complete和两个旧scorecomplete/新root可用。单实际code/config commit、一个archive SHA比较/compile、实际gcheckpoint noquery smoke、启动PID/CWD/cmd/log及完整独立评分，不增加hash链/授权token/报告审批。
