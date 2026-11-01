# 投影前源幅度约束恢复器：从零单阶段30轮

**六景 0/6，合并 clean保持，clean 准确率 86.644643%。** cross_rx clean 净变化 +176 条，cross_both 净变化 +2 条。完整目标未达到，保持 ACTIVE 和默认 DURA。[完整性能与分组](analysis.md)。

- run_id：`20261101-phase1-Restore-PowerAnchor-E30-s2026092701-r01`
- group_id：`restoration-raw-amplitude-anchor-single30`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（完整七景与独立复算已核实）

## 目的与对照

与固定FeatureCond仅新增.1*(rawRMS/同条sourcecleanRMS-1)^2，监督单位投影隐藏的尺度方向；结构/输出和latent指导/源顺序/优化器/30-7410匹配，自动原Main七景。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

原 Main168000×七景完整，六景 0/6、合并 clean保持。84单元与完整分组/配对独立复算 PASS，详见 analysis.md。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 单因素与源端依据

与固定FeatureCond/c865相比，仅增加投影前幅度损失.1*(sqrt(Praw/Pclean)-1)^2，P为逐记录时间平均的I²+Q²。Praw来自当前G的raw_restored，Pclean为同条原sourceclean的detached复功率（原约定约为1）。原IQ+.05STFT、.1输出F0指导、.1连续latent回归、全局clip1、共享encoder/四FiLM/单位输出投影、4670178参数和全部推理运算保持。仅约束训练中的尺度方向，不给部署传入clean或新增F0分支。

固定源路径诊断中，原FeatureCond投影前raw平均复功率75.83，实际输出由单位投影归一。原目标没有直接监督此尺度，不能拿raw当作新部署路径。新的幅度项检验该自由度是否影响学习和泛化；这不是已证明Main原因。另一个末轮源梯度诊断并未发现普遍encoder冲突/globalclip激活，因此本次不做PCGrad/分组裁剪。原权重.1对照由被诊断的模型确定，不按Main成绩排序选择；新anchor权重.1固定，无扫描。

全G同seed初值从零单阶段30轮/7410更新，原R07 Ls6300/Us56700、20draw再复用前10份、dataorder2026110102/modelseed2026092701/B256tail24/247steps/AdamW.001-warm247-cos2e-5/WD1e-4均匹配。原F0 clean63000特征及f57源尺度共用，在线1890000条F0特征；幅度项没有额外神经前向。teacher和clean幅度只作loss，G部署只当前IQ；无旧G/coarse/DURA/神经IQ教师祖先，TX/logits/CE/类原型或新身份头均不读取/训练。F0/defaulthead冻结，方法标F-guided。V/OE/novel/目标不训练、早停或选模，OE仅冻结诊断。

固定final30后自动原Main126000+42000两个总体各七景，全42单元预测固定后独立CPU truth-last/原共享锁。raw/DURA同IQ-ID判决精确复用，新G全量推理。逐景至少140345/168000正确=83.538690%，距raw clean86.538690%≤3pp且保持clean，两个子总体clean损伤单列。源幅度约束满足不替代性能，也不取消测试；默认DURA保持。

13项相关检查（新3+原FeatureCond7/FeatureStrong3）与一次P0P1通过。真实Ls2 smoke包含新项，PASS立即完整训练；E01..30只技术恢复/final30固定。推理结构完全相同，静态260546560实数MAC/0FFT-CG复用原profile，排除F0/norm/activation/FiLM逐点/softmax/插值/投影/内存，不是实测时延。原N607 CVS-RFFI/Torch2.1+cu121/FP32-TF32off，GPU4/6 UUID准入free>=12GB/原compute<=2/只新增1条本任务NN/单卡总<=3，其他owner保持。不可覆盖run-root/CWD/代码/命令/停止规则/产物见登记。


## 幅度约束对照已运行（2026-11-01）

`20261101-phase1-Restore-PowerAnchor-E30-s2026092701-r01`，实际代码`42ea957db57d7d90df67dadbeffe11fc014909a0`。VERIFIED / RUNNING：真实源样本检查与新增幅度项通过，同 FeatureCond 实际 E0 逐张量一致，配置仅幅度项和版本身份变化；E01 模型/AdamW/CPU-CUDA RNG 独立 CPU 读回通过，总损失复算确认幅度项仅加权一次。首轮 247 更新耗时 49.944650 秒。GPU 4 准入剩余 24110 MiB、原有 0 条计算进程，仅新增本任务 1 条；PID、目录、命令、GPU 与日志增长核实，其他 owner 保持。4670178 参数及部署前向不变，固定 scratch 单阶段 30/7410 后自动原 Main 七景，当前性能 PENDING；六景各距 raw clean≤3pp、clean 保持仍是主验收，效率为支线，默认 DURA 不变。

## 幅度约束对照七景完成

VERIFIED / ANALYZED。全 G scratch 单阶段30/7410，全部 Main 预测后独立 truth-last；4阶段 exit0，本 run 进程退出。84单元/2604组/1736 raw 配对/1302额外配对复算 PASS，最大指标误差 1.25e-14，raw/DURA 判决差0。完整60轮日志、同E0/源顺序/draw、仅新增幅度项、逐轮总loss及30恢复文件/E30-final一致性核实。完整目标未达到，保持 ACTIVE 和默认 DURA。
