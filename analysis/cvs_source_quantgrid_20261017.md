# 接收观测格点：完整源域可观测性反证

**ANALYZED／VERIFIED，FAIL_FIXED_QUANTGRID_PREMISE_CLOSED。** 固定 fit L5040 全部完成，0/5040 包满足原共同轴向仿射格点判据；全部72个 TX×RX×day cell 的可观测率均为0%，低于预登记每格95%。固定可观察且有足够幅度的格点理由失败，关闭；不细化格距、放宽容差、换窗/RX/seed/阈值补测，不据此启动量化或dither增强训练。

run_id=20261017-diagnostic-source-quantgrid-fitL-manysig-m1-r01。本轮0模型/训练后checkpoint读取、0网络前向、0新model-epoch、优化/EMA/教师/头拟合更新、0IQ变换或target。主测量420.515955s；独立全量复算0.068117s。固定F＋跨RX同TX／三SIG_TIME／统一Sinc／joint160／NoExplicitCFO／默认cos30正常联合训练与E20预算保持，没有用此诊断替代正式训练收益。

## 唯一问题、数据与实际执行

当前唯一检验的假设是：原源eq1/RMS IQ中仍存在可分辨、跨TX/day稳定且幅度足够的共同轴向格点，能够作为单独接收观测增强的依据。没有预设它来自ADC，未根据目标分数选幅度。完整唯一矩阵、角色、seed、命令/CWD/输出与stop见 [experiment.json](E:/type10-7/automation_reports/CV-SincNet/20261017-diagnostic-source-quantgrid-fitL-manysig-m1-r01/experiment.json)。

数据仅原fitL5040：RX3/4/6/8、day1/2/3、TX0..5、72格各70，split392005。使用已有VALIDATED_ONCE源cache，先核对carrier和effective role IDs，再筛fit4RX。整体cache反序列化包含原L6300以及未使用的V/历史feature字段，但只消费精确fitL5040的x/y/RX/day，heldRX1、U值/隐藏TX真值、V值或旧feature均不用于测量、学习或推断。物理ID只作opaque唯一/角色核对，不解码隐藏标签；没有扩数据或改变原角色。

实际原native args选择wisig_equalized=1；原WiSigCompactDataset(normalize=True)在增强前做RMS归一化。数据集提供未处理和均衡后的256点前导码：[WiSig官方说明](https://cores.ee.ucla.edu/wisig/overview/)。本次缓存不是原ADC码；数据集表示选择与之后新注入practical是否EQ是不同位置。原practical仍residual/post_sync/EQoff G=1/reference30/reflect，六场景/SNR/IQ不改。本次未调用practical。

## 已固定的测度与放弃条件

每包I/Q512实数值共同拟合一个轴向affine格距/offset。distinct≥32；候选来自相邻distinct值差的16个最小值及17个分位数，再除1..32；格距相对component RMS在2^-16至.5，且大于64×float32eps×maxabs分辨界；重估整数码线性拟合后99分位残差≤.02格距，取最大的可拟合候选。只验证这个有界候选集合与共同轴向模型，不宣称数学上排除了任何格点。

预登记先要求所有72cell可观测≥95%；达到后才做18个leave-TX/day的cell log(q/componentRMS)条件均值平方风险描述，至少15/18正折、汇总风险降低≥20%，以及假设独立均匀dither的等效SNR中位数≤50dB。该SNR只是理论幅度代理，不是总接收SNR或实测量化误差，也没有把已处理IQ套入subtractive-dither独立误差定理。

这些只是源统计理由，不能证明身份保持、默认头收益、实际ADC位数或因果RX来源。首项不满足时不得改为只看可观测包或特定RX；其他条件不执行，不伪造数值。全部满足也只支持另登记身份安全检验，非自动E20或Ag晋级。

## 全量实际结果

| 源RX | 合法源包数 | 可观测包数 | 可观测率 | 每个TX/day cell |
|---|---:|---:|---:|---|
| RX3 | 1260 | 0 | 0% | 18格均0/70 |
| RX4 | 1260 | 0 | 0% | 18格均0/70 |
| RX6 | 1260 | 0 | 0% | 18格均0/70 |
| RX8 | 1260 | 0 | 0% | 18格均0/70 |
| 合计 | 5040 | 0 | 0% | 全部72格均0/70 |

每TX840包、每day1680包全部保留。5040条结果均为no_resolvable_affine_grid，没有丢包、重抽或仅选阳性cell。crossfit实际执行0折；RX风险和格距/dither-SNR中位数均为null/不可用，不写成0风险、无限SNR或18个失败折。独立读者已核对所有5040唯一物理ID与原fitL角色、72格/70数量、全部缺失值、全局条件及完整测量前缀，和原summary一致。

此结果否定当前共同轴向格点作为增强依据；没有证明原ADC无量化、均衡一定消除了量化、任意旋转/非均匀格点不存在，或所有接收硬件变化都无效。当前没有实施去格点/再量化/dither，不反演未知TX硬件，也不依据失败细化估计器救援。

## 真实产物与图

[独立analysis_summary](E:/type10-7/local_artifacts/cvs_source_quantgrid_20261017/final/analysis_summary.json)、[全部72分层cell](E:/type10-7/local_artifacts/cvs_source_quantgrid_20261017/final/source_cells.csv)、[完整测量前缀](E:/type10-7/local_artifacts/cvs_source_quantgrid_20261017/final/complete_measurement_prefix.json)。私有逐包ID和测量留原run及local inputs，0原IQ/checkpoint传输。

![全部源cell可观测率](E:/type10-7/local_artifacts/cvs_source_quantgrid_20261017/final/figures/all_source_cell_observability.png)

![全量结果与测量前缀](E:/type10-7/local_artifacts/cvs_source_quantgrid_20261017/final/figures/measurement_outcomes_and_complete_prefix.png)

2PNG＋2同名PDF均实际生成并取回，两PNG已逐张核看。右图95%线仅为总体必要参考，正式条件仍逐cell；这不是训练曲线，本轮没有训练。没有原默认头新预测，不能给出新增accuracy、macroF1、clean代价或跨modelseed性能结果。

## 代码、测试、成本与执行保护

实际运行代码9b8eda74e12c03f385518c139d89f539c98163fa；独立收集/图工具1695f4d65364660d55815e31ad9af115c9d107f8、全量fixture复算5187b5dce47e954b98352cb40c7bca7e5686f8fc，均push后独立remote OID identical。源码package为 [cvs_quantgrid_probe17](E:/type10-7/code/worktrees/phase1_urban_sigtime_20261014/experiments/cvs_quantgrid_probe17/design.py)。精确启动为release根下现有CVS-RFFI Python的 -u -m experiments.cvs_quantgrid_probe17.run，CUDA隐藏、OMP/MKL/OpenBLAS各2；没有direct-file/statistics.py入口冲突。

13协议/输入反例、24已知格距正例、36连续float32/float64及周期负例PASS，compile PASS。新增三组各5040特征的完整合成fixture（无格点、RX关联、仅TX关联）及一项损坏crossfit反例PASS，0真实IQ/CP/NN。开发阶段JSON NumPy标量打印错误与粗格距上界alias在源读取前修正，失败记录保留。独立读者fixture发现数值持平折的NP/stdlib严格布尔差异，仅在风险数值相等容差内接受，全球checks/最终结论仍必须相同，原运行代码和科学阈值未改。

唯一限定P0/P1静态审查PASS；一次普通账户N607预检、一个release archive/一次local-remote SHA比较、remotecompile均VERIFIED。第一次实际运行先合成estimator noquery smoke，PASS立即继续源读取；因为没有模型CP或NN，不冒称真实checkpoint前向smoke。实际PID224834/birth144757302、releaseCWD/moduleargv/CUDA隐藏/CPU2、日志84→378→616字节及逐批增长均独立核对，最终同PID消失、completion存在、failure不存在。原进程自然完成，不停止/重启任何健康任务。collect固定12metadata/stat/log文件，无IQ/cache/model文件，Python未来tarfilter的DeprecationWarning保留，实际收集与复算成功。

主体实际源IQ消费5040包、72cell，0新IQ变换/模型/CP/NN/更新/头拟合/教师/EMA；sourcecache额外反序列化范围已上文披露。主测量420.515955s，独立复算0.068117s，另有合成开发/文件传输/绘图，未单独测全轮墙钟，不能将主体秒数冒充完整任务成本。没有父模型训练成本归属，本轮没有加载或继承模型；F与所有既有E20成本仍保留在原run。

## 对联合目标的含义

最新完成的是一个可执行、全量源域反证及完整交付；不是新训练或性能晋级。当前共同轴向格点理由已失败并关闭；未新增合法可证的IQ Ag，也未证明As组织有额外收益。SupCon混合视图、已关闭时序/session/ordinary等负结果保持，不通过另一名称或阈值重开。

在固定CVS和practical测试下，本轮没有测出任何urban退化缓解；默认头新增提升和clean代价未测，跨seed重现未测。新增Ag独立clean价值、同契约B00/B10/B01/B11与交互、正式clean＋六practical、low urban>60%、suburban接近自身clean、最差未见RX和跨modelseed联合收益均未完成。完整goal保持ACTIVE，不以这次0NN诊断宣告完成。
