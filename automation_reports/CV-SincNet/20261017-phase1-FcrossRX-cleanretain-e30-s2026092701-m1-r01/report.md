**当前状态：RUNNING／启动 VERIFIED。新行已完整完成 E1，尚无新正式成绩。**

# 全源 F：保留普通监督的 practical 配对，从零 E30

状态：LOCAL_VERIFIED，尚未发布、训练或产生新成绩。整体联合目标 ACTIVE。

用户指定参考聊天 01a1427e-49c0-7840-a84d-5bb38d5a8164 的 clean 保持路线。已读取该聊天及实际报告、launcher/model/config。多轮复用 B10 命名，当前最符合描述的是 clock-E30 的 B01，而非其中 Ag-only B10：原 B01 clean85.522024%、mid62.930952%、low51.576190%；完整 As clean82.963690%、mid64.371429%、low53.447024%。这是同一 seed、完整源 E20＋10 的公开探索结果，不能冒充 scratch30 或跨种子证据。clock Ag 与后继固定 FIR 均未带来增量，保持关闭。新研究参考有用的监督组织，不加载按目标成绩选中的权重，也不称新的独立盲测。

本轮唯一问题：该75%普通视图／25%完整 practical监督组织，迁移至当前 scratch E30 后，是否能保持 clean 并保留原 practical 的 urban 适应。此轮是必要的配方迁移检验，未提出新的物理 Ag、特征对齐、分类头或部署模块。未发布的 P36 梯度草稿暂置，不成为启动门槛。

## 固定实施

新行 R25-s2026092701，零父训练：30轮=20L＋10U，222步/轮，6660更新。保持 F：三路SIG_TIME_X3、Sinc CENTER_25K/CENTER_UNITS、NoExplicitCFO、必要包内补偿、真实joint160、原6×160 cosine默认头/scale30，900722参数/206状态；骨干与头联合更新。原AdamW峰值.0002/min.000001、wd.0001、native cosine30、LS.01、RDrop alpha1、SupCon原层位warm5/ramp10/max.1、旧U门控与自身EMA不变。

全部原 RX1/3/4/6/8、day1/2/3、6TX，L6300/U56700隐藏TX/V27000、split392005、7/63/30物理ID、VALIDATED_ONCE保持，无RX留出或内部L_dev。V不训练；不解封U TX。新学生/EMA从零建立，无E20/E50/E200父权重。兼容smoke读取既已核实全源FINAL20，仅2合成包后丢弃；另对既有全源C0 FINAL30作冻结源读出，不是初始化。

每次实际 L128 保留共有 ordinary128；第二半128里，按(batch_position+global_step)%2==1的64个位置，从同一raw物理IQ只调用一次原 native practical，其余64位置保留同一ordinary。只生成实际监督的64 satellite，无拒绝/高置信筛选；不反演TX效应、不额外叠ordinary噪声/多径。两次RDrop使用同一生成256IQ；两半CE/KL各归一化再权重.5，合计ordinary.75/satellite.25，SupCon只用原第一半。4L网络包前向/物理样本，3409920 L NN；实际As426240包。原同物理样本标签、RX/day context切片后finally恢复，不依TX定义session。记录真实As配置、原模块__file__、全部物理序列、状态覆盖、分开CE/KL曲线与实际参数变化。

25MHz/IQ256、dataset原RMS，不在As后恢复RMS。原residual/post_sync、EQfalse即G1、refSNR30、fc2.45GHz、reflect、接收质量适配／相位跟踪／RXIQ补偿／AGC依赖保留；62tap残余核、24点数值延迟保持。新增噪声比不称总SNR，white-input功率代理保持。训练stream202610172201、ordinary私有20261016051、RX会话2026092701；训练随epoch/view刷新，源读出重用原独立两draw；正式固定IQ/测试SNR/边界/难样本全部保持。实际导入为已冻结 release 的 experiments/adv3b02_xuc/code/leo_practical，而非顶层code。

## 对照、源判定和正式收尾

只新增R25，不重跑健康/已完成行。C0复用 P25 E30 B00 ordinary精确重复，AS50复用 P32 E30 B01 practical(raw128)，均同seed、完整源、scratch30/6660/4L/原U。P25 B01是As(ordinary)，不误当raw-As匹配参照；P32 B00是raw副视图，不误当ordinary-repeat C0。历史E20＋10 B01仅配方背景，不混入本轮主比较。

新R25与既有AS50原生args应仅run_id/candidate_id/output_dir不同；C0仅另有sat_view_prob差异。正式前保存完整V clean和同一V1800两draw×三训练场景读出。源保护条件固定：fullVclean≥C0−.25pp，且两个draw各mid/low urban≥AS50−2pp；任一失败则本轮迁移保护假设不成立，不调份额/幅度/阈值/seed或追加预算救跑。不把通过解释为Ag有效、low>60%或联合成功，不自动扩种子。源结论在新正式预测前落盘；固定新行无论科学判定均完成已登记正式测试。

FINAL30冻结后，自动逐row完成原clean＋practical_high/mid/low_suburban/high_urban/mid_urban/low_urban，各168000包；全部7视图logits/decisions固定后exec独立CUDA空CPU scorer才读取truth。报告accuracy、urban均值、Macro-F1、最差RX及全部RX/day/TX，suburban绝对值/自身clean差、同包修复/新增错误、固定C0 clean正确集保持率、源state/quality、默认头margin/30与完整训练曲线。没有Ag因子，不能编造四格交互。low>60%、clean净收益、suburban各距自身clean≤2pp仍为未完成的探索目标；单seed不证明复现。

## 执行与保护

命令：/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_cleanretain_E30_20261017_r01/experiments/cvs_cleanretain37/runner.py。CWD=/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_cleanretain_E30_20261017_r01。唯一owner codex/root/cleanretain37。不可覆盖run=/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-cleanretain-e30-s2026092701-m1-r01；log=/home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-cleanretain-e30-s2026092701-m1-r01/logs；本地=E:/type10-7/local_artifacts/cvs_FcrossRX_cleanretain_E30_20261017。逐row和六类seed在experiment.json维护。

N607普通账户/现有CVS-RFFI/FP32/CPU2，每GPU所有owner合计最多3进程、启动需12GB可用；不干预其他健康任务。一次只读preflight/一个归档一次SHA比较/远端编译；launcher首步真实CP无query smoke，然后独立目录合成真实native64As和F默认头/骨干梯度验证512NN、0更新，PASS立即继续。审查发现合成验证会提前创建训练source目录的P1已本地隔离a.output_dir到BASE/synthetic/native_execution；末尾独立断言真实source仍不存在。

技术停止只限数据/来源/权限错误、活动loss/grad/prediction非有限、输出冲突或不能启动/合法预测评分的系统故障；保留产物，无自动重试，不因低成绩中止。若SSH结果不明，独立读回原PID/产物后再行动。

成本分开报告：新训练30model-E/6660更新/3409920L NN；父训练0；既有两个控制各30E/6660为历史复用成本。新source学生/自身teacher/评估实际counter、As426240、source新读出2×12602NN、兼容2NN、随机模型合成512NN、formal1176002NN及墙钟分别保存，不把多个channel draw当model seed。

当前本地聚焦协议负测和编译通过；唯一P0/P1审查与修复结果在verification.json；训练和新正式结果尚未生成，不能宣称改善。实际代码commit发布前填写。

## 已核实的启动证据

实际训练代码 c577b3daded412337fb4181be05078ce051a64f1 已提交、推送，独立API对比当前分支OID identical。直接Git连接超时后，使用系统原已配置的本机代理恢复推送；未修改系统或Git持久网络配置。唯一发布归档一次SHA比较及远端compile PASS。

同原AS50工厂848个实际native args一致，仅run_id/candidate_id/output_dir不同。真实checkpoint严格加载2合成包无query smoke PASS；随后随机F的真实native64As/512NN合成验证PASS，默认头与Sinc均有有限非零梯度，0optimizer更新，正式source输出未提前创建。

独立读回dispatcher723131、worker723350/GPU0，CWD/argv/CUDA和日志增长正确；实际完整源L/U/V6300/56700/27000，初始E1完成。原生第一批配置读回residual/post_sync/EQfalse/G1/ref30/fs25MHz/fc2.45GHz/reflect，quality adaptation、相位跟踪和IQ补偿开启；实际64条信道记录。原网络900722参数/206状态、6×160头/scale30保持。

用户已确认就是这类保留clean的方案。P36草稿暂置、历史timing/FIR负结果保留，不修改另一对话健康RXpopulation实验。独立分析代码已实现：4个聚焦反例通过，最终读取所有30轮/6660步、426240As物理记录、三行7视图和源状态质量分层；尚未生成新结果或图表，不提前提供图表路径。
