# 保留 clean 份额的连续 E30：两行固定预登记

唯一待检验假设：在原 clean/As 监督份额下，从零连续 E30 的优化轨迹，相比现有分段 E20＋10，能否稳定源域 clean 并改善原 mid/low urban。此项是训练组织消融，不是新 A_g；不会把优化收益冒充 IQ 泛化增强。

起点 HEAD 为 02460c72602877e414450a16285b642f45877e06。原模型、默认头与数据契约保持：F/SIG_TIME_X3、统一 Sinc CENTER_25K/CENTER_UNITS、NoExplicitCFO、原内部融合/参考、joint160/defaultcos30，骨干和默认头联合训练。原5源 RX1/3/4/6/8、day1/2/3、6TX，L6300/U56700 TX隐藏/V27000、split392005、eq1、IQ256/25MHz、RMS center=false。无源 RX 留出或新 L_dev，不合并 V，不读 U TX；训练从零，不加载历史网络。

已读取完整六行原训练曲线。所有原 B00/B01/seed01..03 在 E20 lr=1e−6，E21 重启到2e−4；源 V E21 相对E20下降0.0037至0.3444个百分点，E22后恢复。只说明轨迹确有可检验差异，不能证明重启有害，更不能由正式目标成绩选择新 schedule。原3seed的目标分数不作为此候选晋级依据。

固定两行、首个 modelseed2026092701：新 B00 无 As、新 B01 原 full practical As。两行都从零单段 E30、6660更新、222步/epoch、6×128 L前向/步，共60模型epoch/13320更新/10229760 L前向；单个原 AdamW（lr2e−4、WD1e−4）、单个 cosine30到1e−6，没有E21重启。保持13轮仅L＋17轮原U/EMA/旧门控，原普通增强在E13冻结，原SupCon warm5/ramp10，原 U 视图和损失保持，不添加urban U或额外教师。

保留完整旧cleanretain：共同ordinary128的.5 RD，mixed128的.5 RD，额外raw128的.25 RD，总L系数1.25。B00 mixed的64ordinary＋64raw；B01 mixed的64ordinary＋64原As(raw)。每个RD内两次forward使用同一IQ；As前不叠ordinary。两行都有原ordinary并非完全无增强。训练原As覆盖low_suburban/mid_urban/low_urban，residual/post_sync/EQoff G1/reference30/接收质量与相位IQ补偿/AGC/reflect保持；不减弱正式测试。TX×RX×day覆盖、session不由TX定义、跨epoch刷新及训练/验证/测试流继承原实现。

此比较同时改变学习率时间轨迹、AdamW/EMA/temporalcache生命周期和采样随机流的阶段重置，不能单独归因“只去掉学习率重启”。旧P43相同seed B00/B01 E20＋10只作为固定历史比较，不初始化新行。代码内部segment=20只保留旧scratch hook/observer分支选择，实际nativeepochs=30、cumulativeepochs=30、offset=0、parent=None；不得将其解释成E20预算或额外20+30。

源判据在新结果前固定：新B01完整源V clean不低于新B00；新B01较旧分段B01源V clean不低于−0.25个百分点，各RX不低于−0.5；既有源1800物理样本×2命名draw上的mid/low四项准确率各至少高于旧B01 1个百分点。另要求完整预算、有限loss/grad、默认头联合更新、数据角色与原训练健康正常。任意科学条件失败即关闭此固定连续轨迹，不换LR/日历/U阶段/权重/seed/预算/门槛救援；全部条件通过也不自动扩种子。

两行FINAL30按固定epoch冻结，不按源/目标最优epoch选模。每row完成后立即使用既有固定clean＋六practical receivedIQ（168000/视图），全部七视图prediction固定后才独立CPU truth-last评分。源判据失败仍执行两行正式评价；目标结果不回流候选排序、参数、重跑或其他健康训练。正式指标为urban三场景/low/mid、最差RX、macroF1、clean及suburban绝对值/距clean差；源分层、默认头间隔/固定clean正确集保持率复用已保存logits，曲线/梯度/旧门控使用完整原记录。不能声称跨seed或已验证新Ag。

run：`20261018-phase1-FcrossRX-continuous-e30-s2026092701-m2-r01`；owner：codex/root/continuous52；CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_continuous_E30_20261018_r01`；不可覆盖root：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-continuous-e30-s2026092701-m2-r01`；每row日志：root/logs/row.log；dispatcher日志：`/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-continuous-e30-s2026092701-m2-r01-launch`；本地产物：`E:/type10-7/local_artifacts/cvs_FcrossRX_continuous_E30_20261018`。GPU由既有容量调度选择，各GPU总进程含其他owner最多3且空闲≥12000MB，优先空闲卡；不停止/重启/热改健康旧任务。

命令：`CVS-RFFI/bin/python -u experiments/cvs_continuous52/runner.py`。发布沿现有原生Windows Python publisher，单包一次SHA比较与远端编译/bootstrap。launcher首步7项协议负例，随后实际同契约checkpoint no-query（只smoke，非初始化）2NN，以及两行真实F合成梯度/6L/原As/头与Sinc/预算测试1536NN，PASS立即训练。代码/config commit发布前补齐，无额外审批或seal。

本地7项配置负例/参数映射/命名空间归档/自动测试路径验证PASS（0.093s）；发布远端脚本均仅mock执行。未读取真实IQ、checkpoint、U/target，未进行训练或实际NN。14个源文件AST待记录；单次独立P0/P1审查待返回。实际运行成本（父训练0、当前增量30/行、更新、学生/教师/验证/预测NN、墙钟）按产物记录，不由epoch直接猜测。

技术stop仅本run输入/角色/来源不匹配、非有限loss/grad、输出碰撞、不可启动或非法prediction：保存产物，不自动重试；dispatcher停止未启动行，健康运行行自然完成。科学FAIL不停止已登记正式测试。交付实际source/result、完整训练/sequence、预测/独立分数、source固定判定、逐row汇总、独立图表/曲线、负结果、未完成项及Git。当前无新准确率，整体clean正增、lowurban>60%、suburban接近clean、新Ag及跨seed目标均未完成。

本地14源文件AST PASS；单次独立P0/P1审查PASS，7项本地检查由审查者独立执行通过。实际checkpoint和1536NN合成smoke仍在launcher首步，未提前冒称通过。
