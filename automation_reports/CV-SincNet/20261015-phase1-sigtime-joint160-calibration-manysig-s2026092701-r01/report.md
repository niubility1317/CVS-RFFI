# joint160 方向蒸馏：零训练源梯度校准

状态：LOCAL_VERIFIED，尚未发布。本 run 只读取已经 source-frozen 的 C3 E50/11100 checkpoint 与同 checkpoint 的自有 EMA；不初始化或继续训练任何模型。校准新增 epoch/optimizer update/head-fit 均为 0。

用户 2026-10-15 新设计要求先比较现有间隔、普通输出蒸馏和真实 joint160 方向蒸馏。该阶段只落实直接身份目标，尚不加入中间视图、112 维局部匹配、PA/raw29 等值目标、预测 MLP 或新权重网络。当前四行 E50 release、网络、默认头、既有测试 IQ 和其他健康任务保持。

模型来源：20261015-phase1-sigtime-urban-ssl-e50-manysig-s2026092701-m4-r01/C3-s2026092701/source/final_ssdg.pth，model+ownEMA。own_model 必须核实原实际源角色、scratch/无父/无目标训练、固定final50、11100更新、source_frozen；来源不符即停止。该来源仅用于只读诊断，未来 B/C 仍从零训练，不继承它的参数、optimizer 或 EMA。

源输入：既有 720 条 TX×源RX×day 平衡 L_s 面板（源总 L6300/U56700/V27000，split392005）。不对 U 或 query 做前向，不读取 U 真值/目标分数；V 不用于本校准。固定 mid/low urban 源验证命名流，物理信道完全沿原 residual/post_sync/noEQ、refSNR30、reflect 边界和稳定 RX/day session。两个源场景不意味着正式测试删减。

对每个90样本batch，复用实际 satellite 前向得到 id_feat_joint，并捕获 PA/ref/SIG 的 gain 前特征。使用 stop-gradient gain 重组的辅助特征必须逐值等于实际 joint160，分支值保持梯度。KL 使用原 cosine6×160/scale30，W.detach；方向损失使用原 eps1e-4 的单位方向。两种目标均使用 clean EMA 正确这一条 L 资格，无学生 urban 置信度/正确性门控。CE 仍使用真实 L 标签。

只读 autograd.grad 分别测 t1、t2、SIG local_projection、PA_projection、response_projection 的原始辅助梯度与 0.5×satellite CE 梯度。静态系数取 0.1/全部源batch与五组的中位比值，有界[1e-4,10]，披露有效比值和是否截断；不在线强迫分支梯度相同。检查辅助 W/gain 梯度为0、所有模型/EMA参数及buffer逐值不变、没有累积grad。代码不创建 optimizer。拷贝 A 的 source-only 已实现 50 轮 eta 日程供后续匹配，不读取A目标分数。

预期 2250 个前向样本（720 clean教师、90第一batch actual identity路径核对、1440 satellite学生），实际成本以 results.json 为准。Dropout 只在这次诊断中 eval，未来主训练保持原 train Dropout；不把两类误差混为信道效果。真实 checkpoint smoke 是本程序首个batch，PASS 后立即继续，无独立授权文件。

CPU-only，普通N607账户，CUDA_VISIBLE_DEVICES为空；torch/OMP/MKL/OpenBLAS各2线程。CWD PROJECT/releases/cvs_joint160_source_calibration_20261015_r01；命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u -m experiments.cvs_urban_identity50.calibrate`。唯一 owner codex/root/joint160-source-calibration-20261015，唯一 run/output root PROJECT/runs/20261015-phase1-sigtime-joint160-calibration-manysig-s2026092701-r01。实际代码/config以release_commit.txt的Git commit标识。

本地发布命令：`E:/type10-7/local_envs/ssr-gpu/python.exe -X utf8 -m experiments.cvs_urban_identity50.calibration_publish --output E:/type10-7/local_artifacts/cvs_joint160_identity_20261015/calibration_delivery_r01`。发布一次archive传输SHA比对、远端编译/普通账户预检；随后独立核实PID/CWD/argv/无GPU/日志增长或已完成results。来源/角色/非有限/状态变化/输出冲突等技术失败停止本owned只读程序、保留输出，不自动重试，不停止或热改其他健康任务。

本地7项新检查PASS：实际joint160 train/eval相同、W/gain辅助零梯度与五组正梯度、CE头/gain正梯度、方向缩放不变/零输入有限、错误教师只取消aux、teacher/reliability stopgrad与隐藏标签拒绝、静态系数/截断/零梯度拒绝。旧E50 10项兼容检查PASS。一次独立P0/P1已完成，未发现阻断问题；只核查此零训练阶段；未来B/C训练接线不在本次发布范围。

预期 results.json 包含原始80×2种梯度记录、静态系数、共同eta日程、来源和2250前向/墙钟/状态不变证据；不生成目标prediction或score。此诊断是明确source-only，不能称为Phase1正式性能完成。设计追踪见 analysis/cvs_joint160_distill_traceability_20261015.md。
