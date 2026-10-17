# 完整 clean 保留配方：独立 U 星地教学 E30

状态：LOCAL_VERIFIED，尚未启动。唯一假设：在完整原参考配方上，复用自身clean EMA的可信分布监督同包完整practical U，能优于同成本的额外raw U监督，并保住clean。P38固定源面板已显示teacher置信precision99.88%至100%、coverage89%至92.56%、low加权教学机会38%至40%；这些是源域资格，不是U真值精度或训练增益。旧SIG_TIME E50的SSL增量很小，不能直接复用其结论或叠加收益。

固定3组、1模型seed2026092701，全部独立从零开始，各自20+10累计E30/6660更新（总90modelE/19980更新）。B0为原参考L配方加一次rawU零损失学生前向；URAW以相同rawU学生前向计算额外软KL；UAS将该额外前向换成完整As(rawU)，其余规则相同。每个row的第一阶段13L+7U/cos20，第二阶段严格加载自身固定FINAL20及自身EMA、重置AdamW/cos10/0L+10U；原L在全部30epoch参与。未用旧目标暴露checkpoint初始化；真实旧E20只做兼容性smoke，随后丢弃。

L严格沿原完整参考：普通128两次、混合128两次；混合64ordinary+64完整As(raw)，有效CE监督ordinary.75/satellite.25；RDrop两次IQ相同、原SupCon仅普通pair。原ordinary冻结E13实际强度，保持原L信道逐包3场景名义采样与stream20261016421。F/SIG_TIME_X3/统一Sinc CENTER_25K与CENTER_UNITS/NoExplicitCFO/原joint160默认cos6×160 scale30，骨干和头联合训练；900722参数206state，不增加部署模块、头、诊断分类器、特征全对齐或新的Ag。

源角色精确复用split392005、7/63/30、L6300/U56700隐藏/V27000，RX1/3/4/6/8×day1/2/3×6TX；无训练RX留出、无L_dev、无V梯度。额外U复用原ownEMA out_w，0额外U教师前向；旧confidence/temporal/strong门控、旧U CE/entropy、EMA更新不变。原native在两个新钩子前均断言U标签=-1，新代码只读eq/rx/day/sig/base_index元数据，不读TX。

每个row全部17个Uepoch都执行额外963900学生包前向，含q=0包；B0附加loss为0。URAW/UAS用T1原scale30 softmax，q=clip((maxp_teacher-.95)/.05,0,1)，teacher stop-gradient，loss=batchmean(q KL)，系数0.16×min(1,(累计epoch−13)/5)。不按入选数重新归一化；不要求urban学生高置信；不从困难urban反向监督clean。该系数、ramp、tau不扫描。新增前向用private dropout stream20261018442+step并恢复CPU/CUDA RNG，三个row同流，避免改动后续原训练随机序列。

UAS调用原真实residual/post_sync/noEQ(G1)/reference30/reflect，fs25MHz/fc2.462GHz、质量适配/phase/IQ补偿和传播遮挡噪声依赖不拆散；新U stream20261018441与L/验证/测试分离，按epoch/step/physicalID刷新；硬件stream20261016423和原L一致，按RX/day/eq会话稳定，禁止TX定义session。dataset RMS后As自身unitRMS，无post-As RMS，不反演未知TX。保留三训练场景、全部LOS/PARTIAL/BLOCKED及锁定状态支持，不拒绝困难U。

首次U前，各row仅用固定合法V1800及既有2draw×3scene full源观测作一次自身teacher资格核查，14400NN（student12600/teacher1800），0更新/状态不变；全部条件固定为precision≥99%、coverage≥75%、每RXprecision≥98%、两draw mid/low q加权teacher正确/student错误机会≥10%。失败则本候选后续额外loss永久关闭，仍生成登记视图、完成预算及全部正式评价，不重新校准或救跑。后段只继承本row自身E20已有资格状态和EMA，不加载旧诊断教师。

最终源科学条件：UAS相对URAW在每个draw的mid/low均≥+1个百分点；fullVclean UAS≥B0−0.25个百分点且≥URAW−0.25个百分点，所有own teacher资格通过。未通过则不扩seed、不调整系数/扰动/阈值/epoch救跑。该门槛只评估固定C4，不代表整体Ag或联合目标完成。目标探索条件仍为clean净正收益、low>60%、suburban各场景距自身clean≤2个百分点，均不保证结果。

每row固定最终checkpoint后立即独立预测clean及既有全部6practical；每环境168000、全部7预测写完后才exec独立truth-last scorer。正式received IQ、测试SNR、信道、边界及样本集合不动。源矩阵判定仅dispatcher写，不读取formal；主agent在源判定固定前不读取正式成绩，不据目标结果选择或重跑。某row完成不等待整批才进测试。

执行CWD：/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_cleanurban_U_E30_20261018_r01；环境：N607普通uid1000、CVS-RFFI Python/Torch2.6、FP32、CPU2threads。命令：

```text
/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_cleanurban_U_E30_20261018_r01/experiments/cvs_Uurban39/runner.py
```

唯一路径/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01，日志/home/szu2070436088/2510044040/CV-SincNet/runs/20261018-phase1-FcrossRX-cleanurban-U-e30-s2026092701-m3-r01/logs及同名-launch，owner codex/root/Uurban39；每GPU最多3总进程且计入其他owner，不干预健康旧任务。停止仅限错误角色/来源、越权query/truth、输出碰撞、非有限loss/grad、无法执行或无法生成合法预测等系统技术失败；保留partial，不自动重启。不以低分或科学失败停止。

代码与配置提交身份待实际commit写入；Git先提交push并独立比远端，单release归档一次SHA比较及远端compile。launcher首步真实E20 checkpoint零query smoke，PASS立即继续real-F三组额外U合成梯度烟测；synthetic输出独立不占stage20/source路径。worker分配GPU后先作本次CUDA RNG问题定点回归，0NN/0更新。一次独立P0/P1审查发现3个未定义变量，均已定点修复并PASS；本地6项检查PASS/1项CUDA不可用SKIP、compilePASS。

成本计划：每row新30E，父成本是本row已计入的20E，不另借父预算；LNN3409920、原U学生与教师各963900、额外U学生963900、首次U源检查14400、冻结源读出12600、正式预测1176002及原native验证/初始smoke，实际NN和墙钟按完整日志逐阶段累计。总预算耗尽不等于收敛。预期产物见experiment.json：真实配置/导入文件、source角色、ownparent/EMA/optimizer、完整epoch/step/L-U-sequence、U信道元数据、覆盖、资格、默认头源预测、正式逐row/seed/RX/day/TX七环境scores、独立报告/分层图/训练曲线/成本/负结果。
