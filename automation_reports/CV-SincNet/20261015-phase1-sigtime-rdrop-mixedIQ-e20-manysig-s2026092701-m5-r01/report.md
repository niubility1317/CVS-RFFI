# SIG_TIME＋RDrop：E20 泛化与星地IQ混合增强

状态：RUNNING，五个真实scratch模型已启动并各自完成至少首轮。固定seed2026092701，fullL6300/U56700/V27000、源RX1/3/4/6/8和day1/2/3。原SIG_TIME RAW_HZ/joint160/NoExplicitCFO、默认cosine6×160/scale30和正常联合训练保持。

先前冻结源算子安全PASS（135002前向/0更新）只支持进入训练，不是泛化收益。当前唯一假设：轻度接收响应变化与完整practical分别监督，可保留clean泛化和星地适应的共同收益。若G不提高clean或B11吃掉clean增量、未超过B01urban，则保留负结果，不堆对齐损失救场。

R_LEGACY保留原普通L增强与原RDrop两次前向。B00/B10/B01/B11关闭旧普通L增强，每个真实物理样本三槽raw、G/raw、S/raw，每槽同一已生成IQ做两次RDrop。权重0.25/0.25/0.50，总L尺度1；在native中替换原CE，不是再完整叠加。所有合法有标签satellite参与监督，不需要urban学生先高置信。不新增跨view一致性、诊断头、部署分支或U增强。

全部scratch总E20=13L+7U、4440更新，原AdamW/cosine/旧batch_neighbor门控与ownEMA。初始198个state张量匹配，四因子组6次L前向成本相同，原参照2次。五行预计L前向14776320，U/EMA/源V/正式预测实测计数，不拿同epoch冒称同计算量。没有加载E50、E200或追加训练。

A_g固定|a|≤0.06的received-IQ三tap线性响应并恢复输入RMS，虚拟RX/day/eq/epoch session与TX无关。A_s从原IQ调用原practical residual/post_sync、25MHz、referenceSNR30、EQfalse(G1)、reflect，保留几何/状态/补偿依赖，不先叠普通AWGN/多径。场景分布固定为80%原三场景均匀＋20%全部六场景均匀。正式IQ/信道/SNR不改。

共用同一V角色中的固定分层720条作源机制观察，V不拆角色、不更新任何学习状态；全V27000继续原健康评价。source规则在experiment.json固定，仅从源记录判断后续seed资格。原E20默认头只用于首步无query smoke及固定源clean正确集合，不初始化、教学或参与新优化；原20轮/4440/923.560061秒作为参考成本另列。

每row训练冻结及源结果保存后立即自动原clean＋六practical预测；全部七视图logits/decisions固定后fresh CUDA隐藏CPU scorer连接truth。sourceFAIL不删除固定对照或选择性漏测；目标成绩不回流参数、轮次、采样、source规则、候选或重跑。首轮一个seed不声明跨seed复现。

启动和停止按同目录experiment.json：唯一owner、不可覆盖root、all-owner/preCUDA每GPU≤3且空闲≥12000MiB。技术失败保留产物、停止pending并让已启动健康行自然完成，不操作他人任务。真实CP smoke是launcher第一步，PASS立即继续；没有额外设计批准。

已执行真实网络五条件合成检查：同初始state、真实native L scope/标签/损失hook、同IQ RDrop对、默认W/fusion/SIG有限非零梯度、EMA深拷贝归属/未更新状态、2/6前向计数、无持久forward hooks；3328个合成前向、10个输入/顺序负例，0真实IQ/CP/优化更新。另13项固定预算/数据/配置及GPU评分拒绝检查PASS。限定独立P0/P1审查与唯一问题的定向复核均PASS。另已验证observer/EMA前向计数与作用域恢复。

本地首次检查暴露共享source导入code/SSDG旧实现；根据实际导入路径定位后，在加载共享source前显式缓存已审计的adv3b02_xuc native。修复后的完整五条件检查PASS，不是删除参数绕过解析。真实发布还会记录actual __file__与resolved args。

冻结配置、逐row命令、环境/CWD、数据/seed、输出路径、科学阈值与预期产物以experiment.json为准。实际Git code/config commit在release与resolved_config中保存；报告和注册不是性能结果。

审查发现的源panel发布竞态已修复：同目录临时JSON完整关闭后原子发布最终文件。两项针对性检查PASS。没有额外审查或报告批准。真实训练已启动；首步CP smoke为PASS。

实际发布代码 `1fe3f00bdddfdb0af494f74c6570db65490ddf72`，远端单归档传输比对与编译通过。运行后独立检查五个PID/CWD/argv/GPU和日志增长均VERIFIED。R_LEGACY为E4、B00/B10为E3、B01/B11为E1时完成首轮执行核验；星地组每轮222更新、28416物理L星地包、170496个L前向符合登记。此为执行证据，训练和正式评分尚未完成。
