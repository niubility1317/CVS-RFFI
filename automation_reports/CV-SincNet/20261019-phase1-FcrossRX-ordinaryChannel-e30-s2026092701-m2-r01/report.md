# P78：保留完整 practical，仅移除 L 普通视图的 AWGN／多径

状态：REGISTERED，实际训练尚未启动；代码提交在本地实现、聚焦检查及唯一独立 P0/P1 检查完成后固定。起点 ca91644d9e13efdfb9fb7f343f9041743772af96。

唯一假设：由完整 practical 承担真实传播损伤后，L ordinary 中的通用 AWGN／multipath 是否仍有价值。只在末 E10 旁路这两种原语，保留相位、时移、增益、PA、原 RDrop／SupCon、U／自身 EMA、默认 cosine 头与网络结构。普通原语完整执行后返回其输入，保留同调用随机抽样和模拟成本，不宣称整条训练轨迹 RNG 逐位一致。As 仍从 raw 独立生成；原实现没有把 ordinary 损伤再串进 As，本次不修复一个不存在的重复叠加问题。

此前 ordinaryChannel 只冻结两个旧源折 E20 默认头、没有 As 和任何训练；no_both 相对 full 的源 fitL 提高约 1.5 个百分点，但未满足当时严格 raw 身份保护条件，旧 FAIL 保留。LRaw 则把全部 L ordinary 换为 raw，连相位、时移、PA 等都删除。当前用户已明确放松过程身份保护，要求实际匹配训练验证。本次改变的是“只移除两种通用传播损伤，交由保留的完整 As 负责”的训练交互假设，不复活旧诊断结论，不在失败后选择单项或调比例。

两条新行：B00=OC_NOAS30，B01=OC_AS30。两行分别严格加载 P43 B00/B01 自身固定 E20 student 与自身 EMA，追加 E10，总 E30/6660 更新；尾段重置 AdamW lr=.0002、wd=.0001、cos10、CENTER_UNITS。LRaw d14450bf 仅提供已冻结代码运行环境，不继承其训练权重，也不加载 P77/SAT18 父。复用原 P43 NOAS30/AS30 读出，0 重预测／重评分，构成末十轮交互四格。两条父的 As 历史不同，不冒充同一 As 父的无 As 尾段。

固定 F＋跨 RX 同 TX、SIG_TIME_X3、真实 joint160 和默认 cosine30 头，骨干与头正常联合训练。完整 L6300/U56700（TX 隐藏）/V27000、源 RX1/3/4/6/8、day1/2/3、六 TX，split392005，无训练域留出。IQ2×256、25 MHz、原 RMS center=false；按原 normalize_input 后完整 practical residual/post_sync，EQ 关闭则 G=1，reference_snr_db=30、reflect。接收质量适配、相位跟踪、IQ 补偿、session 硬件和 AGC 的原依赖保持。实际导入 __file__、原生 args 和 resolved config 保存，不用默认值代替运行值。正式 IQ、六 practical、2.45 GHz 和噪声条件保持。

每步 6L/768，同一已生成 IQ 的两次 Dropout 前向分别算原目标；L 总 CE 系数 1.25。无新损失、分支、头、额外 NN 或部署模块。全部合法有标签 As 参与 CE，不受学生置信度门控。每行尾段原生 L 序列、U 567000 包、教师与伪标签规则保留；实际原语命中、影子生成与包数、PA 等保留变化、U wrapper 透传检查及完整训练成本均记录。不把接受率当隐藏 U 精度。

固定源晋级条件：唯一候选 OC_AS30，全 V clean 相对 OC_NOAS30 至少 −0.1 绝对百分点；固定 1800 物理包×2 draw 的 mid/low 均值相对原 AS30 至少 +0.5 个百分点。失败仍完成两行 clean＋全部六种 practical，关闭本固定组合，不扩 E50／seed，不拆项救跑。只有源条件通过才讨论当前已授权的匹配总 E50；本次发布不预先追加。

比较同时报告原 As−NoAs、OC_As−OC_NoAs、OC_As−原 As、OC_NoAs−原 NoAs，以及四格交互。正式每视图 168000 包；每行全部七视图 prediction 固定后才由独立 scorer 连接 truth。owner 先观察保存的 source decision 再读取新正式成绩，正式分数不回流参数、候选或重跑。

新增预算：两条 E10 尾段，20 模型轮／4440 更新；各 native3162962、source12600、formal1176002 前向，共享真实 CP 无 query smoke 预计1540，实际值单列。本地真实 NN=0。两旧 E20 父与两旧 E10 参照加两新 E10，唯一训练合计80模型轮／17760更新；四条E30曲线120行不是120轮新增成本。影子原语仍计算，不宣称加速或同NN等于同墙钟；预算耗尽不等于收敛。

环境与命令：N607 普通账号既有 CVS-RFFI Python3.10/Torch2.1/NumPy2.2，full FP32、CPU2。CWD=PROJECT/releases/cvs_FcrossRX_ordinaryChannel_E30_20261019_r01；执行 python -u experiments/cvs_ordinary_channel78/runner.py。唯一 owner=codex/root/ordinary-channel78，不可覆盖输出 PROJECT/runs/20261019-phase1-FcrossRX-ordinaryChannel-e30-s2026092701-m2-r01；每 GPU 最多2个全 owner 进程，健康旧任务不改动。

发布执行一次 preflight、一个归档的一次本地／远端 SHA 比较、一次远端编译；launcher 首步真实 CP/default-head/ownEMA smoke，PASS 直接训练。仅在所属 run 的数据角色／父／头／预算／非有限／输出碰撞、L wrapper 泄入 U 或不能产生合法 prediction 等技术失败时按预登记规则停所属句柄；保留产物、不自动重试、性能差不停机。

预期产物：actual_modules／resolved_config／严格父与自身 EMA 初始化；完整2220步的原目标、普通原语旁路与影子、U透传、物理覆盖和epoch日志；固定FINAL30与冻结记录；源分层与source决定；all7预测、独立评分；全部四条曲线、默认头margin/30、固定NoAs正确集合保持率、RX/day/TX和state/lock/新增噪声比分层、配对纠错、交互、实际成本及PNG/PDF。P43真实父clean正确缓存未确认时不混用SAT18集合，不新增诊断头或为了补表重推理。

REJECTED_EXTRA_GATE：通用技能的单独设计审批、规格文件和重复审查不在项目八项白名单内；用户已授权本目标内必要实现与实验，沿当前最小流程执行。

实际代码提交 5d6514b00b96614a1bd8e2b276e286bf3bed37e6，独立远端OID一致；聚焦检查和唯一P0/P1均PASS，真实训练待发布首步smoke。

实际发布：代码 5d6514b00b96614a1bd8e2b276e286bf3bed37e6，92355byte单归档比较与25文件remote compile均VERIFIED。真实CP/默认头/自身EMA＋两图smoke1540NN PASS，sourceIQ0/queryfalse/updates0；强制原语门仅用于合成smoke并恢复。两row PID/CWD/argv/CUDA/actualGPU0/1独立核实，source_rolesEXACT_MATCH/scratchfalse/targetcontactfalse。训练进行中，无源晋级或正式成绩结论。

首轮已观察：两row均完成尾段1/10、各222更新，日志增长且无技术failure。actual_modules打印证实leo_practical/DataAugmentation/SSDG均来自固定r02的experiments/adv3b02_xuc/code，非仓库顶层code。
