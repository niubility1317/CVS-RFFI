# 完整源域 F＋跨 RX：只改变 practical 的输入

状态：PREREGISTERED，尚未发布/启动。唯一矩阵见experiment.json，仅RAW_PRAC-s2026092701一行。固定参照为正在按原计划运行的完整源域P22 B01普通拼接；未读取任何目标分数。本行依据是实际代码与配置确认：原native先施加旧ordinary（允许PA/CFO/AWGN/multipath/shift），再施加practical。这可能重复损伤，不等于这些旧普通增强已验证身份保持。

唯一差异：原B01为clean=ordinary(x)，satellite=practical(ordinary(x))；本行仅令satellite=practical(x)，x为同批原始合法源归一化IQ。clean、真实TX标签及其权重、原native concat_full、全部物理信道参数、随机流、RX/day session、接收质量适配与补偿、RDrop、clean-only跨RX同TX SupCon均不变。接收状态正常重新计算，不冻结为clean、不弱化原practical、不假定算子可交换。未知TX硬件不反演。

全部原源RX1/3/4/6/8、day1/2/3、6TX，L6300/hiddenU56700/V27000、split392005/7-63-30精确原物理IDs，无额外留RX/Ldev，不合并V或解封U。训练仅为全新scratch E20=13L+7U、222steps/epoch、4440updates、2273280 L网络样本量，默认F三路SIG_TIME/CENTER_UNITS/joint160/cos30正常联合训练，0新部署模块/头/损失。U/ownEMA/旧门控保持；新raw缓存仅L使用。

每个L包clean/sat各0.5，原CE/RDrop都按两半分别平均，复用同一拼接IQ的两次dropout前向；SupCon只读clean128半批。训练环境low_suburban/mid_urban/low_urban，原native场景先验和跨epoch/view刷新；25MHz/2.45GHz,residual/post_sync,EQoff(G=1),reference30,quality/phase/IQ补偿开启,reflect，无星地后外部RMS。训练三场景全部状态支持保留；测试received IQ/SNR/边界与六场景一律不改。

旧的留RX配对As曾有clean失败，原产物和判据保留。本次是用户明确恢复完整原训练域后的直接比较，不用旧留域结果替代完整基线、不加载其权重，也不宣称新方法/Ag成立。P22的默认头/损失/F优化器代码按不可变dc4e47fe0复用，新包只包装同批raw输入。在独立新进程设置模块配置引用，不修改P22代码文件或健康进程。

数据流与7项negative检查PASS：合成128包raw逐点相等、clean不变、标签两半一致、缓存消费一次、stale/U/错shape/预算/role拒绝；0真实IQ/CP/NN/物理channel/更新。编译PASS，一次独立增量P0/P1审查PASS，无未解决P0/P1。原完整F数学不变，复用P22已执行全图检查。一次N607预检PASS。

第一执行步使用已核实完整源契约CROSS_RX FINAL50（原scratch/无祖先/默认头、FINAL50且冻结前无target）的2合成样本严格加载兼容smoke；不初始化新训练或EMA，历史50epoch/11100成本单列。新候选只增加20modelE/4440updates；两项P22参照40modelE不隐藏、不当作本行父训练。

逐row FINAL20固定后自动原clean＋六practical的168000×7预测，全部prediction完成后另exec独立truth-last；只有候选和参照七视图/评分均完成才读取比较数据。没有源/目标性能选模、晋级或选择性重跑。探索目标预先固定相对P22 B01：urban均值+3pp、low+5pp、clean下降≤0.5pp；low>60%和各suburban距自身clean≤2pp为探索目标，不保证达到。不满足时保留负结果，不扫本输入路由的比例、强度、场景、seed或预算救援。一个seed不证明复现，没有新Ag或Ag×As交互结论。

环境/CWD为普通N607 CVS-RFFI Python，/home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_rawpractical_E20_20261017_r01；每GPU总进程≤3含其他owner，空闲≥12000MB；独占run root /home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-fullsource-rawpractical-e20-s2026092701-m1-r01，launch owner/root。精确入口：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u /home/szu2070436088/2510044040/CV-SincNet/releases/cvs_FcrossRX_fullsource_rawpractical_E20_20261017_r01/experiments/cvs_fullsource_rawpractical23/runner.py`；新worker/source/prediction/scorer均使用新包入口。dispatcher/scorer CUDA空。代码commit发布前登记，单一归档传输比较/remote compile；不触碰健康旧任务。

停止：固定预算，不按微小源分数早停；数据角色/标签/来源/非有限loss/grad/预测/技术错误保留产物、阻止未启动row，不自动重启。其他健康任务继续。预算结束不等于收敛。预期实际config/module_files/raw路由证明、物理覆盖、完整step/epoch曲线与成本、FINAL20、七视图prediction、独立score、逐row/seed CSV和同row报告见experiment.json。

实际实现commit：`31d7fc60f75819ff25a698a29df9fc7ec03691c8`，目标分数未读；固定此实现发布，不更改P22健康运行。


## 实际发布与路由生效

实际执行commit `31d7fc60f75819ff25a698a29df9fc7ec03691c8` 已推送，远端OID独立核对一致；一次归档传输比较VERIFIED，remote compile PASS。独立dispatcher/worker已启动，原P22进程保持运行，GPU由实际资源分配（本行GPU5），未停止或热改任何其他任务。真实fullsource CP兼容smoke PASS，2合成NN、0训练更新、无query。

PID/start_ticks/CWD/argv/GPU/日志增长与全部原L6300/U56700/V27000精确角色独立读回，无fold参数。raw输入首批证明：128×2×256，与同批ordinary输出MSE=0.003188482951372862；clean不变、同一L物理元数据、接收状态正常重算、无TX条件信道或未知TX反演。首批LOS46/PARTIAL11/BLOCKED71、locked54/degraded16/unlocked58，与原固定流参照首批一致，EQoff/reference30等实际配置保持。

截至早期完整可解压step记录，本行至少236次真实更新，活跃CE/RDrop/KL有限、nonfinite loss/grad skip=0；这只证明训练和输入路由实际生效，不证明urban增益/收敛/跨seed/Ag。当前没有读取目标分数，没有改变任何健康运行配方。
