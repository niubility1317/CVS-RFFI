# Tweak V7：稳定欧氏距离与source侧选模

状态：ANALYZED / VERIFIED；100epoch与20行独立评分完成，论文数值复现未达成。用户于2026-09-10授权修复并重新发布。

## 修复范围与实验假设

| ID | 问题与处理 | 状态 | 验证 |
|---|---|---|---|
| V7-1 | `torch.cdist`默认MM路径发生近邻相消；全部评分使用显式差分欧氏范数，校准均值/半径以float64累计 | verified | 同一近邻查询在1/32样本batch上预测一致；旧代码把32个第二类样本全部错分第一类，新实现通过 |
| V7-2 | 公共`shared_triplet_loss`统一平方L2，修正文档陈旧miner描述 | verified | Eq.(1)独立手算3.1，旧代码1.1失败，新代码通过 |
| V7-3 | 固定source训练池监控替代最低hard loss的LR/checkpoint排序 | verified | 高source识别率优先于margin-floor loss；test区NaN投毒不影响source监控；probe完整状态重置和best恢复测试通过 |
| V7-4 | 保存预测与opaque ID，再统计accuracy，保留用于独立复算的校准状态 | verified | 正式20行共781,200次判定经独立NumPy float64显式差分复算，预测零差异，opaque ID连接后的正确数全部一致 |
| V7-5 | 新release/run/log；保留V1—V6 | verified | 完成5个LR probe与100epoch；checkpoint、results、20prediction与20truth完整，原PID已退出 |

V6审计中的uniform缩尺度方向对所有只保留严格hard的目标都成立，并非全量miner独有，也不能单凭该方向证明更换miner能够解决塌缩。当前没有足够证据推出作者的准确miner；V7保留V6平方L2严格hard集合、网络和物理split，只修复已实证的欧氏距离相消，并修复选模对最小尺度的偏好。不新增normalization、CE或未经验证的loss。

source监控固定读取Config2每设备训练逻辑位置0…255作参考、256…505作监控样本（一次前向读到511，最后6帧不计入M=10聚合）；两部分互斥，均属于既有75%训练池，不访问25%测试帧。它是训练侧监控准确率，不是独立validation或泛化性能。采用相同Algorithm1/2和M=10；LR仍五个各1epoch探测、要求训练loss下降，优先source监控准确率，确定性loss/LR打破平局；选择后从相同初始权重重新训练100epoch，每epoch用相同source监控指标保留best。论文未公开该选模规则，明确记为UNPUBLISHED_DEFAULT，不声称严格数值等价。

Config2 source、设备1…10、四配置单校准4×4及四配置联合校准4行、seed=20260908、SGD momentum=.9、batch64=8×8、N=10%、M=10均保持。

## 发布路径与技术停止范围

N607根：`/home/szu2070436088/2510044040/CV-SincNet`。release=`releases/tweak_config2_portability_20260910_v7/source`，output=`runs/tweak_config2_portability_20260910_v7/official_config2_full`，log=`logs/tweak_config2_portability_20260910_v7/train.log`。数据沿用`datasets/tweak_official_lora_configurations_20260908/Diff_Configurations_Setup`。

只允许处理本run的确定性技术故障/非法输入/输出碰撞；低性能不停止。既有数据、checkpoint和日志保留。完整结果后与论文同row比较；source监控进步不代表论文数值复现。

## 本地验证

三个已确认回归先RED再GREEN；含source边界、预测拒绝覆盖和状态重置/最佳checkpoint恢复的相关测试共36项，编译通过。真实Config2 source诊断在同一seed、LR=.001下执行3,000batch，16组梯度有限，source训练池监控准确率在step1/100/1000/3000为16.0%/26.4%/56.0%/55.6%。最后loss=.101575，中心距=.033419、半径=.055210；这并不证明消除塌缩，V7保留source表现最佳的epoch，后续以正式完整结果判断。

冻结V6 checkpoint在修复后的source监控上为62.8%，说明非常小的embedding差异仍保有身份信号；不能单凭小绝对尺度断言已丢失全部可分信息。此前报告把评分可重复等同于排除数值评分错误、把strict-hard缩尺度方向归为全量miner独有，结论过强，本报告据回归实证修正。独立P0/P1审查无发布阻断，提醒监控不是独立validation、正式预测仍须完整后独立评分。

## 已发布与运行证据

- 代码提交：`03ad828e320a563d2088cb65864982ae71dcb433`，push及远端branch OID读回一致；源码tar为81,920字节，本地/远端SHA-256均为`0a2a0995bc0e7c96245ba5b69741aacc41c34918b29cfde87ab824c5fee875ad`。
- N607真实source CUDA前反传：`[64,2,128]→[64,12]`、平方strict-hard loss=.5866077542、16组梯度有限；source monitor实际执行成功。远端10个模块和诊断工具编译通过，40条正式数据记录及对应metadata存在。
- 正式PID=`852891`、PPID=1、物理GPU3（`CUDA_VISIBLE_DEVICES=3`，进程内`cuda:0`）。首次41秒读回PID存活、CWD及argv匹配新release，CPU115%、GPU448MiB。train.log当时0字节：按整epoch输出，尚未完成首个probe，不作为训练完成或最终健康保证。
- 实际命令：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -m paper_reproduction.gaskin_tweak_2023.official_lora_experiment --data-root /home/szu2070436088/2510044040/CV-SincNet/datasets/tweak_official_lora_configurations_20260908/Diff_Configurations_Setup --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/tweak_config2_portability_20260910_v7/official_config2_full --device cuda:0`。未使用smoke限制，5×1epoch LR probe后fresh100epoch，保留20行原矩阵。
- 已创建30分钟自动监控`tweak-n607-v7`；状态不变时安静，完成后用保存的20行prediction和独立truth进行复算再写最终报告。V1—V6均保留。

## 完整训练与独立评分（2026-09-10）

14:24 CST远端只读检查确认原PID852891已退出、日志以complete结束，42个结果文件齐全。完整39,514字节日志与所有42个文件已读回本报告根目录的`train.log`和`readback/`。未重启、未修改远端结果，也未启动V8。

独立工具[score_tweak_saved_predictions.py](../../../tools/score_tweak_saved_predictions.py)不导入生产预测器，以NumPy float64显式`query-centroid`差分重算Algorithm2，先核对全部保存预测，再按每行唯一opaque query ID连接truth。20行每行39,060个M=10判定，合计781,200次判定（不同校准行共享测试样本，不能视为781,200个独立观测），预测差异为0，正确数20/20完全一致。原results的accuracy由float32均值输出，与精确整数比的微小舍入差不超过5e-8。保存checkpoint的training元数据与results一致，权重全部有限。

完整日志106条JSON记录全部解析：5个LR probe、连续E1—E100、1条complete；日志训练记录与results逐字段一致，数值全部有限，没有非JSON异常/traceback。共1,922,550个active batch，所有probe/epoch均18,310个active batch。LR=.001；最高source训练池监控64.4%首次出现在E79，checkpoint正确选取E79，而不是loss最低epoch。全程source监控范围41.6%—64.4%。

| 时点 | mean loss | source训练池监控 | source平均中心距 | source平均半径 |
|---|---:|---:|---:|---:|
| E1 | 0.1020022930 | 57.6% | 0.01100395 | 0.01645946 |
| E79（选中） | 0.1000059996 | 64.4% | 0.00120948 | 0.00163267 |
| E100 | 0.1000049374 | 61.2% | 0.00127315 | 0.00152136 |

loss与几何尺度继续缩小，不能声称训练塌缩倾向已消除；但Config2测试62.906%和逐类25.755%—91.193%也不支持“全部身份信号丢失”。source监控是训练池诊断，不能当独立validation。未记录逐epoch墙钟/GPU峰值，不能补造训练时间或资源峰值结论。

## 同row最终准确率

下表为百分数；括号为正确数，分母统一39,060。模型始终仅在Config2训练，行表示校准配置，列表示测试配置。

| 校准配置 | Config1测试 | Config2测试 | Config3测试 | Config4测试 |
|---|---:|---:|---:|---:|
| Config1 | 45.773%（17879） | 4.373%（1708） | 13.013%（5083） | 0.246%（96） |
| Config2 | 7.696%（3006） | 62.906%（24571） | 15.177%（5928） | 12.002%（4688） |
| Config3 | 16.039%（6265） | 2.138%（835） | 31.011%（12113） | 12.058%（4710） |
| Config4 | 5.420%（2117） | 7.606%（2971） | 12.424%（4853） | 28.090%（10972） |
| 四配置联合 | 29.329%（11456） | 46.272%（18074） | 18.367%（7174） | 17.906%（6994） |

论文图13b同配置柱约75%/90%/74%/68%，图14联合校准柱约56%/82%/56%/53%（沿用此前原PDF读图近似值，不是作者原始数值表）。V7两组均明显低于论文，没有达成论文数值复现。论文未公开的TX选择、物理split、miner、选模等差异仍保留UNPUBLISHED_DEFAULT标签，不能把当前实现等同作者代码。

跨配置单校准大幅失配，例如Config1→Config4仅96/39,060；联合校准也低于各自同配置校准。Config2同校准的10类预测均实际出现（每类预测计数2738—4649），不是单一类常量预测。修复评分数值错误后，剩余域迁移与类可分性问题仍是真实结果，但不能仅凭此确认唯一根因。与V6历史分数的变化混合了评分、LR和checkpoint选择变化，不作单变量训练收益解释。

输入来源仍为本run从随机初始化训练的Config2 checkpoint，不继承V1—V6权重；LR探测与epoch选择只访问既定source训练池。此任务是官方LoRa原场景的外部论文复现，不是CVS的LEO/Stage2晋级结果。所有历史产物保留；本轮监控`tweak-n607-v7`已删除（调度器返回deleted），不根据测试分数继续调参或自动重跑。
