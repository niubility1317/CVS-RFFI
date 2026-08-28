# SF-TAPFT-PACE E0–E3实验报告

## 预登记

- run ID：`stage2_sf_tapft_pace_e0e3_rx20_1_s392002_20260828_r1`。
- 科学问题：在D0 Compact之后追加120步受保护time-Norm容量升级，并用support cross-fitted head logits拟合6参数零和类别bias，能否在一分钟预算内提高旧6类`DA1_REG0`性能且抑制局部负迁移。
- 矩阵：E0=`D0 Compact`；E1=`PACE-W`（新增`t2.norm.weight`）；E2=`PACE-All`（新增`t2/t1/time_fuse`的weight）；E3=`PACE-All+40步bias-only OOF`。
- 固定损失：`lambda_tail=0.03`、`lambda_preserve=0.10`；E1–E3扩展120步；禁止HardPair、Adapter、完整t3、frequency/domain更新和EMA。
- 数据：`p2_min_v1/VALIDATED_ONCE`；receiver=`20-1`；scene=`leo_clear_weak`；seed=`713102`；旧6类K=10，共60条support；独立Query每类20条，共120条。该seed在本轮前未形成SF-TAPFT评分记录，prediction前不连接truth。
- adaptation capsule：`d18-enrollment-before-rx20-1-seed713102-k10-pace`；split：`stage2b-rx20-1-seed713102-before-support-prefix`。
- query capsule：`sf-tapft-pace-rx20-1-s713102-clear-k10-independent20-v1`；split：`p2_min_v1-rx20-1-s713102-clear-old6-k10-independent20`。
- checkpoint：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- Git提交：`f83154f3b503cb012574be36db1bdf51dc65ba7b`（共享工作树中的并行ERBT任务提交时一并纳入本轮已暂存PACE文件；文件内容与51项本地验证一致）。
- 环境/CWD：N607，`/home/szu2070436088/2510044040/CV-SincNet`；Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 本地配置：`configs/stage2_sf_tapft_pace_e0e3_rx20_1_s392002_20260828.json`。
- 远端输入：`runs/stage2_sf_tapft_pace_e0e3_rx20_1_s392002_20260828_r1/inputs`；输出：同run根目录下`E0`–`E3`及`query/<row>`，均不可覆盖。
- GPU：E0/E1/E2/E3分别为GPU0/1/2/3。
- 启动命令：每行使用`code/scripts/run_sf_tapft_slim_matrix_row.py --mode deploy --deployment-inplace --delta-only`；完成delta后使用truth-blind query闭合器产生`DA0_REG0/DA1_REG0`prediction，最后由独立scorer连接truth。
- 预期artifact：各行`selection.json`、`sf_tapft_delta_bundle.pt`、GNU time与GPU采样；各行Query的两状态prediction、prediction receipt、truth-after-prediction和`score.json`。
- 技术停止规则：仅协议/query泄漏、错误receiver/scene/K/split、输出覆盖、错误checkout、无prediction闭合、scorer连接错误、进程归属不清或同一确定性预prediction异常可停止；不得因低性能停止。
- 晋级门槛：相对E0，BA不降、floor不降、最差类别变化不低于-5pp、NLL不高于E0+0.02；warm-resident中位数不超过60秒；delta不超过16KB。

## 当前状态

`ANALYZED`。四行support适配、两状态prediction、truth-last评分和E0常驻10次资源基准均已闭合。

## 最终结论

保留E0=`D0 Compact`，不晋级E1、E2、E3。E0在新seed=`713102`的120条独立Query上，把`DA0_REG0`的BA从72.50%提升到77.50%，floor从55.00%提升到60.00%，NLL从0.857852降到0.743074，ECE从16.63%降到8.98%。E1、E2扩大Norm容量后，BA反而比E0低0.83pp和1.67pp，floor均低5pp；旧D3的增益没有跨seed复现。E3的support OOF bias虽然把support NLL从0.350422降到0.340954，但真实Query BA比E0低4.17pp，类别1准确率低15pp，Query NLL高0.060801，是明确的support过拟合。

## 优化落地内容

### D0后再扩展

新增`pace_expand_start_step`和`pace_norm_rules`。E1–E3严格先执行D0的520步：300步初始阶段、150步fast-tail、70步head polish；step520保存唯一D0 support教师logits，再执行120步扩展。本轮修复了“扩展点位于同一phase内部时不触发重新解冻”的时序问题，E1–E3审计均显示扩展optimizer step恰为120。

### 受保护损失

扩展阶段在CE、LOO-prototype和L2-SP之外加入Top2类别平均损失`lambda_tail=0.03`，以及D0教师到候选的support加权KL`lambda_preserve=0.10`、温度2.0。教师logits全部detach，只读support。当前稳定权重采用D0正确类概率与margin的确定性组合；设计报告中的多次增强方差项未启用，因为增强策略和风险阈值尚未在地面冻结。

### 广义Compact suffix

新增`TimeNormPrefixCache`和`CompactTimeNormSuffix`，E0在`t3.norm`前缓存，E1在`t2.norm`前缓存，E2/E3在`time_fuse.1`前缓存。冻结frequency/domain分支、fusion尾部和identity head辅助量，只重放必要time suffix，不持有完整checkpoint引用。`t2.norm`和`time_fuse.1`边界均通过完整路径logit/梯度等价测试；四行完整backbone训练forward均为0。

### head-only OOF bias

E3只计算一次最终support embedding；4-fold每fold只训练head40步，再用全部OOF logits拟合40步6参数零和bias，总head-only step为200，不重跑backbone。得到bias：

`[-0.253860,-0.398020,-0.090714,-0.008946,+0.465857,+0.285683]`。

### bundle兼容

delta升级为`cvs.sf_tapft.delta.v3`，clean-single升级为`cvs.sf_tapft.clean_single.v3`，两者均显式保存`head_bias`，旧v1/v2缺省零bias。独立P0/P1审查发现并修复了clean-single丢失E3 bias的问题；v3非零bias恢复、旧v2零bias兼容均通过，定点复审PASS。

## 冻结矩阵

|行|训练范围|总步数|扩展步|bias-only步|
|---|---|---:|---:|---:|
|E0|head+`t3.norm` weight/bias|520|0|0|
|E1|E0+`t2.norm.weight`|640|120|0|
|E2|E1+`t1.norm.weight`+`time_fuse.1.weight`|640|120|0|
|E3|E2+6参数零和bias|640|120|200|

HardPair、Adapter、完整t3、frequency/domain和EMA均冻结。E1–E3按预登记强制执行，不使用Query决定扩展或调参。

## 真实Query总体结果

四行`DA0_REG0`完全相同：BA=72.5000%，floor=55.0000%，NLL=0.857852，ECE=16.6269%。

|行|`DA1_REG0` BA|floor|NLL|ECE|相对DA0 BA|floor变化|NLL变化|
|---|---:|---:|---:|---:|---:|---:|---:|
|E0|77.5000%|60.0000%|0.743074|8.9805%|+5.0000pp|+5.0000pp|-0.114778|
|E1|76.6667%|55.0000%|0.745256|9.1162%|+4.1667pp|0.0000pp|-0.112596|
|E2|75.8333%|55.0000%|0.749848|8.0218%|+3.3333pp|0.0000pp|-0.108004|
|E3|73.3333%|55.0000%|0.803875|8.1689%|+0.8333pp|0.0000pp|-0.053977|

E2的ECE最低，但不能把该单指标与E0的BA/floor拼接成“最佳候选”；同row综合最优仍是E0。

## 各类别准确率

每类20条，单条对应5pp。

|状态/行|类0|类1|类2|类3|类4|类5|
|---|---:|---:|---:|---:|---:|---:|
|`DA0_REG0`|11/20=55%|16/20=80%|16/20=80%|11/20=55%|16/20=80%|17/20=85%|
|E0 `DA1_REG0`|13/20=65%|15/20=75%|18/20=90%|12/20=60%|16/20=80%|19/20=95%|
|E1 `DA1_REG0`|12/20=60%|16/20=80%|18/20=90%|11/20=55%|16/20=80%|19/20=95%|
|E2 `DA1_REG0`|12/20=60%|15/20=75%|18/20=90%|11/20=55%|16/20=80%|19/20=95%|
|E3 `DA1_REG0`|12/20=60%|12/20=60%|18/20=90%|11/20=55%|16/20=80%|19/20=95%|

E0相对DA0：类0、2、3、5分别提升10、10、5、10pp，类4不变，类1下降5pp。E1/E2没有获得额外正确样本，E3额外损失类别1的3条正确样本。

## 逐类NLL

|状态/行|类0|类1|类2|类3|类4|类5|
|---|---:|---:|---:|---:|---:|---:|
|`DA0_REG0`|1.5414|0.5063|0.8464|1.1668|0.5633|0.5229|
|E0 `DA1_REG0`|1.5552|0.5913|0.4095|0.9959|0.7436|0.1631|
|E1 `DA1_REG0`|1.4657|0.6220|0.4325|1.0153|0.8088|0.1272|
|E2 `DA1_REG0`|1.4969|0.5654|0.4177|1.0408|0.8597|0.1185|
|E3 `DA1_REG0`|1.6319|0.8072|0.4386|1.0754|0.7681|0.1021|

E0的剩余风险主要是类0和类3。E3的bias显著恶化类0和类1，说明support OOF NLL改善不能替代真实Query校准证据。

## 配对变化

|行|错→对|对→错|始终对|始终错|双侧exact McNemar p|
|---|---:|---:|---:|---:|---:|
|E0|16|10|77|17|0.326940|
|E1|16|11|76|17|0.442068|
|E2|15|11|76|18|0.557197|
|E3|14|13|74|19|1.000000|

E0方向为正但单seed120条下McNemar未达到0.05，不能声明跨seed显著优势。

## 相对E0晋级判定

|行|BA变化|floor变化|最差类别变化|NLL变化|单次墙钟|delta|结论|
|---|---:|---:|---:|---:|---:|---:|---|
|E1|-0.8333pp|-5.0000pp|-5.0000pp|+0.002182|15.38s|6103B|拒绝：BA/floor失败|
|E2|-1.6667pp|-5.0000pp|-5.0000pp|+0.006774|11.55s|7029B|拒绝：BA/floor失败|
|E3|-4.1667pp|-5.0000pp|-15.0000pp|+0.060801|17.16s|7029B|拒绝：BA/floor/类别/NLL失败|

E1–E3资源合格，但科学门槛不合格。

## 资源审计

### 单次同场运行

|行|主优化器可训练元素|实际变化元素|附加bias|cache|suffix forward|完整backbone训练forward|墙钟|最大RSS|delta|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|E0|1152|1152|0|929520B|450|0|12.32s|1594712KiB|5544B|
|E1|1248|1248|0|1666800B|570|0|15.38s|1710364KiB|6103B|
|E2|1368|1368|0|3141360B|570|0|11.55s|1723112KiB|7029B|
|E3|1368|1368|6|3141360B|570|0|17.16s|1717668KiB|7029B|

E2单次时间短于E1是并发调度波动，不能解释为更宽suffix天然更快。

### E0常驻3次预热+10次测量

|指标|中位数|P90|最大值|门槛|
|---|---:|---:|---:|---:|
|墙钟|9.2409s|9.3014s|9.3082s|≤60s|
|进程RSS峰值|1.6668GB|1.6669GB|1.6669GB|≤1.9GiB|
|适配额外RSS|26.6KB|81.9KB|319.5KB|≤64MiB|
|CUDA allocated峰值|168.61MB|168.61MB|168.61MB|≤256MiB|
|CUDA allocated适配增量|151.57MB|151.57MB|151.57MB|≤256MiB|
|CUDA reserved峰值|304.09MB|304.09MB|304.09MB|≤384MiB|
|delta|5544B|5544B|5544B|≤16KB|

resident模型tensor为4199312B，support cache为929520B。能量、星载功率和热降频没有传感器合同，记为`NOT_CAPTURED`，不虚构J、W或温度结论。

## 验证与发布

- 本地聚焦回归：51项通过。
- 真实checkpoint无Query smoke：support=60，FP32 logit/gradient最大差均为0，prediction一致，`query_opened=false`。
- release：`stage2_sf_tapft_pace_e0e3_3b5f06c1.zip`。
- release SHA256：`259d44857d100ca2375395c131c4b9040e692d451e5cc4c174cd5a249a15e271`，本地与N607一致。
- 远端编译：PASS。
- 实现提交：`f83154f3b503cb012574be36db1bdf51dc65ba7b`；PACE追踪提交：`3b5f06c12daa0f317893cf92acd4e08ec7532241`。共享工作树中的并行ERBT任务在本Agent提交前提交了已暂存PACE文件，因此前一提交说明不是PACE专用，但文件内容与51项验证一致。
- N607 run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_pace_e0e3_rx20_1_s392002_20260828_r1`。

## 实现边界与下一步

当前实现完成固定矩阵中的“D0后受保护扩展”和support-only校准，但没有把在线自动PACE状态机晋级为默认。设计要求的风险阈值`tau_L/tau_M`必须在多个地面伪目标接收机上预先冻结，不能从本轮已揭示Query反推。因此本轮不设置触发阈值、不追加同seed调参，也不把E1–E3部署为默认。下一步若继续，应先在多个未进入本轮Query的地面接收机上冻结阈值和稳定权重方差项，再用新seed/新receiver整体评估。当前默认部署候选仍是E0 D0 Compact。
