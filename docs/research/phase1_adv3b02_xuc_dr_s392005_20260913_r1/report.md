# XUC+DAOT+FastTrust-RC4新矩阵

用户授权：为M14、M11、M05、M08、M07、M12、M13加入DAOT+FastTrust-RC4并发布；完整联合由M08（原C2）、M12（C*+课程）、M13（C*固定课程）分别覆盖。原实验与健康进程全部保留。

数据沿用ManySig equalized1、source RX1/3/4/6/8与day1/2/3，target RX0/2/5/7/9/10/11与day0/1/2/3，L6300/U56700/V27000、6TX；seed392005，E200，FP32，从零训练。目标接收机不进入训练、校准或选模。当前目标benchmark已暴露历史结果，新运行属于用户明确要求的研究迭代，不称为新的盲确认集。

实现路线：在既有CORE90事务化完整目标上调用已固定原生A1的DAOT labeled/unlabeled objective、RC4校准/路由/完整损失。保留各行X/U/C/课程；RC4替换原普通U伪标签分支，避免重复监督。DAOT scale、EMA及辅助头状态只在接受的主更新后提交，EG两次场评估使用同一教师、校准包和随机输入；第二场不能漏DAOT/RC4。U使用原生U256，从E1可用，DAOT与RC4身份约束按原生E21开始；每轮主更新仍为CORE90的49步，公开这与原生222步/轮的预算区别。

|ID|需求|目标文件|状态|验证|
|---|---|---|---|---|
|R1|七个指定移植组合及完整联合|matrix_dr.json|verified|配置逐行检查|
|R2|DAOT原生A1已启用目标真实反传；关闭项不作激活声明|dr_objective.py/runtime.py|verified|非零loss/梯度、完整EG|
|R3|RC4 source-V校准、H/P路由、domain/self目标|dr_objective.py|verified|校准只读、路由/梯度|
|R4|X/U/C及课程保留，不静默关闭|objective.py/tickets.py/runtime.py|verified|受控激活与原方案对应|
|R5|状态单次提交，完整目标用于EG|dr_objective.py/solvers.py|verified|事务状态与回放测试|
|R6|scratch/同数据契约/四场景truth-last|dispatcher与checkpoint重建|verified|真实checkpoint smoke与参数读回|
|R7|本地Git后发布、独立P0/P1审查、保护旧任务|发布工具|verified|CUDA验收、PID/CWD/argv/GPU、增长日志及参数读回|

不因低准确率或C*自然没有动作重跑；只处理可复现技术异常。新run与release不可覆盖。发布前记录精确版本与命令，启动后追加实际激活证据。


## 已冻结的调度与验收

7行依次DR-M14、DR-M11、DR-M05、DR-M08、DR-M07、DR-M12、DR-M13；无M09/M10重复行，完整联合已由M08/M12/M13覆盖，不额外启动第8组。原父行X/U/控制/课程/采样开关逐项同值。

AdamW lr=0.0002、weight_decay=0.0001、梯度裁剪5。采用原生FastTrust legacy LR：E1–5线性预热，E6–160全局余弦降到0.1；E161–180骨干乘0.2，E181–200骨干乘0.05，其余头维持全局0.1。对应E1骨干/其他4e-5/4e-5，E6为2e-4/2e-4，E160为2e-5/2e-5，E161为4e-6/2e-5，E181–200为1e-6/2e-5。学习率与loss权重绑定ticket origin epoch，不随课程排序错误提前退火。

单一source-V在E1/21/41/91/161进行RC4校准；课程窗口在这些边界截断，不跨校准阶段。DAOT和RC4身份E21启用，RC4 domain/self从E1。RC4 all-U域项E91–110恢复，身份项E181–200按原生hard/partial尾部日程退火。EMA decay0.999，仅主更新接受后一次；DAOT尺度同样仅提交一次，EG预测场不污染后续状态。原有卫星日程、E80辅助CE起点和各行C课程保留；DAOT自己的teacher/student视图保持A1定义。

A1固定2-view mean，tangent/nuisance/fingerprint权重0；nuisance head附加用于接口完整性但不声明有梯度贡献。原生RC4路径未更新classification prototype，所以DAOT prototype输入仍为None，并明确不声称prototype目标激活。保留legacy骨干，通过同骨干identity-only接口与原生目标接线，不称为整个native A1 trainer的等预算复现。teacher channel seed等原生参数见resolved_dr_config.json；model/data/evaluation seed为392005。

本地验收：原16项回归PASS；新增3项矩阵/开关对应、LR边界、GPU准入及受控RC4身份梯度测试PASS；真实合成E1/E21/E131/E181 objective+EG+严格自生成checkpoint重建PASS。DAOT、RC4、X/U实际梯度与字段、教师不变、两场完整目标、scale单次提交均检查。真实E1训练入口也PASS。所有合成权重标为诊断，严禁正式初始化使用。

资源策略：每GPU最多2个训练进程且准入前空闲显存至少6500MiB，优先少进程/多显存设备；同时计入其他owner尚未CUDA初始化的单卡train脚本PID及自身预占。当前seedscan owner3521900严格max_processes_per_gpu=1、5运行1排队，不竞争第二槽；旧XUC15仅M09/M10运行，无待训行。此方案不是跨owner原子锁，后续新增owner仍须统一检查并发。所有健康任务保留。7行全部预测固定（4704000条）后再独立评分。首次5b0f03fe发布已落地但因没有空闲GPU在CUDA smoke前安全退出，无正式launch/run，产物保留；新版本按上述有界第二槽策略发布。

N607 preflight：普通账号、目标主机/CWD核对通过，约7185GiB空闲磁盘；新run尚不存在，旧M09/M10仍为原PID3317646/3317657。源数据角色直接复用已有VALIDATED_ONCE物理契约，训练入口逐行比较角色，不重建目标包。

合成短程模型未自然通过RC4 H/P校准门槛，不能把该短程检查当作自然身份选中证据。另用受控H/P路由验证E1身份为0、E21 hard/partial损失及logit梯度非零；该测试只验证分支接线，不影响正式门槛。正式运行中逐步记录hard/partial实际选中数，E21前只要求RC4 domain/self路径实际执行。


## 发布结果与独立读回

2026-09-13T11:13:49.657724+08:00：VERIFIED。执行commit=4d3880208e40e0e8a875284e2dc31e41ad0d54f4，release=/home/szu2070436088/2510044040/CV-SincNet/releases/adv3b02_xuc_dr_4d3880208e；run=phase1_adv3b02_xuc_dr_s392005_20260913_r1，唯一dispatcher PID3535115。7组全部RUNNING，0排队，0失败；不是训练完成或评分完成。普通账号UID1000，owner和各worker的CWD/argv匹配发布目录。前次仅落地的5b0f03fe目录保留，未启动正式run；本次4d388020为唯一正式执行版本。

|实验|GPU|PID|接受更新记录数|最新总loss|RC4总项|
|---|---:|---:|---:|---:|---:|
|DR-M14|2|3535129|112|17.29100|0.80433|
|DR-M11|3|3535201|117|17.19502|0.77050|
|DR-M05|5|3535275|100|17.39212|0.80230|
|DR-M08|6|3535759|96|17.50929|0.80274|
|DR-M07|7|3535834|91|17.30324|0.77403|
|DR-M12|1|3535908|89|17.49346|0.83185|
|DR-M13|0|3535981|79|17.71600|0.81486|

两次读回确认所有行actions日志增长，最新更新accepted=true、loss/grad有限，训练日志无Traceback/RuntimeError/CUDA error。每行resolved_config的xuc_row与冻结matrix逐字段相同，scratch_only=true、checkpoint_sources为空、target_contact=false、RC4辅助头已进入optimizer；X/U开关与父行一致。旧M09/M10和其他8卡原有进程全部存活，每GPU至多2个CUDA进程。

远端CUDA诊断在E1/21/131/181验证完整目标与EG、教师状态、scale单次提交和自身checkpoint严格重建，PASS。正式训练当前仍在warmup，实际RC4 domain/self非零，DAOT和身份项未到origin E21而为0，符合配置；没有把诊断中的E21激活等同于正式行已经达到E21。

每小时监控xuc15已更新为同时覆盖原15组与新7组，频率每小时一次，状态保持PAUSED。此前关于恢复监控的询问尚未收到回答，因此目前不会自动执行巡检/修复；训练dispatcher本身不依赖该定时任务，继续运行并在E200后自动预测、评分。

验收证据：delivery_verification.json、live_launch_initial.json、live_launch_verification.json、remote_cuda_acceptance.json；完整矩阵matrix_dr.json。初始速率不能可靠外推C*审计/EG与后期DAOT成本，暂不承诺完成时间。
