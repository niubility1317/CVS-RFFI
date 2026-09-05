# ADV3B02配对约束改革：24行实验发布记录

## 目标与矩阵

按用户最新要求，将44行缩减为24行：删除全部14行细参数扫描，以及仅特征／仅伪标签两个拆分候选的6行。保留8种机制，每种3个配对seed：392005、392006、392007，每行200epoch。

|候选|实验含义|行数|
|---|---|---:|
|A_POINT|点约束基准|3|
|B_SAFE|安全区域约束，和点约束互斥|3|
|ASYMMETRIC|非对称配对|3|
|TANGENT|点约束＋切向限制|3|
|ROUTE|点约束＋路由限制|3|
|TANGENT_ROUTE|点约束＋切向＋路由|3|
|POINT_MEMORY|点约束＋特征记忆|3|
|MATCHED_ZERO|相同公共训练配置，配对特征及伪标签损失权重均为0|3|

MATCHED_ZERO是本轮同配置对照，不等同于历史旧版本。共同使用冻结CORE90权重继续训练，不将3个seed描述为3次独立从头预训练。保留成熟ADV3B02公共辅助损失；本轮检验配对改革的增量作用。

## 参数、来源与科学边界

可执行清单：`configs/phase1_adv3b02_pair24_manifest.json`；生成配置：`configs/phase1_adv3b02_pair24_20260905.json`。清单保存24条完整argv、环境、输出路径及显卡分配。

共同预算：E200，label阶段130、pseudo阶段70，batch和数据比例继承已核验MUSE公共基线；配对从epoch11开始，配对伪标签从epoch21开始。pair_weight=0.5、pair_pseudo_weight=0.2、alpha=0.5、U容忍度0.1、teacher_mix=0.25；切向权重0.035、路由权重0.05仅对应候选开启。关闭候选的两项配对权重均为0。

源域L/U/V比例遵循0.07/0.63/0.30；U不使用TX真值，V只读。训练清单保持source-only选择和final-only checkpoint，外部最终评估标志阻止训练器内部目标评估。本队列不会调用旧worker的目标评分流程。

预登记选择：仅完整完成3个seed的候选参加源V比较，以最终epoch三种LEO弱场景准确率的等权均值为主指标。B_SAFE对A_POINT是唯一预设验证性比较；其余机制比较作为探索，不作多重比较下的确认性结论。候选相对A_POINT平均提高至少0.5个百分点，配对增益95%t区间下界大于0，且clean平均下降不超过0.5个百分点，才有资格进入独立确认。df=2的临界值4.30265273。合格候选按主指标排序，并列按候选名；固定使用seed392005的checkpoint，禁止挑seed。无合格候选则不晋升。

本轮不进行目标集候选排序。源V完成后先冻结一个候选，再单独执行truth-last目标确认；本记录不将源域训练完成标为目标实验闭环。显卡并发会影响耗时，不能据此声称独占显卡速度提升。

## 资源分配与技术处理

用户明确授权超过原每卡两个训练进程限制。GPU0—7各固定分配3个新任务，总计24个；seed之间轮换显卡。调度器最多同时启动24个新任务、每卡最多3个本队列任务；已有进程保留。启动时剩余显存须不少于8192MiB，资源暂不足时只等待该行分配的卡，不挪动旧任务。

首次启动只执行一次真实CORE90权重、无query的六路径前后向检查，成功后直接训练。新输出根目录原子创建，已有目录拒绝重启。每行独立日志、PID和输出目录。单行技术失败保留产物、不自动重跑；同类技术错误累计两行则停止新增任务，健康任务继续。低性能不停止。退出0还须读回final_ssdg.pth及epoch200日志，才标记TRAINING_COMPLETE_SOURCE_ONLY。

## 发布路径与状态

run_id：`phase1_adv3b02_pair24_manysig_e200_20260905_r1`。

远端Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，实测Python3.10.19、PyTorch2.1.0+cu121、CUDA可用。

远端项目：`/home/szu2070436088/2510044040/CV-SincNet`；只读代码发布至`releases/<run_id>`；产物至`runs/<run_id>`。数据和初始化权重绝对路径详见清单。发布前两目录均不存在，数据和checkpoint存在。

启动命令：上述Python执行`releases/<run_id>/tools/pair_matrix_start.py releases/<run_id>/configs/phase1_adv3b02_pair24_manifest.json`。状态见输出根目录`status.json`，每行`process.log`、`metrics_epoch.jsonl`和最终checkpoint。

当前记录阶段：本地验证与发布准备；实际启动证据在发布后补充。没有准确率结果，尚不能判断改革是否改善性能。
