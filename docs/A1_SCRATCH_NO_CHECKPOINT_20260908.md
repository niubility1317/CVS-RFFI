# A1-R3：无需checkpoint的从零训练实验

## 目标与来源

按用户“设计无需checkpoint重新训练的实验，然后发布”和“融入R3的设计”执行。已读取引用任务“制定ADV3B02因子化重构改造计划”（01a05d35-4b69-78e1-8fc4-73553c6d5777），并追踪固定实现`8f1de7971853aa9650e4f83d6ad979f359c434c2`的FCR因子、损失、schedule及`code/train.py`真实调用。

学生随机初始化，EMA从本轮学生内存副本开始；不预训练CORE90、不加载历史或其他行权重。融合R3的canonicalizer/content/fingerprint/nuisance结构和control decoder；分类、身份对抗、EMA、RC4、DAOT统一使用`z_f_id`。这是A1+R3+无锚点预热的新组合，不是历史R3严格复现。

## 冻结矩阵

run_id=`a1_r3_scratch_s392005_20260908_r1`。

|行|GPU|R3辅助权重|执行|预算|
|---|---:|---:|---|---:|
|R3_STRUCTURE_CONTROL|1|0|legacy|E200|
|R3_REFERENCE|2|1|legacy|E200|
|R3_FAST_SEQUENTIAL|3|1|identity_sequential、批量尺度/日志读回、mean冗余项跳过|E200|

三行同架构M/lite_d/dual、160维身份特征、6个TX、256点输入、seed392005和R3结构。CONTROL仍执行相同辅助前向，匹配BN和随机数消耗，仅辅助loss乘0；它是辅助目标对照，不是无R3结构的原模型。REFERENCE与FAST只差执行flags。

ManySig equalized=1；source receiver=1,3,4,6,8、day=1,2,3；target receiver=0,2,5,7,9,10,11、day=0,1,2,3；TX=0,1,2,3,4,5。split=`tx_rx_day_1_7_2`、`legacy_l_u_v`；L/U/V=0.07/0.63/0.30，预期6300/56700/27000。原A1 sat_seed/sat_view_seed保持2027；新增辅助视图generator由seed/epoch/batch/role独立确定。L batch128、U batch256；其他A1配置继承冻结矩阵，不依据target成绩修改。

## 从零训练与R3目标

`from_scratch=true`、`a1_scratch_only=true`、`use_a1_r3=true`；baseline_ckpt/teacher_ckpt为空；`rc4_use_anchor=false`、`rc4_cache_anchor_logits=false`、`rc4_lambda_feature_anchor=0`、`sat_anchor_ssl=false`；冻结teacher clean/sat KL和zid MSE为0。训练guard拒绝checkpoint依赖；只允许训练结束后加载本行E200产物进行独立评估。

RC4保留EMA双弱视图和source V校准，无锚点公式的anchor槽用EMA weak1，并非独立第三教师。E1校准，前20轮关闭RC4身份伪标签；E21重新校准后启用身份项，后续E41/E91/E161更新。DAOT前20轮轨道权重为0，后续原日程保留。监督、无标签domain/self目标和EMA仍工作，不称预热期只有CE。

R3辅助API只接收source IQ、域和物理ID，不接收TX标签。L/U从同一物理片段生成clean→LEO对，crop一致；U不建立fingerprint/transplant配对。辅助对全部执行，区别于原A1主增强的概率采样。场景课程为E1–40 clear、E41–90 low-elev/rain轮换、E91–200三弱场景轮换。

self=双路平均复高斯NLL+平均MR-STFT+平均相位增量；swap=双向重构NLL；shared=content与归一化fingerprint的对称stop-gradient距离。cross decode使用源content、目的fingerprint和目的nuisance；激励到指纹响应保留detach。control decoder为`u_hat=s_hat+delta_f`，不是完整物理解码链。

|阶段|self|swap/shared|说明|
|---|---:|---:|---|
|E1–40|1|0|重构预热|
|E41–90|1|从0线性到1|E41仍为0|
|E91–150|偶数实际更新计数为1，奇数为0|同左|保留原R3日程；奇数步无R3必要性目标，A1主目标继续|
|E151–200|0.25|swap=0.25，shared=1|身份精修|

η没有匹配的合法监督映射，显式`eta_valid=0`、ηloss=0，不能算作激活。cycle、necessity、transplant、完整physics正则、三轴及factor反塌缩不属于本轮R3。FCR因子和复杂数损失保持FP32/complex64，外层骨干可AMP。

每轮新增`train/r3_l_s_*`和`train/r3_u_s_*`：各分量原始/加权loss、pair数、η有效数、阶段激活和辅助总loss。真实激活以日志/梯度为据。独立审查发现EMA `.data`更新不能使CosFace权重缓存失效；本轮三组共同改为no_grad下原参数原地更新，使版本递增，旧非R3分支保持历史行为。已运行任务的release不变。

## 评价与发布

主比较FAST−REFERENCE的执行一致性、总训练耗时/峰值显存和四场景性能；REFERENCE−CONTROL观察辅助目标贡献。历史checkpoint续训只作不同初始化背景；单seed不晋级默认。

训练只用source V，不用target选模。最终本行E200 checkpoint预测clean+三个`leo_*_weak`，每场景168000条；复用已验证的无标签opaque输入包。prediction固定后独立scorer连接truth，结果不回流训练/调参/重跑。数据不变，不重复数据验证。

入口：`python code/scripts/run_a1_scratch_no_checkpoint.py --project-root /home/szu2070436088/2510044040/CV-SincNet --run-id a1_r3_scratch_s392005_20260908_r1`。Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD为Git独立release。已有root即拒绝；产物`runs/<run-id>/`，日志`logs/<run-id>/`。GPU4/5现有selected-checkpoint任务保持运行。

配置/数据边界错误、checkpoint意外依赖、无法执行、非有限数、OOM或确定性异常属于技术失败，保留现场不自动重试；低性能不停止。批量教师不加入本轮。

## 设计落地追踪

|要求|实现与验证|状态|
|---|---|---|
|无checkpoint训练|scratch guard、runner、随机M/lite_d与RC4真实函数检查|本地通过，checkpoint_loads=0|
|R3身份统一|a1_r3_model、model_dual、post_stage_common|主输出/identity-only/非aux/exact重建检查通过|
|source辅助loss|a1_r3_objective与SSDG L/U主loss接线|各阶段、有限梯度、零权重对照检查通过|
|执行配对|check_a1_scratch_execution、check_a1_fast_execution --use-a1-r3|EMA修复后CPU/CUDA通过，FP32/AMP教师误差0；E1/E21/E161更新对比通过|
|独立P0/P1|本候选有界审查|1项EMA缓存P1已修复并定点闭合，91项测试通过，无剩余P0/P1|
|Git与N607|验证后提交push，再发布与GPU/log/PID读回|VERIFIED，详见下方启动记录|
|科学结果|完整E200日志与最终prediction/score|未完成，启动不等于结论|

## N607启动记录（2026-09-08）

状态：`RUNNING`，发布`VERIFIED`；三组E1均已完成。运行代码commit=`9b16413edd0d2031d7d6a43a1c687117a0cb47d3`，远端分支已独立读回一致；release=`releases/a1_r3_scratch_9b16413e`，归档SHA256=`ad1c42c010d6d4cc93a36626765b6a96728bf3c62993ec22523b26d792606f84`，本地/远端一致。

dispatcher PID=3850390。三行PID/CWD、实际GPU进程、启动日志、CUDA执行检查及首轮CSV已独立读回，完整证据见`analysis/a1_r3_n607_startup.json`。服务器PyTorch2.1.0+cu121，R3 AMP自重构/交换/共享真实函数的有限梯度检查通过；此为合成执行正确性证据，不是泛化结果。

|行|GPU|PID|E1秒数|E1 R3 L/U加权loss|有效更新|
|---|---:|---:|---:|---|---|
|R3_STRUCTURE_CONTROL|1|3850661|123.33|0/0|221/222|
|R3_REFERENCE|2|3850666|115.45|28.449008/28.448537|221/222|
|R3_FAST_SEQUENTIAL|3|3850671|116.37|28.449069/28.448600|221/222|

三组E1首batch出现一次非有限梯度，首次位置`id_backbone.sinc.low_hz_`，loss均有限；既有AMP保护跳过该次更新并将scale降到32768，随后221步成功，未出现非有限loss。`first_rc4_anomaly.pt`保留在各行输出目录，没有加载它恢复训练或重启任务。这里不能把“CUDA合成检查通过”写成“真实训练零异常”。零星AMP降尺度由现有训练器处理；持续非有限数或无法有效更新才属于需恢复的执行失败。

真实E1的REF/FAST有微小浮点差异，不能据合成输入误差0宣称跨GPU全程逐位相同；首轮速度也未显示加速，因为DAOT尚未进入E21后的有效日程。后续需按完整日志检查耗时、收敛与运行时机制。swap/shared将在后续日程开启，η仍明确为0。

GPU4/5原selected-checkpoint进程3823617/3823622仍在运行，本次未修改其release、参数或输出。新实验尚无E200结果、prediction/score或科学晋级结论。

最终交付前读回已到E2：CONTROL有效更新222/222，REF/FAST各221/222（各再跳过一次非有限梯度），三组非有限loss均为0且进程仍运行。E2 R3 L/U加权loss：CONTROL=0/0，REF=15.216146/15.213927，FAST=15.219982/15.217772。当前异常属于已自动跳过的零星梯度事件，不能宣称零异常，也不能据此称持续技术失败；保留原保护、异常包和所有输出，不重启或干预健康任务。最新JSON包含E2快照，表格保留E1初次启动证据。
