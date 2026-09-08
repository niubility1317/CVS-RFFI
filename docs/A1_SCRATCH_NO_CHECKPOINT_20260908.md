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
|执行配对|check_a1_scratch_execution、check_a1_fast_execution --use-a1-r3|CPU教师误差0；E1/E21/E161更新对比通过，EMA修复后复验|
|独立P0/P1|本候选有界审查|1项EMA缓存P1已修复并定点闭合，91项测试通过，无剩余P0/P1|
|Git与N607|验证后提交push，再发布与GPU/log/PID读回|待发布证据|
|科学结果|完整E200日志与最终prediction/score|未完成，启动不等于结论|
