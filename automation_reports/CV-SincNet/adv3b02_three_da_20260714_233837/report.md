# ADV3B02三种域适应方法Stage2-B实验报告

## 1.实验信息

|字段|内容|
|---|---|
|实验ID|adv3b02_three_da_20260714_233837|
|时间|2026-07-14 23:38 HKT|
|操作者|Codex|
|目标|以同一个严格ADV3B02 checkpoint作为特征提取器，对比ProtoNet CDA、MRIOR-SDA、DADDA-SDA在Stage2-B目标旧类上的域适应效果|
|科学边界|ADV3B02-backbone extension；不是三篇方法原始网络结构的paper-faithful复现|
|根目录版本状态|E:\type10-7不是Git仓库；本报告同步到Git承载面github_publish/CVS-RFFI-repo|

## 2.假设与公平对照

三组共享checkpoint、source数据、target receiver、K-shot support/query切分、LEO场景和seed。查询标签仅在预测后计算指标，不参与训练或模型选择。

|方法|ADV3B02用法|目标域方法|目标阶段梯度|
|---|---|---|---|
|ProtoNet CDA|严格ADV3B02的z_id160|每类K-shot support均值作为目标原型，欧氏最近原型分类|0；符合ProtoNet支持集原型注册|
|MRIOR-SDA|严格ADV3B02 ID分支+原分类器|每步先对估计器T执行7次DV-KL上升，再冻结T对ADV3B02/分类器执行1次source CE+target-support CE+DV-KL下降|200步/场景|
|DADDA-SDA|严格ADV3B02 ID分支+原分类器|z_id用于全局MMD，ID子分支特征拼接用于局部LMMD，动态alpha联合source CE与target-support CE|200步/场景|

历史直接地面模型旧类均值73.8667%作为无适应参考；每个新实验还保存同一query在适应前的直接分类准确率用于严格配对。

## 3.数据协议

- Stage2-B，仅target-old六类：14-10、14-7、20-15、20-19、6-15、8-20。
- 目标接收机：20-1、3-19、7-14、7-7、8-8。
- K={1,2,5,10,20}，seed={713101,713102,713103,713104,713105}。
- 场景：leo_clear_weak、leo_low_elev_weak、leo_rain_weak。
- 每方法125行、三方法共375行；每行三个场景、每场景六类×20 query。
- support/query使用固定support_pool_max_k=20的嵌套切分，K变化不改变query。
- target query不用于训练、阈值、模型选择或联合推断。

## 4.实现与验证

|本地文件|用途|
|---|---|
|paper_reproduction/cvs_aligned/adv3b02_supervised_da_runner.py|严格重建ADV3B02并运行三种方法|
|paper_reproduction/cvs_aligned/supervised_da.py|MRIOR原生极小极大更新顺序|
|paper_reproduction/scripts/run_cvs_publication_matrix.py|支持把完整矩阵路由到共享ADV3B02 runner|
|paper_reproduction/configs/adv3b02_stage2b_three_da_20260714_n607.json|正式矩阵配置|
|tests/test_adv3b02_supervised_da_runner.py及相关测试|协议、原型、更新顺序和路由回归测试|

本地验证：

```text
conda activate ssr-gpu
python -m pytest tests/test_adv3b02_supervised_da_runner.py tests/test_cvs_supervised_da.py tests/test_cvs_supervised_da_runner.py -q
17 passed
```

dry-run通过。正式checkpoint和数据仅位于N607，因此本地只完成协议/接口验证，严格重建与显存烟测在N607执行。

## 5.N607执行计划

|字段|计划值|
|---|---|
|远端工作目录|/home/szu2070436088/2510044040/CV-SincNet|
|Conda环境|ssr-gpu|
|checkpoint|runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth|
|输出根|runs/adv3b02_three_da_20260714_233837|
|日志根|logs/adv3b02_three_da_20260714_233837|
|并发策略|先读GPU/进程状态和三方法单行烟测；正式任务不超过每GPU两个训练进程|
|成功条件|375/375 artifact-complete；严格加载0 missing/0 unexpected/0 mismatch；无非有限loss；support/query零重叠；每方法K档25行|
|提前停止|任何协议违规、非有限loss、严格checkpoint失败或artifact contract失败立即停止相应worker|

待烟测后补充精确GPU、PID、启动命令、耗时估计和正式矩阵结果。

## 6.风险

- 端到端ADV3B02适应显著重于冻结特征头，完整375行可能是小时至天级任务；先用真实单行耗时决定分片。
- ProtoNet目标阶段无梯度是方法定义，不代表实验忘记解冻。
- MRIOR/DADDA保留方法目标，但特征网络替换为ADV3B02，因此结果只能解释为共享ADV3B02上的方法比较。
- 不允许用target query选择epoch或超参数；固定200步属于预注册预算。
