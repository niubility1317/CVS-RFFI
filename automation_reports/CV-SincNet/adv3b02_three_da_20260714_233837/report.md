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
- 不允许用target query选择epoch或超参数；固定200步属于预注册预算。

## 7.N607预检与烟测结果

预检时间2026-07-14 23:39 HKT。GPU0–3各有一条既有RIEI任务；GPU4–7空闲。本任务未干预既有进程。N607未安装名为ssr-gpu的环境，正式远端运行使用服务器现有且已验证的CVS-RFFI环境（Python路径/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python，torch2.1.0+cu121，CUDA可用）；本地代码测试仍使用ssr-gpu。

同步文件及SHA256：

|远端相对路径|SHA256|
|---|---|
|paper_reproduction/cvs_aligned/adv3b02_supervised_da_runner.py|1270dbdb40285393519796a65a4f9bce3a0a89debdfce0e9a3ca1521a930a9db|
|paper_reproduction/cvs_aligned/supervised_da.py|e1c36c7e52bc2e5b34dc4953254c371c20a29b97b8e1670691ffc28c5e770419|
|paper_reproduction/scripts/run_cvs_publication_matrix.py|97b983fae3d19420baa4bfa525be8b61f5b31e06447c2f081d894a38ea966b81|
|paper_reproduction/configs/adv3b02_stage2b_three_da_20260714_n607.json|734eb583d55b4cb5fd1b0fb849fe76a876f20e2dc7f1cbcdfb5f64477858a4d1|

三条烟测均使用receiver=20-1、K=1、seed=713101、每场景2个适应步；输出位于runs/adv3b02_three_da_20260714_233837/smoke。

|方法|GPU|最终状态|适应前old_acc|2步后old_acc|变化|损失检查|
|---|---:|---|---:|---:|---:|---|
|ProtoNet CDA|4|artifact-complete|69.72%|49.72%|-20.00pp|原型注册无梯度|
|MRIOR-SDA|5|artifact-complete|69.72%|53.61%|-16.11pp|DV-KL=0.52–1.02，全部有限|
|DADDA-SDA|6|artifact-complete|69.72%|68.33%|-1.39pp|MMD、LMMD、alpha、总loss全部有限|

烟测只验证接口、严格加载、梯度更新、损失稳定性与artifact契约，不用于正式方法排名。

## 8.正式矩阵启动设计

|worker|GPU|方法|shard|预计行数|
|---|---:|---|---|---:|
|proto|4|ProtoNet CDA|1/1|125|
|mrior0|5|MRIOR-SDA|0/2|约63|
|mrior1|6|MRIOR-SDA|1/2|约62|
|dadda0|7|DADDA-SDA|0/2|约63|
|dadda1|4|DADDA-SDA|1/2|约62|

每个worker通过paper_reproduction/scripts/run_cvs_publication_matrix.py启动，并使用--module-override paper_reproduction.cvs_aligned.adv3b02_supervised_da_runner。输出根为runs/adv3b02_three_da_20260714_233837/formal，日志根按方法/分片隔离；每个子任务完成后立即退出Python进程，不保留SSH连接。
每个worker通过paper_reproduction/scripts/run_cvs_publication_matrix.py启动，并使用--module-override paper_reproduction.cvs_aligned.adv3b02_supervised_da_runner。输出根为runs/adv3b02_three_da_20260714_233837/formal，日志根按方法/分片隔离；每个子任务完成后立即退出Python进程，不保留SSH连接。

## 9.全量artifact审计

|检查项|结果|
|---|---|
|正式行数|375/375|
|方法计数|ProtoNet CDA=125，MRIOR-SDA=125，DADDA-SDA=125|
|逐样本score|135000条|
|loss trace|8625条|
|checkpoint SHA256|2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98，唯一|
|严格加载|0 missing、0 unexpected、0 shape mismatch|
|support/query重叠|0|
|query用于训练/模型选择|0|
|非有限loss|0|
|worker失败|0|

完整逐行结果见artifacts/per_run_results.csv；其余K、receiver、scenario、class和loss汇总均位于本报告artifacts目录。

## 10.核心结果

本矩阵同一query上的ADV3B02直接分类共同基线为75.21%；它是本次严格配对的主要参考。历史73.87%来自另一直接评测聚合口径，只作外部参考，不能替代本次逐任务before值。

|方法|适应后old_acc|95%CI|相对共同基线|胜/平/负任务|算法适应时延/行|
|---|---:|---:|---:|---:|---:|
|MRIOR-SDA|79.11%|77.31%–80.92%|+3.90pp|74/2/49|16.68s|
|DADDA-SDA|76.91%|75.18%–78.64%|+1.70pp|70/1/54|13.04s|
|ProtoNet CDA|67.87%|65.69%–70.04%|-7.34pp|10/0/115|0.031s|
|ADV3B02直接分类|75.21%|73.42%–77.00%|0|—|0|

同一125任务配对：MRIOR比DADDA平均+2.20pp，85胜/40负；MRIOR比ProtoNet平均+11.24pp，117胜/8负；DADDA比ProtoNet平均+9.04pp，125/125全胜。

## 11.K-shot曲线

括号内为相对同任务ADV3B02直接分类的变化。

|方法|K=1|K=2|K=5|K=10|K=20|
|---|---:|---:|---:|---:|---:|
|MRIOR-SDA|69.88%（-5.33pp）|73.71%（-1.50pp）|79.17%（+3.96pp）|84.50%（+9.29pp）|88.31%（+13.10pp）|
|DADDA-SDA|72.58%（-2.63pp）|73.27%（-1.94pp）|76.74%（+1.53pp）|79.36%（+4.14pp）|82.61%（+7.40pp）|
|ProtoNet CDA|59.47%（-15.74pp）|65.68%（-9.53pp）|70.28%（-4.93pp）|70.86%（-4.36pp）|73.07%（-2.14pp）|

关键结论：MRIOR和DADDA都存在低K负迁移，K=5后均值才稳定转正；MRIOR对support数量最敏感，但K=10/20增益最大。ProtoNet随K增加而改善，但K=20仍未超过原ADV3B02分类边界。

## 12.Receiver与场景结果

|方法|20-1|3-19|7-14|7-7|8-8|
|---|---:|---:|---:|---:|---:|
|MRIOR-SDA|77.90%（+11.57pp）|66.67%（+3.89pp）|85.78%（-3.61pp）|84.28%（+1.22pp）|80.94%（+6.44pp）|
|DADDA-SDA|70.83%（+4.50pp）|64.48%（+1.70pp）|88.19%（-1.20pp）|81.87%（-1.19pp）|79.19%（+4.69pp）|
|ProtoNet CDA|59.61%（-6.72pp）|53.28%（-9.50pp）|84.06%（-5.33pp）|73.78%（-9.28pp）|68.62%（-5.88pp）|

3-19仍是最困难接收机。7-14的直接分类本来很强，MRIOR和DADDA均出现负迁移，说明统一固定步数适应会破坏已经对齐的决策边界；由于禁止使用query选模，本实验没有针对7-14提前停止。

|方法|clear|low_elev|rain|
|---|---:|---:|---:|
|MRIOR-SDA|81.29%（+4.12pp）|78.18%（+3.75pp）|77.87%（+3.84pp）|
|DADDA-SDA|79.39%（+2.22pp）|75.49%（+1.06pp）|75.85%（+1.82pp）|
|ProtoNet CDA|70.75%（-6.41pp）|66.25%（-8.18pp）|66.60%（-7.43pp）|

MRIOR和DADDA在三种LEO场景的平均增益均为正；ProtoNet在低仰角和雨衰下退化最明显。

## 13.类级与损失诊断

类级平均准确率中，MRIOR的六类范围为66.00%–92.64%，DADDA为64.28%–91.45%，ProtoNet为55.53%–81.56%。20-19、14-7和14-10仍是主要难类；8-20、6-15和20-15相对稳定。

|方法|loss范围|关键对齐项范围|判断|
|---|---:|---:|---|
|MRIOR-SDA|-0.319–4.378|DV-KL=-228.12–77.20；estimate zeta=-254.89–72.55|全部有限，已消除历史百万级负DV-KL塌缩；估计器仍高方差|
|DADDA-SDA|0.027–9.027|MMD=0.0137–0.2472；LMMD=0.1074–0.6095；alpha=0.0099–0.0975|稳定有限，动态权重主要偏向全局对齐|
|ProtoNet CDA|0|无梯度原型注册|计算最轻，但目标原型不足以保持ADV3B02决策边界|

MRIOR的交替极小极大更新修复是必要的：T先上升、再冻结T让ADV3B02/分类器下降后，没有再出现旧runner的百万级负目标值。不过DV-KL仍有较大振幅，因此结果可用于方法比较，但若后续做工程部署，应另行预注册梯度裁剪或稳定MINE消融，不能事后用query挑选。

## 14.结论与建议

1.在共享严格ADV3B02基座、相同125任务和固定200步预算下，MRIOR-SDA总体最佳，79.11%，相对直接分类+3.90pp。
2.DADDA-SDA更稳健且计算稍低，76.91%，相对直接分类+1.70pp；在K=1/2时虽仍负迁移，但幅度小于MRIOR。
3.ProtoNet CDA不适合直接替换ADV3B02分类头：总体-7.34pp，125任务中115个退化；K=20仍低2.14pp。
4.推荐路线是K≥5使用MRIOR；若更重视低K稳定性和较低计算，优先DADDA。K≤2默认保留直接ADV3B02分类头，除非使用support-only安全门。
5.7-14等高基线域需要support-only的保守更新或回退门，不能用target query做早停。当前结果只支持Stage2-B旧类域适应，不支持新类增量、unknown拒识或星上部署成功声明。
