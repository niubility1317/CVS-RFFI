# qKNNV42逐任务support-only适应875任务正式重跑

本文件是根目录正式报告`E:\type10-7\automation_reports\CV-SincNet\qknnv42_support_only_taskadapt_875_20260715_v1\report.md`的Git承载镜像。完整内容、运行协议、验证证据、N607命令与后续结果均以根目录报告为工作面，并在每次状态更新后同步到本文件。

## 当前摘要

- 状态：875/875正式任务完成，完整协议、artifact、loss与日志审计PASS。
- Git提交：`08bce60`(875矩阵)、`62ee379`(结果审计器)、`6a7a19a`(loss汇总)、`5c85366`(严格ADV3B02逐任务审计)。
- 矩阵：125个单qKNN+FFT96基线+6×125个逐任务support-only适应=875。
- 类别：6个旧类+2个已注册新类。
- 训练数据：只允许当前(receiver,seed,K)任务的目标receiver LEO support；禁止clean、source、proxy和query。
- 推理：单视图、FFT96、禁dense query、无角色Oracle、无类别配额。
- 本地验证：33项pytest通过，Python编译、Bash语法和875矩阵dry-run均通过。
- 结果：待运行完成后补充完整主表与行级artifact索引。

## N607预检与同步摘要

- 直连预检PASS；8张RTX 3090空闲，无活动训练；`/home`剩余7.6TB。
- checkpoint、ManySig、ManyTx、远端Python导入、协议校验和Bash语法均PASS。
- 两个被更新的远端旧脚本已下载到本地快照。
- 6个运行文件已同步，逐项本地/远端SHA256一致；完整映射与哈希见根目录正式报告。
- prepare PID`1231463`已正常退出；三份LEO raw IQ+FFT96母缓存生成并通过审计。
- 每场景5,600行、目标行3,200；每个receiver×类别最少65行；三场景物理ID一致；无clean视图；严格ADV3B02加载为0/0/0。
- 正式manifest为875任务，125基线+750逐任务适应，clean=false、query-fit=false。
- 8个shard已启动，PID为`1233044–1233051`，GPU0–7各一个worker；首批显存506–686MiB。
- 启动约15秒时125个基线已完成，E2完成8项，无失败；子进程任务键包含独立receiver/seed/K/epoch。

## 最终结果

| arm | old_acc | new_acc | H | ΔH配对95%CI |
|---|---:|---:|---:|---:|
| 单qKNN+FFT96 | 71.344% | 58.247% | **63.168%** | — |
| E2 | **71.347%** | 58.207% | 63.140% | −0.028±0.067pp |
| E5 | 71.322% | 58.133% | 63.088% | −0.079±0.109pp |
| E10 | 71.293% | 58.087% | 63.045% | −0.123±0.173pp |
| E20 | 71.287% | 58.007% | 62.989% | −0.178±0.200pp |
| E30 | 71.242% | 58.040% | 62.981% | −0.187±0.224pp |
| E60 | 71.073% | **58.247%** | 63.075% | −0.092±0.270pp |

审计：875/875评测、750/750训练、15,875/15,875逐epoch loss全部存在；905个日志文件共40,022行无failed、Traceback、OOM、Killed或nonfinite。750个adapter均严格ADV3B02加载0/0/0，只用任务LEO support，禁clean/source/proxy/query，禁dense query与Oracle。

结论：最强总体H仍是无adapter单qKNN+FFT96。适应loss随epoch稳定下降，support准确率最高提升约1.69pp，但query H无稳定提升；这是support拟合未转化为query泛化，不是训练未执行。adapter只有154参数、308B FP16状态和34,816MAC/query，资源很轻，但当前没有性能收益。

详细逐K表、全量loss表、资源解释、artifact路径与SHA256见根目录正式报告及本地汇总目录`E:\type10-7\local_artifacts\qknnv42_support_only_taskadapt_875_20260715_v1_summary`。
