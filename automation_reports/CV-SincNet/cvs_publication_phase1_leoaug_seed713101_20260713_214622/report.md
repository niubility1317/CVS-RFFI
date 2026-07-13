# CVS Phase1星地增强重跑报告

## 实验身份

- run ID：`cvs_publication_phase1_leoaug_seed713101_20260713`
- 设计时间：2026-07-13 21:46 HKT
- 操作者：Codex
- 阶段：Phase1域泛化对比重跑
- 方法：CVCNN-CE、RIEI-FD、DRIFT
- 对照：`cvs_publication_phase1_seed713101_20260713`中的No-Sat-Aug三方法结果
- 目标：在不改变数据划分、seed、epoch、模型结构和正式测试场景的条件下，仅开启训练阶段星地信道增强，重新获得正式Phase1对比结果。

## 假设与声明边界

训练批次同时包含原始clean view和由同一源域样本生成的LEO satellite view。若旧结果的主要瓶颈是clean训练与LEO测试之间的分布失配，则三种方法的三场景星地准确率应显著高于No-Sat-Aug对照。

本实验是CVS-aligned extension，结果名称必须写为`CVCNN-CE+LEO-Aug`、`RIEI-FD+LEO-Aug`、`DRIFT+LEO-Aug`，不得冒充论文原始训练结果。所有LEO场景仍属于source-synthetic heldout channel stress，不是真实在轨链路验证。

## 固定协议

|字段|设置|
|---|---|
|数据|ManySig，equalized=1，out_len=256|
|源域训练receiver|0–6|
|目标测试receiver|7–11|
|训练day|0,1|
|测试day|2,3|
|训练比例|0.1，8400条clean源域样本|
|验证比例|0.9，75600条source validation样本|
|正式main OOD测试|204000条/场景|
|seed|713101|
|epochs|200|
|训练LEO场景|`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`|
|增强概率|`sat_view_prob=1.0`|
|训练批次语义|每个clean batch拼接一个随机LEO场景增强副本|
|正式测试LEO场景|同上三场景，`sat_seed=2027`|
|clean视图|训练与验证控制；不进入正式星地主结果|

## checkpoint选择规则

正式checkpoint必须为`best_by_val.pt`，选择规则固定为200个epoch中source validation TX accuracy最高的epoch：

1. `best_metric=acc`；
2. `BestValTestGate(mode=max)`；
3. 仅当未取整的`val_stats["tx_acc"]`严格提高时覆盖`best_by_val.pt`；
4. 不使用test、LEO test、receiver test或最终epoch进行模型选择；
5. 正式详细评估必须读取该`best_by_val.pt`并记录checkpoint epoch与SHA256。

## 本地版本与验证

- Git仓库：`E:\type10-7\github_publish\CVS-RFFI-repo`
- 分支：`codex/cvs-rffi-release-20260626`
- 本轮无需修改训练代码；复用已支持`--use_sat_channel_view_aug`的Git-backed launcher和baseline入口。
- `conda run -n ssr-gpu python -m pytest -q tests/test_baseline_training_behaviors.py tests/test_cvs_rffi_launcher.py`：22通过、1跳过。
- `bash -n scripts/launchers/run_cvs_baseline_queue.sh`：通过。
- 本地dry-run确认三种方法均携带：
  - `--epochs 200`
  - `--seed 713101`
  - `--use_sat_channel_view_aug`
  - `--sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak`
  - `--sat_view_prob 1.0`
  - 三个同名正式LEO评估场景。
- checkpoint代码审计：`baselines/common/cvs_trainer.py`默认`best_metric="acc"`，以source validation TX accuracy严格提高为保存门。

## N607计划

- 远端cwd：`/home/szu2070436088/2510044040/CV-SincNet`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU：CVCNN-CE→GPU0，RIEI-FD→GPU1，DRIFT→GPU2。
- run root：`paper_reproduction/runs/cvs_publication_phase1_leoaug_seed713101_20260713`
- log root：`paper_reproduction/logs/cvs_publication_phase1_leoaug_seed713101_20260713`
- 预期输出：每方法`metrics.json`、`best_by_val.pt`，以及scheduler log和manifest。
- 启动前PID：待预检后填写。
- 启动命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet && \
METHODS=cvcnn_ce,riei_fd,drift \
GPU_IDS=0,1,2 \
PYTHON_BIN=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python \
SEED=713101 STANDARD_EPOCHS=200 \
SAT_SCENARIOS=leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
SAT_VIEW_AUG=1 \
SAT_TRAIN_SCENARIOS=leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
SAT_VIEW_PROB=1.0 SAT_VIEW_SEED=2027 \
RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/runs/cvs_publication_phase1_leoaug_seed713101_20260713 \
LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/logs/cvs_publication_phase1_leoaug_seed713101_20260713 \
nohup bash scripts/launchers/run_cvs_baseline_queue.sh \
> /home/szu2070436088/2510044040/CV-SincNet/paper_reproduction/logs/cvs_publication_phase1_leoaug_seed713101_20260713/nohup.out 2>&1 &
```

## 监控与成功条件

启动约4–5分钟后检查进程、GPU、scheduler和三个方法日志。启动健康条件：

- 三个worker均存活并进入epoch循环；
- launcher记录`sat_view_aug=1`；
- 每个方法命令包含`--use_sat_channel_view_aug`及三个训练场景；
- 无Traceback、RuntimeError、OOM、NaN/Inf；
- 每GPU不超过两个训练进程。

完成条件：

- 三种方法均完成200/200epoch；
- `best_by_val.pt`存在，checkpoint epoch等于全量metrics中最高source validation accuracy所在epoch；
- 正式详细后评估对每种方法输出612000条score与894条六层receiver/TX明细；
- 与No-Sat-Aug同seed结果进行同场景、同样本、同checkpoint选择规则比较；
- 主报告保留No-Sat-Aug行并新增LEO-Aug行，不覆盖历史证据。

## 风险

- `sat_view_prob=1.0`会把每个clean batch扩展为clean+satellite双视图，单步显存与计算量高于旧实验；若OOM，只允许先降低batch size并记录有效优化步差异，不得关闭增强。
- 三个训练场景随机采样会增加梯度方差；checkpoint仍只由source validation准确率选择，避免使用正式LEO test泄漏。
- 单seed只能完成受控配对重跑，论文显著性主张仍需后续补齐多seed。
+

## 2026-07-13启动与兼容修复记录

- 21:50:59 HKT直接N607预检通过；GPU0/1/2均为空闲状态，仅分配给本实验。
- 首次调度器PID：`4103472`；CVCNN-CE worker PID：`4103528`，GPU0。
- 首次launcher已确认三种方法命令均包含`--use_sat_channel_view_aug`、三种LEO训练场景、`sat_view_prob=1.0`、`epochs=200`和`seed=713101`。
- 首次启动中，RIEI-FD与DRIFT在进入epoch前因公共训练器接口同步不完整失败：`run_validation_gated_training() got an unexpected keyword argument 'test_eval_interval'`。该失败未生成checkpoint或metrics，不作为实验结果。
- 本地Git工作区完成兼容修复，同时保留`satellite_detailed_metrics.csv`导出，并明确保证间隔测试不会改写`best_by_val.pt`；正式checkpoint仍只由未取整source validation TX accuracy严格提升触发。
- 定向验证：`conda run -n ssr-gpu python -m pytest -q tests/test_baseline_training_behaviors.py tests/test_cvs_rffi_launcher.py`，结果23通过、1跳过。
- 修复提交：`1952d52 fix: preserve validation-gated baseline checkpoints`。
- 同步文件及远端SHA256：
  - `baselines/common/cvs_trainer.py`→`0a5f9a693301a3253a34c6bc0bce2e3ae9357c1b35a067cc2f60db96a9d0c875`
  - `baselines/riei_fd/train_cvs.py`→`fb9b5dba9d4e183e2ec82a4c5f3ac0e164b072870fd7a16536bd3a7434dd9342`
  - `baselines/drift/train_cvs.py`→`2ef3d37200be54f2c147ffcc8c9a6273902fd07792184d3a7f161a70cc48b231`
- 21:59:27 HKT仅重启失败的RIEI-FD与DRIFT；重试调度器PID：`4108427`；RIEI-FD worker PID：`4108476`（GPU1）；DRIFT worker PID：`4108483`（GPU2）。
- 22:00:07 HKT启动健康检查：三个worker均存活，CVCNN-CE已进入epoch6，RIEI-FD与DRIFT均进入epoch1；GPU0/1/2各一个训练进程，均未超过并发上限；最新日志未发现新的Traceback、OOM或RuntimeError。
- 启动日志：
  - `paper_reproduction/logs/cvs_publication_phase1_leoaug_seed713101_20260713/nohup.out`
  - `paper_reproduction/logs/cvs_publication_phase1_leoaug_seed713101_20260713/retry1_nohup.out`
  - 方法日志以`baseline_<method>_cvs_day_rx_seed713101_<timestamp>.log`命名。
- 每次SSH/SCP操作后均核验本地`ssh.exe=0`且到N607/桥接机的TCP22连接数为0。

