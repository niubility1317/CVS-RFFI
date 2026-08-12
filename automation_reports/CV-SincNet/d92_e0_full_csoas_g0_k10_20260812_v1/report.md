# D92 E0 FULL CSOAS K10 G0预注册

状态：`WAITING_FOR_SCIENTIFIC_COMMIT`

本文件是机械发布预注册，不是运行结果。当前不得封archive、SSH/SCP、launch或commit。

## 1.实验身份与目标

|字段|冻结值|
|---|---|
|experiment ID/run ID|`d92_e0_full_csoas_g0_k10_20260812_v1`|
|记录时间|2026-08-12（Asia/Hong_Kong）|
|operator/owner|Codex Luna/max；机械发布准备，非科学owner|
|候选|`E0_FULL_CSOAS`|
|目标|固定真实K10、单一outer、三场景truth-free G0机制与资源门|
|比较目标|同一outer的E0 FULL/D42封存基线；不读取性能分数|
|假设|CSOAS在`DA1_REG1`、`K>2`时可保持协议闭合并使D42发布state非E0，同时满足一次FULL fit与资源硬门|
|声明范围|`DEVELOPMENT_ONLY_SUPPORT_G0 / NO_PERFORMANCE_RESULT`|
|当前科学commit|`0000000000000000000000000000000000000000`（占位；待主代理填入）|
|当前Git状态|HEAD=`4704c543`；仅设计/计划已落地，CSOAS科学实现尚未落地|

## 2.冻结矩阵与数据边界

|字段|冻结值|
|---|---|
|outer|`rx_7_7__seed_713106__k_10__new_5`|
|receiver/TX split|沿用已验证sealed package；不改变receiver、TX、物理ID或support/query划分|
|K/new|`K=10`；`new=5`|
|scenario|`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`；三场景物理样本集合互斥|
|schema|`protocol_schema=p2_min_v1`；`phase2_data_status=VALIDATED_ONCE`句柄沿用原sealed job|
|query|仅逐样本预测；truth-free、query不fit、不update、不selection、不回退、不stop、不跨query重排|
|禁止读取|clean/source、query truth、query role、true batch class count、class quota、global assignment/scorer结果|

### 2.1已核对sealed输入

以下路径和hash复用已完成TCRA safe-v2 G0的sealed job/ground证据；本次只登记，不重新验证、不修改、不复制远端内容。

|输入|固定值|
|---|---|
|sealed job root（N607）|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5`|
|ground root（N607）|`/home/szu2070436088/2510044040/CV-SincNet/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component`|
|ground manifest SHA256|`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`|
|before enrollment seal SHA256|`e3da38668a1e6ec4053e65e669e2a1845bb43198891644d830950e4550b5cea9`|
|before apply seal SHA256|`736852188c32255647b8105bc7a68d4cc92ca73615e4734d0ed5f4bdd0f04473`|
|after enrollment seal SHA256|`2600a21ee9a2f95a8d17fa1f4d2263b0e04d243424e3257474502953ed6d9286`|
|after apply seal SHA256|`afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a`|

## 3.G0唯一判定门

G0只判定下列机制、协议和资源字段；不读取`score`、`accuracy`、`H`、`BA`、`floor`、`forgetting`、混淆矩阵或任何独立scorer输出。

|门|要求|失败动作|
|---|---|---|
|active|三场景均`active=true`|
|fallback|三场景均`fallback=false`且无fallback reason|
|FULL lifecycle|`DA1_REG1`实际FULL fit恰为1；无额外BLOCK/LOO/Fisher/扫描|
|D42 non-E0|D42部署state SHA与E0 state SHA不同；support跨组margin变化达到真实codec量子|
|query boundary|全部query禁用字段为`false`：fit、update、selection、truth、role oracle、class quota、global reassignment及扩展字段|
|wall|三场景registration wall P90（nearest-rank）`≤150ms`；目标`≤120ms`|
|peak|三场景peak memory均`≤E0 baseline+512KiB`|
|artifact|fit/resource/query/receipt闭包完整、hash与run root匹配|

任何门失败均为`NO_PERFORMANCE_RESULT`。不得根据中间性能值停止或选择场景。

## 4.不可覆盖路径与唯一命令

### 4.1 immutable roots（待科学commit填实）

|用途|N607路径|
|---|---|
|source root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_g0_source_0000000000000000000000000000000000000000_20260812_v1`|
|output root|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_csoas_g0_k10_20260812_v1`|
|log root|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_csoas_g0_k10_20260812_v1`|
|archive|`<source_root>/d92_csoas_g0_runtime_0000000000000000000000000000000000000000.tar.gz`（未封存）|

### 4.2唯一detached command（当前禁止执行）

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_g0_source_0000000000000000000000000000000000000000_20260812_v1 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

该命令只有在主代理填入真实科学commit、封存runtime archive、完成本地验证并明确交给sole runner后才可执行。当前无PID、无GPU分配、无远端落地状态。

### 4.3环境与GPU占位

|字段|预注册值|
|---|---|
|Python|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`|
|CWD|上述`source_root`（解包后`<source_root>/code`）|
|GPU|`CUDA_VISIBLE_DEVICES=0`；仅一张卡|
|CPU threads|`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`|
|PID|`PENDING_NOT_LAUNCHED`|

## 5.期望artifact与停止规则

期望artifact（实际文件名须由科学入口与主代理最终填实）：

- `source/launch.sh`、`source/d92_csoas_g0_runtime_<scientific_commit>.tar.gz`及其SHA256；
- `output/after/fit_audit.json`、`output/after/execution_receipt.json`、`output/after/resource_audit.json`、`output/after/prediction_artifact.npz`；
- `logs/import_closure.out`、`logs/import_closure.err`、`logs/prediction.out`、`logs/prediction.err`、`logs/g0_validation.json`；
- `g0_validation.json`必须记录三场景active/fallback、实际FULL fit、D42 non-E0/codec quantum、query禁用字段、wall P90、peak delta、`performance_claim=false`和`truth_or_scorer_used=false`。

预注册停止规则：wrong checkout/hash或source/output/log已存在、任一P0协议/权限违反、query禁用字段为true、active=false、fallback=true、FULL fit≠1、D42 state byte-exact E0、codec量子门失败、wall/peak硬门失败、缺失预测闭包、launcher异常或两项确定性零预测异常。停止只保留partial logs/receipts，不删除或覆盖任何artifact。

`fresh_run_retry=false`。同一run ID不得重试、恢复、覆盖或改命令；修复必须新建不可覆盖run ID并由主代理重新预注册。

## 6.本地→远端映射

|本地路径|N607远端路径|状态|
|---|---|---|
|`E:\type10-7\code\snapshots\d92_125wt\automation_reports\CV-SincNet\d92_e0_full_csoas_g0_k10_20260812_v1\report.md`|`/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/d92_e0_full_csoas_g0_k10_20260812_v1/report.md`|仅预注册，未SCP|
|`...\launch.sh`|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_g0_source_<scientific_commit>_20260812_v1/launch.sh`|仅预注册，未SCP|
|`...\DELIVERY_MANIFEST.txt`|`/home/szu2070436088/2510044040/CV-SincNet/automation_reports/CV-SincNet/d92_e0_full_csoas_g0_k10_20260812_v1/DELIVERY_MANIFEST.txt`|仅预注册，未SCP|
|`E:\type10-7\automation_reports\CV-SincNet\d92_e0_full_csoas_g0_k10_20260812_v1\{report.md,launch.sh,DELIVERY_MANIFEST.txt}`|外部同路径镜像|本地已生成，未远端同步|

## 7.当前验证与交接字段

当前只验证发布骨架，不运行项目代码测试：

- `bash -n launch.sh`：PASS（snapshot repo）；
- snapshot Git工作树`git diff --check`：PASS；
- 两处镜像逐文件SHA256一致：PASS。当前`launch.sh`骨架SHA256为`7310091d200c9f8d3785c01417ac640bd6d176338b78b7c6da8f1f3918593182`；`report.md`与`DELIVERY_MANIFEST.txt`的最终SHA256记录在manifest并在镜像复核中一致；
- 未执行：Conda测试、N607 SSH preflight、SCP、远端hash/compile、detached launch、scorer。

主代理科学提交后必须填入：真实scientific commit、runtime archive路径/大小/SHA256、launch SHA256、CSOAS入口文件清单及局部验证结果；随后由主代理统一封存并决定是否交给sole runner。本机械owner不修改科学代码、配置、测试或公式。

## 8.风险与边界

- 当前HEAD只有设计冻结commit，不能把设计、占位命令或sealed输入复用误写成CSOAS实现证据。
- TCRA safe-v2的archive/hash仅作为sealed job/ground路径参照；不得把TCRA runtime archive冒充CSOAS runtime archive。
- `peak≤E0+512KiB`需要真实运行时资源收据；本预注册不提供该结果。
- 本报告及镜像不构成性能、投稿或Phase3完成声明。
