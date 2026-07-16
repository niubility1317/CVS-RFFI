# qKNN会话污染清理报告

## 基本信息

|字段|内容|
|---|---|
|清理时间|2026-07-16|
|目标会话|`019f6610-86af-7572-b857-2544e7b598ba`|
|目标|删除该会话及其对当前qKNN工作树、实验报告和后续探索造成的影响|
|Git承载面|`E:\type10-7\github_publish\CVS-RFFI-repo`|
|保留边界|保留独立主目标链中的LOPO source结果和K=1轻量适配证据|

## 谱系判定

原始会话JSONL中的Git提交命令与仓库对象逐项匹配，确认该会话直接创建52个提交，首个为`654d2f7`，末个为`6cc889e`。这些提交不是连续区间：同时运行的主目标任务在其间产生了独立提交，因此禁止按时间范围或整段历史回退。

清理采用重建树作为唯一内容基准：从`09408be`开始，仅重放下列5个独立提交，随后将当前索引树精确变换为该重建树。

|保留原提交|重建提交|保留原因|
|---|---|---|
|`0500446`|`90faa87`|修复LOPO source评估的NumPy2桥接|
|`bc2da9d`|`8438683`|记录`JG_R8_LR020`适应后88.8354%的source-only结果|
|`976f6ae`|`47f2061`|跨K值BPJG适配测试|
|`48993bd`|`d507fa1`|缩小K=1适配更新范围|
|`afd7bf3`|`74c4088`|记录K=1层适配结果|

## 删除范围

- 删除目标会话直接创建的52个提交的净内容影响。
- 删除依赖该严格运行时链继续产生的strict smoke/matrix、EvidenceNorm Round1和JP-R4 Round2代码、报告与本地产物。
- 删除`qknn_ground_effective8_strict300_20260715`报告、严格计划、运行时证据和矩阵汇总。
- 保留既有v14 source训练、LOPO v21结果、K=1 BPJG结果及其他会话的产物。
- 未触碰`mitigating_da_rootcause_20260710_104628/progress.md`和`task_plan.md`中的用户未提交修改。

Git索引净变化为1815个文件、61行新增、398839行删除。索引与重建分支`codex/cleanup-019f6610`逐文件一致。

Git删除后再次检查被`.gitignore`隐藏的物理目录，发现`qknn_ground_effective8_strict300_20260715`仍残留11个strict plan子目录、3608个未跟踪文件、14210174字节；已按精确目录整体删除，复核`Test-Path=false`。

## 保留的88%结果及声明边界

|候选|适应后准确率|最低类准确率|相对P4 identity|资源|结论边界|
|---|---:|---:|---:|---|---|
|`JG_R8_LR020`|88.8354%|75.9140%|+0.5133pp/+2.1505pp|6400参数、5epoch/50step、57084B状态|保留；source receiver、K=10、6个source类的old-class适配诊断，不是target Stage2-C新类注册结果|

## 验证

|检查|结果|
|---|---|
|索引树对重建树|`MATCH`|
|`git diff --cached --check`|PASS|
|保留链定向测试|`57 passed,4 skipped`|
|无关未提交修改|保持未暂存、未覆盖|

## Codex会话记录清理

|检查面|结果|
|---|---|
|`E:\codex\home\state_5.sqlite/threads`|目标ID记录为0|
|`E:\codex\home\sqlite\codex-dev.db/local_thread_catalog`|目标ID记录为0|
|目标rollout JSONL|不存在|
|目标可视化目录|不存在|
|项目会话索引|重建978条记录，目标ID无命中|

当前运行时没有任务删除/归档API；未直接操纵Codex界面。删除通过精确数据库事务和单路径文件删除完成，未修改其他任务记录。

## N607清理结果

2026-07-16 09:37 CST直连preflight通过；清理前后`gpu_compute=[]`、`active_training_processes=[]`，没有strict runner。远端根固定为`/home/szu2070436088/2510044040/CV-SincNet`。

|项目|结果|
|---|---|
|strict运行根|删除13个：无suffix及`v2`至`v13`|
|v14内runtime artifact|删除14个：`runtime_artifacts_strict_v1`至`v14`|
|目标专属共享文件|11个实际存在文件已删除；全名单复查无残留|
|被覆盖的共享文件|恢复21个当前保留版本文件|
|恢复包SHA256|`8527fb465444a6ac12148b492f8ed5ee29e102bcc8cfd9ae704ed94d8514377b`|
|本地/远端逐文件SHA|`REMOTE_SHA_MATCH=21`|
|残留连接|`ssh.exe=0`、N607/桥接TCP22连接为0|

删除范围约4.5GiB strict运行产物和约126MiB runtime artifact。未删除数据集、ADV3B02 checkpoint、v14 source训练、LOPO v21结果或K=1 BPJG证据。

## 最终边界

清理后当前可引用的88%版本仍为source-only`JG_R8_LR020=88.8354%`；被删除的strict300、EvidenceNorm和JP-R4结果不得再作为当前版本、探索轮次或性能证据引用。后续优化应从保留的LOPO/K=1谱系重新开始，同时补齐真实target新类注册指标。
