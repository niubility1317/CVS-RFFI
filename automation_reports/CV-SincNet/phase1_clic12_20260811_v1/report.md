# Phase1 CLIC 12臂训练预注册与运行报告

## 1. 基本信息

- 实验ID：`phase1_clic12_20260811_v1`
- 预注册时间：2026-08-12（Asia/Hong_Kong）
- 操作方：Codex主控；N607唯一runner待交接
- 当前状态：`LOCAL_VERIFIED / READY_TO_LAUNCH / NOT_LAUNCHED`
- 目标：在冻结的六折source-only Phase1矩阵上训练CLIC对照臂C与机制臂G，为后续clean、三种LEO weak、source-proxy unknown和目标域盲态评测生成12份真实final checkpoint与严格训练回执。
- 假设：固定lag集合`{1,2,4,8}`的多尺度三点复曲率token，在不改变训练数据、主干、分类头、损失、优化器、epoch和物理批顺序的前提下，可能改善LEO weak下的身份域泛化和source-only未知能量几何。
- 对照：C=`raw_phase_control`；G=`complex_local_invariant_curvature`。两臂仅token算子不同。

## 2. 科学与数据边界

- Phase1训练只读取WiSig/ManySig source域；不读取target、query、target role/truth或正式unknown。
- 训练角色固定为source全池`0.07/0.63/0.30`：有标签训练、无标签训练、互斥source validation。
- 训练每个source物理样本只使用clean与一个按预注册循环取得的LEO weak视图；三场景为`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。
- `L_base=clean CE+0.10×KL(clean-stopgrad→single-LEO)`；LEO CE为0；不增加CLIC专属loss。
- 未知类拒识、目标域盲态确认和ADV3B02比较属于checkpoint冻结后的独立阶段；训练健康监控不得读取或据此选择性能。
- WiSig/ManySig是地面代理数据，LEO weak是物理启发压力代理，不构成真实在轨验证。

## 3. Git与冻结文件

- Git工作区：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`
- 训练发布commit：`5e34483cb4324f1c8dac766a077e12e78537fe61`
- commit说明：`Implement CLIC postfreeze evidence chain`
- 发布方式：只允许从上述commit生成无工作树污染的Git archive；当前未完成的Task7目标评分文件不得进入训练release。

| 文件 | SHA256 |
|---|---|
| `code/scripts/launch_phase1_clic12_20260811.sh` | `40DC21E33F160254C068E3220F452F751D2CC2E40C77F7D64039C874414753D6` |
| `code/SSDG/train_ssdg.py` | `8E3812B3DA4110063BA43A442B0E80EBE3FD5B9CE0C436E0AA8FF7DDD30CBEF8` |
| `code/model.py` | `171F7A483CDB94D6C974442C12ABA2B9C48A552C8D146BCA11F8F8805FE33B8D` |
| `code/cvsrffi/phase1_clic.py` | `7BCE1F4FA6214E1EC945B9100E8E2E9AE1ED2D2198DF12C72E33339E1D065605` |

Task5独立终审已经给出`P0=0/P1=0`；训练实现不依赖仍在开发的Task7目标评分模块。

## 4. 本地验证

环境：`ssr-gpu`。

- `py_compile`：CLIC核心、模型、训练器通过。
- `pytest code/tests/test_phase1_clic.py -q`：`164 passed`；仅既有AMP弃用warning。
- `train_ssdg.py --help`：通过。
- launcher `bash -n`：通过。
- launcher dry-run：12行，C=6、G=6、`lambda_sat_cons=0.10`为12/12、40epoch为12/12。
- 训练发布文件与commit`5e34483c`逐文件无差异。

## 5. 冻结矩阵与资源

共同配置：seed=`7281164`，40epoch，batch=128，AdamW，lr=`2e-4`，weight decay=`1e-4`，AMP=true，final-only checkpoint，local4分类，旧机制全部关闭。

| fold | source train TX | held source-known TX | C/G |
|---|---|---|---|
| F1 | `20-15,20-19,6-15,8-20` | `14-7` | C+G |
| F2 | `14-10,20-19,6-15,8-20` | `20-15` | C+G |
| F3 | `14-10,14-7,6-15,8-20` | `20-19` | C+G |
| F4 | `14-10,14-7,20-15,8-20` | `6-15` | C+G |
| F5 | `14-10,14-7,20-15,20-19` | `8-20` | C+G |
| F6 | `14-7,20-15,20-19,6-15` | `14-10` | C+G |

GPU映射固定为：GPU0=`F1C,F5G`；GPU1=`F1G,F5C`；GPU2=`F2C,F6G`；GPU3=`F2G,F6C`；GPU4=`F3C`；GPU5=`F3G`；GPU6=`F4C`；GPU7=`F4G`。不超过每GPU两个训练进程。

## 6. N607发布与唯一启动合同

- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`
- 预期release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic12_20260811_v1_5e34483c`
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic12_20260811_v1`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_clic12_20260811_v1`
- outer log：`/home/szu2070436088/2510044040/CV-SincNet/phase1_clic12_20260811_v1_outer.out`
- Conda/Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- warm-start根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_loto_clsgeo12_20260808_v1`

唯一启动命令冻结为：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic12_20260811_v1_5e34483c/code
nohup env RUN_ID=phase1_clic12_20260811_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic12_20260811_v1_5e34483c/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python bash scripts/launch_phase1_clic12_20260811.sh >/home/szu2070436088/2510044040/CV-SincNet/phase1_clic12_20260811_v1_outer.out 2>&1 &
```

必须在启动前确认release/run/log/outer均不存在或release为本次原子落地目标，8张GPU资源符合每GPU最多两个训练进程，12个warm-start checkpoint存在。只允许一次launch调用；fresh-run retry=`NO`，若失败必须回主控修复并使用新run ID。

## 7. 启动健康与停止规则

启动后核对outer PID、12个candidate PID、CWD/cmdline/run-root绑定、GPU映射、`pids.tsv`和日志增长。首波查看配置、epoch开始、AMP/VJP/CLIC技术回执字段以及Traceback、OOM、NaN、参数拒绝或输出覆盖错误。

只在以下情况下停止本run拥有的进程树：P0协议/安全问题；wrong checkout/hash/CWD；覆盖风险；launcher-wide确定性故障；或至少两个不同candidate在产生有效checkpoint前出现同一规范化确定性异常指纹。不得因中间accuracy、floor、AUROC、u-gap或任何性能值停止、重启或调整。

## 8. 预期工件与后续指标

每个candidate至少产生：`final_ssdg.pth`、`phase1_clic_terminal_receipt.json`、完整训练日志及技术统计；log根产生`pids.tsv`。训练全部完成后才运行source clean、三种LEO weak、fixed400 TX互斥source-proxy、PAIR、G deployment bundle及目标域盲态评分。

后续每个实验指标必须同时报告叠加LEO weak的目标域结果，核心目标保持为未知类拒识与域泛化。正式目标unknown仅把明确`decision=unknown`计入拒识分子，`defer`单列；registered被unknown/defer均按身份错误。性能不达门时保存`passed=false`证据，不把它伪装成技术失败。

## 9. 运行回填

- launch时间：待runner回填
- release archive SHA/bytes：待runner回填
- outer PID：待runner回填
- candidate PID/GPU：待runner回填
- 首波健康：待runner回填
- 最终状态：待runner回填
