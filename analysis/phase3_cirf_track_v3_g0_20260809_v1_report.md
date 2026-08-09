# Phase3 CIRF-Track v3 G0技术发布报告

状态：`PREREGISTERED_READY_FOR_RUNNER`

日期：2026-08-09
实验ID：`phase3_cirf_track_v3_g0_20260809_v1`
操作方：主代理集成；N607发布由唯一实验runner执行

## 1.目标与边界

本次仅验证CIRF-Track v3的CPU技术G0闭环，验证事件权限、调度重放防护、双阶段QP、MHT迟到事件零变异、truth／role隔离和不可覆盖工件。它不读取真实query真值，不训练模型，不产生性能结果，也不构成Phase3未知拒识、确权、注册或真实多星协同能力证明。

假设：冻结实现能够在单次CPU运行中生成自洽技术工件；跨进程重放、账本截尾回滚、回执／账本／store篡改均fail-closed；严格正曲率QP与精确奇异QP边界不混淆；超过N-scan的未见事件不改变轨迹状态。

比较目标：冻结设计commit `1f7444770a8837e5fae11f254562d47a145e80e4`及追溯表G0-01／G0-02／G0-03。无性能baseline。

## 2.本地实现与验证

Git工作树：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`
分支：`codex/phase3-responsibility-20260807`
实现及发布commit：`f76cee26f00a21968ed1849e3610b9fb15309fd0`

|文件|用途|SHA256|
|---|---|---|
|`code/cvsrffi/phase3_cirf_track_v3.py`|CIRF事件、融合、调度、风险与MHT技术实现|`1CFD07BC77AFE5A0BA5AC97A098A09B6A18491AE0FEF355A20947548F25AD0A0`|
|`code/scripts/phase3_cirf_track_v3_g0.py`|CPU-only不可覆盖G0入口|`829490838A2C1E157C5EDCC427289A4A2C3277E34C0FAAF023B14728114AA51B`|
|`code/tests/test_phase3_cirf_track_v3.py`|性质、攻击与CLI负测|`93F32E5376B9F74A29F6BCC71E9AEB45721502629F777CA2E2AB35E57930D0F2`|
|`analysis/phase3_final_goal_traceability_20260809.md`|最终目标差距与G0追溯|`3D6172D064AA2CA4E6C6C769B8B45D9F4BA764DB97127782068EA0F53238584C`|

本地验证：

- `ssr-gpu`下三文件`py_compile`通过。
- `test_phase3_care_poe.py＋test_phase3_cirf_track_v3.py`：49项通过，warning按error处理。
- 双进程同session：首进程`REQUEST`，第二进程`UNKNOWN_NETWORK_STATE`。
- 删除最新generation后，携原latest回执的第三进程被拒绝。
- existing session缺回执、换store、回执篡改、ledger篡改均被拒绝。
- QP边界：`1-2^-53`正定反例返回`[0.5,0.5]`；精确奇异情形保持`beta0`。
- MHT超过N-scan的未见事件返回audit-only且轨迹状态零变异。
- CLI首次写入退出0，第二次同目录写入退出非0。
- `git diff --check`通过，仅Windows换行提示。
- 独立冻结字节复核：`P0=0、P1=0、ALLOW`。

## 3.N607预注册

发布前置：Git提交完成、report更新为固定commit、direct N607只读preflight通过；若direct不可用且身份配置正常，仅使用已验证lab bridge。新release／run／log路径必须全部不存在。

预定路径：

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase3_cirf_track_v3_g0_20260809_v1_f76cee26`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase3_cirf_track_v3_g0_20260809_v1`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase3_cirf_track_v3_g0_20260809_v1.out`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`<release>/code`
- GPU：CPU-only，不分配GPU；仍记录8卡只读占用情况

固定启动命令：

```bash
nohup setsid env PYTHONPATH=/home/szu2070436088/2510044040/CV-SincNet/releases/phase3_cirf_track_v3_g0_20260809_v1_f76cee26/code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -X utf8 -W error /home/szu2070436088/2510044040/CV-SincNet/releases/phase3_cirf_track_v3_g0_20260809_v1_f76cee26/code/scripts/phase3_cirf_track_v3_g0.py --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/phase3_cirf_track_v3_g0_20260809_v1 > /home/szu2070436088/2510044040/CV-SincNet/logs/phase3_cirf_track_v3_g0_20260809_v1.out 2>&1 < /dev/null & echo $!
```

## 4.预期工件与技术门

预期小工件：

- `g0_technical_artifacts.json`
- `scheduler_replay_receipt.json`
- `g0_manifest.json`
- `g0_manifest_receipt.json`
- `scheduler_replay_store/<session>/generations/*.json`

成功条件：

1.唯一启动退出0，输出路径不可覆盖；
2.工件hash／schema／self-hash／外部manifest回执闭合；
3.`cpu_only=true`、`technical_synthetic=true`、`performance_result=false`、`truth_sidecar_opened=false`；
4.append-only ledger至少包含root与终态generation；
5.日志无Traceback、RuntimeError、warning或非有限值；
6.本地回收小工件后逐项核bytes和SHA。

系统性技术停止条件：错误checkout／hash、覆盖风险、truth／role泄漏、launcher-wide确定性异常、零工件或协议receipt不闭合。不得依据任何性能字段停止或选择。fresh retry：`NO`；若失败，仅保留证据并返回`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE／NO_PERFORMANCE_RESULT`。

## 5.结果表

|候选|类别|数据／角色|节点数|性能指标|技术状态|最终判定|
|---|---|---|---:|---|---|---|
|CIRF-Track v3 G0|CPU合成技术闭环|无真实query／无truth scorer|技术fixture|N/A|PENDING_N607|不得作性能或晋级结论|

## 6.已知风险与后续边界

- G0只证明实现性质和工件闭合，不证明真实unknown FAR、safe rejection、定位、追踪或确权性能。
- 当前Phase1仍缺一个未被拒绝、真实checkpoint绑定并输出`z_id、z_dom、q、d_class、e_unknown、p_local`的完整deployment bundle。
- old-class协同适应、授权fresh-K到Stage2-C、四状态`DA0_REG0／DA1_REG0／DA0_REG1／DA1_REG1`、真实`N_sat={1..5}`与A/B/C/D完整矩阵仍未完成。
- N607技术工件闭合后，目标保持active，不得把本次G0标为完整Phase3完成。
