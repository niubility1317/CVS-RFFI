# RIEI Table III论文一致性修复与优化报告

- 实验ID：`paper_repro_riei_parity_repair_20260714_145800`
- 目标：修复RIEI期刊Table III的预处理、优化器和评估窗口偏差，先做Table III第1行8候选受控消融，再以固定配置确认完整12行。
- 论文目标：12行均值`73.30%`；第1行`77.88±2.23%`。

## 已确认问题

| 问题 | 修复前 | 当前修复 |
|---|---|---|
| 信号预处理 | `riei_original`硬编码逐包RMS归一化 | 严格候选仅信道均衡并关闭RMS；保留RMS control |
| 优化器 | 固定Adam，FED同一批次连续两个Adam step | 增加无momentum SGD及Adam消融，保持Eq.20–21交替顺序 |
| 评价窗口 | Table III使用last10 | 2025期刊版默认last5；旧会议版last10只作历史口径 |

当前fixopt前8行均值`60.34%`，论文前8行均值`73.37%`，平均差`-13.03pp`；source validation接近`100%`而target receiver偏低，主因是跨接收机泛化，不是NaN/OOM。

## 本地验证

- 修改：`baselines/common/cvs_data.py`、`baselines/riei_fd/train_cvs.py`、`run_wisig_paper_scope_queue.sh`、`code/scripts/launch_riei_parity_repair_matrix_20260714.sh`、`tests/test_riei_parity_repair.py`。
- 根目录聚焦测试`15 passed`；Git镜像聚焦测试`3 passed`。
- `py_compile`、`bash -n`和8-job dry-run通过。
- 发现矩阵固定第1行、seed1337、200epoch、last5；目标域间隔曲线只作诊断，禁止target-oracle选epoch。
- 当前N607 fixopt仍有4个RIEI训练；未同步、未启动、未影响Phase1。后续先完成DRIFT v2，再在容量门通过后启动本RIEI矩阵。

## 2026-07-14 17:20启动前门控

- DRIFT五seed确认已5/5完成并退出；正式final均值`72.75±5.93%`，与论文73.54%差-0.79pp。Git提交`be02078`已将mean/no-cap配置固定为唯一支持的DRIFT论文复现入口。
- 本地重新验证：`ssr-gpu`下`py_compile`通过；RIEI launcher与canonical DRIFT launcher的`bash -n`通过；RIEI 8-job dry-run完整展开P01–P08；聚焦测试根目录15 passed、Git镜像3 passed。
- 待同步文件SHA256：`cvs_data.py=a2093e0a...`、`riei_fd/train_cvs.py=950b6008...`、paper queue=`2ba90874...`、RIEI launcher=`e2a87932...`。
- 直接N607预检通过；实时GPU compute仅GPU3有1个既有Phase1进程，GPU0–2、4–7为空。RIEI矩阵每GPU新增1个，峰值GPU3=2、其余=1，满足每GPU不超过2。
- 计划正式命令：`bash code/scripts/launch_riei_parity_repair_matrix_20260714.sh --launch --gpu-ids 0,1,2,3,4,5,6,7 --max-train-per-gpu 2`。独立run/log根为`paper_repro_riei_parity_repair_20260714_145800`，不会覆盖既有产物。
