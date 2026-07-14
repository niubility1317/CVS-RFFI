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
