# LoRa 物理轨迹 DURA 与完整适配 Tri 的配对验证

固定同种子 2026092702 的原 R69 物理轨迹模型和两种完整 LoRa REPL，均在同一合法源合同冻结。两种 Tri 都保留，各自比较直通、R69 全量、R69 半量和门控切换。门控切换使用同一个仅由源 IQ 得到的 R69 效用门控，触发时把 R69 半量输出交给适配 Tri，其他样本沿用冻结原 Tri-set 的直通预测。切换只使用当前接收 IQ，没有目标真值、批量统计、场景名或接收机身份输入。

比较相同 30,000 包、15 台未见 RX、clean 和六种 practical。16 条固定读出，包括物理轨迹 DURA 的八路已冻结对照和两种 Tri 的各四路。复用 1,050 份盲预测，新增 630 份，共 1,680 个数组与 112 个评分单元。目标真值仅在所有预测冻结后由独立程序读取。报告每个方案的绝对 clean、六景、城区低、距 clean 及净纠错。门控切换单列相对原 Tri-set 直通、纯物理轨迹 R69 半量的逐包差值。clean 要求同时核对绝对分数及错改数。

N607 普通账户、GPU0–7 容量调度。发布命令 `python -m experiments.pamf_lora_main.full_dura_publish --native-tri-submit ... --r69-submit ... --adapted-joint`，传入两种完整 LoRa REPL 与轨迹恢复的记录。Git commit 定义实现，发布 CWD `CV-SincNet/releases/20261122-phase1-lora-adapted-dura-tri-joint-r01-release`，唯一运行根 `CV-SincNet/runs/20261122-phase1-lora-adapted-dura-tri-joint-r01`。源无查询 smoke 在两份上游盲预测冻结后执行。来源、门控身份、非有限或不完整预测才技术停止。
