# Phase1 FastTrust-QB3 C2多seed补齐与伪标签质量审计

## 最小预登记

- run ID：`phase1_adv3b02_fasttrust_qb3_c2_ms_e200_20260826_r1`
- 目标：在不改变已冻结QB3数学定义与现行`LEO_WEAK`日程的条件下，补齐seed713101、seed713102的C2 E200结果；同时建立`V_select-as-U`独立伪标签质量审计和真实共享参数梯度遥测。
- 候选矩阵：seed713101 C2、seed713102 C2；C2为H+P-set，P-conditional关闭。
- 固定协议：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，source-only，U_s训练期不读取TX真值，target/query不参与训练、校准、选模或调参。
- 训练预算：E200、U batch256、`eval_batch_size=512`、逐epoch恢复checkpoint。
- 终评：Clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`分别保存。
- 预期artifact：每行200条epoch记录、`final_ssdg.pth`、四场景独立指标与日志；`V_select-as-U`truth-blind artifact及独立truth-last评分结果。
- 技术停止：协议/query越权、错误seed/split/checkpoint、输出覆盖、错误checkout、同一确定性异常至少两行、prediction无法闭合或进程归属不清。低性能不停止。
- 当前状态：设计追踪与本地实现中。
- 精确启动命令：`bash /home/szu2070436088/2510044040/CV-SincNet/releases/<release>/code/scripts/launch_phase1_adv3b02_fasttrust_qb3_c2_ms_e200_20260826.sh`

## 需求追踪

| ID | 报告要求 | 实现面 | 验证状态 |
|---|---|---|---|
| `QB3-P0-QUALITY` | `V_select-as-U`生成truth-blind逐样本artifact，独立连接truth评分 | 待实现脚本与测试 | `IN_PROGRESS` |
| `QB3-P0-GRAD` | 对实际H/P损失与共享参数求导，报告范数比与余弦 | `code/SSDG/train_ssdg.py`及聚焦测试 | `IN_PROGRESS` |
| `QB3-P1-C2MS` | 仅新增seed713101、seed713102的C2 E200 | 新matrix与launcher | `IN_PROGRESS` |
| `QB3-SPEED` | 缓存冻结anchor clean logits并向量化路由预算 | 训练路径与速度A/B | `IN_PROGRESS` |
| `QB3-SINC` | `torch.sinc`+FP32滤波器合成，独立匹配验证 | `code/model.py`及数值测试 | `PENDING_SEPARATE_COMMIT` |
| `QB3-RG` | P-set/P-cond独立预算与rank风险门控 | source-only候选 | `PENDING_P0_EVIDENCE` |

## 口径冲突处理

附件将`mixed_orbit`描述为正式默认，但当前`项目.md`明确Phase1默认采用三类`LEO_WEAK`日程。本轮按当前`项目.md`执行，附件中的历史`mixed_orbit`主张不进入本次矩阵。

## 本地实现与验证

- `code/cvsrffi/phase1_pseudolabel_quality.py`：实现truth-blind逐样本记录和独立H/P质量评分，包含class、receiver、receiver/day、class×receiver分组。
- `code/scripts/phase1_fasttrust_vselect_quality.py`：`generate`只接收移除TX字段后的V_select-as-U数据；`extract-truth`和`score`为独立子命令。
- `code/SSDG/train_ssdg.py`：将H/P梯度遥测绑定到实际loss与共享参数，断图记录`NaN`而非0，并增加相对labeled梯度余弦。
- `configs/phase1_adv3b02_fasttrust_qb3_c2_ms_e200_20260826.json`：仅包含seed713101、seed713102的C2 E200。
- `code/scripts/launch_phase1_adv3b02_fasttrust_qb3_c2_ms_e200_20260826.sh`：专用绑定C2矩阵，拒绝误落入旧速度profile默认值。
- 独立P0/P1审查首轮发现同进程truth接触和launcher默认矩阵两项问题；定点修复后复审结论为`READY`，未增加白名单外gate。
- 本机Git Bash探针为`MSYSTEM=`，不满足`MINGW64`，本地`.sh`通道记为`FAILED`；远端发布后执行`bash -n`和behavioral dry-run。
- Windows原生`ssr-gpu`验证：聚焦测试`58 passed`；三个Python实现文件通过`py_compile`；`git diff --check`无空白错误。
