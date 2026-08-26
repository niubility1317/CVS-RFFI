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

## 真实checkpoint smoke修复

- 首次远端`V_select-as-U generate`在进入模型前触发`KeyError:0`。根因是无标签dataset edge已返回独立domain张量，生成器却再次把去除TX字段后的metadata传给`domain_from_extra`。
- 失败进程没有生成artifact，失败输出目录不复用；新增无标签batch解包回归测试后，生成器直接使用dataset edge返回的domain张量。
- 修复后的聚焦测试为`4 passed`，脚本通过`py_compile`；等待新Git提交与新release执行真实checkpoint复验。
- 新release复验继续推进后在receiver读取处触发`KeyError:1`：truth-hidden batch已被解包为metadata字典，但生成器仍调用只接受训练期原始batch包装的解析器。该次同样发生在推理前且没有生成artifact。
- 新增metadata字典直读回归测试，receiver改为从已解包metadata构造只含观测域信息的张量；相邻聚焦回归为`57 passed`，待再次提交并用全新release/output root复验。
- 第三次复验进入RC4路由后出现CUDA gather越界。定位为dataset edge返回的raw domain ID未按checkpoint的`domain_label_map`映射为紧凑域索引；训练主路径本来执行该映射，问题仅在新增审计脚本。
- 审计脚本现复用训练主路径的`domain_from_extra`映射并对未注册域fail-closed；增加`{3:0,4:1}`紧凑映射回归，聚焦回归仍为`57 passed`，后续使用全新release/output root验证。
