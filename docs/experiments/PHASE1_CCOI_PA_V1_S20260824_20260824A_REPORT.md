# PHASE1_CCOI_PA_V1_S20260824_20260824A实验报告

## 预登记

- 状态：`ANALYZED`；C0–C4 prediction已闭合并由独立scorer连接truth完成分析。
- 候选：`CCOI-PA-V1`，单seed最小矩阵`C0/C1/C2/C3/C4`。
- 科学对照：冻结`ADV3B02_CORE90_SOFT_E200`；C1–C4同容量、同初始化、同split、同seed、同训练和评估预算。
- Git实现提交：`f7f2ab4a8431091d1674439fb99f2e414010ae6e`，分支`codex/phase1-ccoi-pa-v1-20260824`，远端OID已独立核对一致。
- 本地变更：`code/model.py`仅增加无参数`pa_token_map`；新增`ccoi_pa.py`、`ccoi_losses.py`、runner、独立scorer、launcher、配置、设计/追踪报告及聚焦测试。
- 本地验证：`ssr-gpu`环境；5个Python文件`py_compile`通过；55项聚焦测试通过；一次P0/P1只读复审完成并定点修复C1真实q旁路、缺少卫星激励、holdout感受野循环和truth写出顺序。
- 本机Git Bash：`FAILED`，指定Git for Windows后仍被路由为`/bin/bash`且`MSYSTEM`为空；未执行launcher。远端发布前必须运行`bash -n`。
- 源域协议：`L_s/U_s/V_cal/V_select=0.07/0.63/0.15/0.15`，`rho_label≤0.1`；目标域、query、query role和query truth不进入训练/校准/选择。
- 场景：clean、`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`分别输出。
- 命令：`RUN_ID=PHASE1_CCOI_PA_V1_S20260824_20260824A GPU=<preflight-selected> bash code/scripts/launch_phase1_ccoi_pa_v1_20260824.sh`。
- 环境/CWD：N607普通账户；`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；`/home/szu2070436088/2510044040/CV-SincNet`。
- 输入checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- 输入数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`。
- 输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_ccoi_pa_v1_20260824/PHASE1_CCOI_PA_V1_S20260824_20260824A`；smoke使用独立不可覆盖后缀根。
- 日志：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_ccoi_pa_v1_20260824/PHASE1_CCOI_PA_V1_S20260824_20260824A.out`。
- GPU：5。预检时GPU0–4各有1个既有训练进程，GPU5–7无训练进程；本run只在GPU5启动1个训练进程，不触碰既有任务。
- 技术停止规则：仅在协议/数据越权、错误checkout/seed/split、输出已存在、真实checkpoint严格加载失败、非有限loss、prediction闭合失败、独立scorer连接失败或同一确定性预prediction异常重复时停止；不得因中间性能低而停止。
- 预期artifact：`protocol_and_smoke.json`、挑战预训练历史、每row sidecar、`prediction.jsonl`、后置`truth.jsonl`、`metrics.json`、challenge audit、matrix manifest和完整日志。
- N607预检：直连普通账户、项目根、checkpoint、ManySig数据和Python环境均`VERIFIED`；目标run与smoke根均不存在。
- release归档：`E:\type10-7\release_archives\phase1_ccoi_pa_v1_3ed07d9b.tar.gz`→`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v1_3ed07d9b.tar.gz`；本地/远端SHA256均为`15b33bc9ce88ca8ab0bd2bff8bfe23e9bf7a33f570bdea60d9e6a2e977fb13fd`。
- release目录：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_ccoi_pa_v1_3ed07d9b`；远端5个Python文件编译通过且`.pyc`已读回，launcher远端`bash -n`通过。
- 启动后绑定：launcher PID`2315866`，CWD为release目录；smoke子进程PID`2315868`使用GPU5，首次读回显存约654MiB；CWD、cmdline、run-root、GPU映射和日志增长均已独立核对。
- smoke闭合：supervisor日志已依次出现`[CCOI-PREDICTIONS] COMPLETE`、`[CCOI-SCORE] ANALYZED`、`[CCOI-SMOKE] PASS`和`[CCOI-LAUNCH] FULL MATRIX`；smoke的`prediction.jsonl`先闭合，随后独立scorer连接`truth.jsonl`，未把truth送入训练、校准、选择或预测过程。
- smoke严格协议读回：`protocol=Phase1_source_only`；`L_s/U_s/V_cal/V_select=5880/52920/12600/12600`，比例为`0.07/0.63/0.15/0.15`，`rho_label=0.1`；源/目标接收机交集为0；checkpoint严格加载为true，`missing/unexpected/shape_mismatch=0/0/0`，共195个state tensor；`input_len=256`，PA张量形状`[64,64,64]`，分类logits形状`[64,6]`且有限；`target_or_query_access=false`。
- smoke产物：`..._REAL_CKPT_NO_QUERY_SMOKE/C2/protocol_and_smoke.json`与`metrics.json`已存在并完成独立读回。该结果只证明实现、协议、真实checkpoint推理和评分链闭合，不构成C0–C4科学增益证据。
- 正式矩阵：launcher随后启动C0/C1/C2/C3/C4，子进程PID`2317388`，同一seed、split、训练/评估预算，`q_epochs=10`、`head_epochs=20`，使用GPU5；矩阵已完整结束，结果见下文。
- 连接状态：运行期间曾出现直连N607和实验室桥接SSH横幅超时，当时按`UNKNOWN`处理且未重启、重发或干预进程；本次直连预检已恢复并完成最终只读核验。

## 结果

### 完成状态

- 服务器完成时间：2026-08-24 19:29:55+08:00。
- launcher、训练子进程均已退出；2026-08-24 23:25只读预检显示8张GPU无计算进程。
- supervisor完整日志依次记录正式矩阵的CCOI-PREDICTIONS COMPLETE、CCOI-SCORE ANALYZED和CCOI-LAUNCH ANALYZED，没有Traceback、OOM、Killed、非预期NaN/Inf或确定性异常指纹。
- C0–C4每row均有train_history.json、prediction.jsonl、truth.jsonl、metrics.json、calibration.json和challenge_audit.json；C1–C4还保留sidecar checkpoint。最高状态为ANALYZED。

### 同row结果

| row | clean | clear | low-elev | rain | LEO均值 | 相对C0的LEO均值变化 |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 90.140 | 78.456 | 75.569 | 75.189 | 76.405 | 0.000 |
| C1 | 90.148 | 78.478 | 75.584 | 75.189 | 76.417 | +0.012 |
| C2 | 90.150 | 78.470 | 75.574 | 75.187 | 76.410 | +0.006 |
| C3 | 90.152 | 78.469 | 75.583 | 75.198 | 76.416 | +0.012 |
| C4 | 90.143 | 78.472 | 75.583 | 75.183 | 76.413 | +0.008 |

C2/C3/C4相对同容量C1的LEO均值变化分别约为-0.0069、-0.0008和-0.0046个百分点，均未达到预登记的+0.30个百分点晋级阈值。clean没有明显下降，但也没有形成可归因的星地场景收益。因此本轮结论是SCIENTIFIC_FAILURE_NO_PROMOTION，不是技术失败。

### 数据与数值健康

- 每个row的prediction与truth均为1,632,000条；独立scorer完整解析全部记录，验证sample ID唯一、两流ID集合完全相等、prediction不含true class、truth不含prediction，并在prediction闭合后连接truth。
- 1,632,000是四场景和全部命名诊断loader的记录数，其中按天、按接收机loader会重复引用主测试物理样本；正式主聚合每场景为204,000条，对应同一204,000个物理测试样本的四场景评估。该重复是诊断切片，不是独立物理样本扩增，也未进入训练。
- source角色为5,880/52,920/12,600/12,600，比例0.07/0.63/0.15/0.15，rho_label=0.1；源/目标接收机交集为0，target/query访问为false。
- q预训练完成10个epoch、每epoch 917 step；C1–C4均完成20个epoch、每epoch 91 step。所有下载并完整解析的训练历史、metrics和calibration均无非预期NaN/Inf。
- challenge audit中的d3为NaN且count=0，这是没有M1语义匹配样本时的显式N/A，不是数值崩溃；不得据此声明语义级挑战匹配。
- C4留出预测NMSE为real 0.1379、shuffle 0.1395、random 0.1789、constant 0.1692，说明holdout头确实学到可预测结构，但real相对shuffle只小幅改善，尚不足以证明强挑战条件辨识。
- 挑战码本48个code中33个被使用、15个未使用；最大单code约占49.5%。没有完全塌缩，但存在明显集中，需要作为后续机制诊断，而不是数据损坏。

### 接收机元数据异常

prediction流中的receiver字段统一为-1。因此当前metrics.json内自动计算的receiver_floor实际等于场景总体准确率，不能作为真实逐接收机下界使用；这不影响true class连接、总体/场景/loader准确率，但使正式receiver-cell floor证据无效。代码映射已定位：WiSig batch的extra是`(domain_tensor,meta_mapping)`，而`_meta_value`只在extra自身为Mapping时读取`rx_i`，因此没有进入`extra[1]`；随后回退到目标域经过source domain map后的-1。

利用已经存在的test_rx_7...11和test_unseen_day_rx_7...11 loader计数，可只读重建目标接收机7–11的诊断下界：

| row | clean floor | clear floor | low-elev floor | rain floor |
|---|---:|---:|---:|---:|
| C0 | 81.038 | 58.625 | 55.804 | 56.067 |
| C1 | 81.288 | 58.708 | 55.792 | 56.092 |
| C2 | 81.221 | 58.608 | 55.750 | 56.088 |
| C3 | 81.238 | 58.617 | 55.783 | 56.092 |
| C4 | 81.250 | 58.683 | 55.750 | 56.079 |

这些诊断同样没有显示超过0.30个百分点的稳定下界增益，但不能替代修正后的正式receiver-cell scorer输出。下一候选之前应单独修复接收机元数据导出/评分映射；本run保持不可变，不覆盖、不重跑。

### 最终判定

- 执行健康：VERIFIED。
- prediction/truth及主场景数据闭合：VERIFIED。
- receiver元数据与自动receiver floor：FAILED。
- CCOI-PA-V1科学晋级：FAILED/NO_PROMOTION。
- 总体交付状态：run已ANALYZED，但接收机下界子指标不可用；保留全部产物，修正评分元数据后只做现有prediction的重新评分，不因该问题重复训练。
