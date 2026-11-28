# LoRa 与 WiSig 共用 2% 有界教师逐频恢复筛查

- run_id：`20261128-phase1-shared-bounded2-teacher-s2026112957-r01`
- group_id：`phase1-crossdata-shared-dura-realgain`；类别：`comparison`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前状态：SOURCE_SCREEN_SCORED；两侧均未通过预登记晋级条件。

## 目的与对照

共同逐频实增益网络保持源选择锚点与训练配方，逐包修正上限从 1% 改为 2%，从头独立训练并在同源筛查。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
代码 `b192fc9cd314958ca813317c2a30be5d20755794`；控制器 3853613，LoRa worker 3853829 / GPU 7，WiSig worker 3853830 / GPU 2。真实专家、教师及 KL 到网络的有限非零梯度首步通过。 N607 预检、归档 SHA、远端编译、PID/CWD/GPU 和日志增长已核实；独立 P0/P1 审查通过，其他 owner 进程保留。

## 结果与覆盖

固定源 V 两景预测已独立评分；正式测试尚未进行。用户授权的源 U 冻结教师 KL 加波形和 clean 锚保持训练；LoRa W24 E104、WiSig Tri-SIG-TIME SSL F0 分别作为本数据集教师。LoRa 教师采用原 scipy 前向及 WideTracker 替代梯度。LoRa/WiSig 预定 8000/48000 个教师样本前向，另计反向与教师／专家祖先成本。 两侧共用网络与训练/推理一致的 2% RMS 上限，独立权重的 skip 分别为 LoRa 实增益和 WiSig R17。锚点以及从 1% 扩至 2% 均参考已多轮曝光的同一源 V 面板，不能将本筛查表述成盲测；不使用 U TX、源 V 或目标进行训练，未访问目标。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

### 固定源 V 同包结果

| 数据集 / 场景 | 输入直通 | R17 | 实增益 | DURA 对照 | 2% 教师 | 2% 无教师 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LoRa clean，60 包 | 60 | 59 | 60 | 60 | 60 | 60 |
| LoRa 低城区，60 包 | 30 | 15 | 34 | 33 | 34 | 34 |
| WiSig clean，360 包 | 360 | 360 | 360 | 360 | 360 | 360 |
| WiSig 低城区，360 包 | 200 | 332 | 190 | 313 | 333 | 332 |

LoRa DURA 对照为 LoRa 源训练 R69 结构适配；WiSig 为原固定 DURA v1。教师相对 LoRa 实增益改对 1、改错 1，净 0 包，配对 bootstrap 95% 区间 -3 至 +3，固定半区 +1、-1；相对 WiSig R17 改对 3、改错 2，净 +1 包，区间 -3 至 +6，半区 0、+1。未达每半区预登记门槛，未进入正式测试。

教师相对匹配无教师：LoRa 改对 0、改错 0；WiSig 改对 3、改错 2，净 +1/360，95% 区间 -3 至 +6；两侧 clean 净变化均 0。WiSig 源 TX3 为 44/55，对照为 46/55。逐 TX 和全部对照见本地教师版 `paired.json` 及各 `readback/*/screen-score.json`。不同 GPU 浮点轨迹可能不同；同一源 V 被用于锚点和设计选择并多轮曝光。

修正网络各 11,394 参数。教师版 LoRa/WiSig 分别曝光源 U 4,000/24,000、教师样本前向 8,000/48,000，训练 448.26/323.28 秒，PyTorch 峰值约 175.8/78.5 MiB；无教师对照 325.08/287.27 秒、约 122.6/55.3 MiB。专家及识别器祖先成本另计；实测端到端时延和正式七景待完成。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
