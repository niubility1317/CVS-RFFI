# 以实际B函数为先验的残差LocalRidge与support监督adapter联合适应注册

- run_id：`20261001-phase2-d92-anchor-joint-support-m2-r01`
- group_id：`d92-anchor-joint-support`；类别：`diagnostic`；阶段：`Phase2-support-only-diagnostic`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ARTIFACTS_COMPLETE（实际状态按events.jsonl及独立证据更新）

## 目的与对照

固定旧类参考测度与核尺度，B分类函数作为C先验；全类残差闭式LocalRidge经平滑CE联合学习函数坐标adapter；保留完整interaction，不做参数网格。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

LOCAL_VERIFIED/VERIFIED：AJLR单一数学结构已实现，54个不同相关检查通过；core22、ops23、entry4、summary5。整数dtype、summary唯一参考pair、实际1800路径和原生bool验收边界均已修复；完整正常复算与10种篡改拒绝通过，独立审查NO_UNRESOLVED_P0_P1。实际B函数prior、固定旧物理参考测度与tau/gamma、残差闭式头、全类CE伴随联合微调；只有R0/R_AJLR_seq。完整四row/160parent预登记与既有source-only缓存身份preflight已核实，新run/release/archive无冲突。未发布、未启动，尚无真实性能；A与B−A=N/A，目标ACTIVE。

RUNNING/VERIFIED：AJLR已单次发布启动，实际runtime a1a003f59e8ed14640a252ada8290a03af3420c5。supervisor341635、workers341647/341648的实际PID/argv/cwd独立匹配；两个rx3 row训练中，两个rx1待排队。CPU两lane/BLAS2、query/source样本不读、encoder/checkpoint不加载。首次readback1790790609；完整160parent尚未结束，尚无真实性能分析；禁止重复publish/重启/热修改。54不同相关检查与唯一P0/P1已完成。A与B−A=N/A，目标ACTIVE。

ARTIFACTS_COMPLETE/VERIFIED：AJLR完整四row/160parent已结束，supervisor及全部workers退出；完整marker/state/实际计数独立读回一致。实际更新2592次、头拟合36564次、latent SVD3240次。未读query/源样本；完成不代表性能改善。下一步单次独立support分析与完整训练诊断，root sole owner，目标ACTIVE。

ANALYSIS_R01_FAILED/VERIFIED：完整训练产物保留；首次独立汇总PID386856已退出，无summary/output。实际阶段流为prep B→fit B→prep C→fit C，汇总器重建顺序错误，6项回归检查通过。仅修复分析器，保持严格流核对；预登记新analysis release d92_anchor_joint_analysis_20261001_r02，root sole owner，尚未启动，不重跑训练。

ANALYSIS_R02_RUNNING/VERIFIED：使用阶段顺序修复的独立分析已单次启动，PID394399/argv/cwd独立读回匹配，analysis commit fc5cd282f93e2b7cc1a2fcd686b4217fb4abfe4a。本地等待handle99201有效，完整summary尚未产生。四row/160训练已完成；r01失败证据保留，不重复训练或任何analysis启动，目标ACTIVE。

ANALYSIS_COMPLETE/VERIFIED：AJLR完整独立r02已正常结束，160parent/1800path和版本证据核实，完整80行K×新增类数报告已生成。96个新类存在的可测support持出配置中，R0 B71.0417/C旧63.7674/新54.8464/H58.4168；AJLR B71.1458/C旧62.9253/新57.0938/H58.9841。新类+2.2474pp、H+0.5673pp，但C旧−0.8420pp，注册下降从7.2743增至8.2205pp，不能称全面优于或自动晋级。A/B−A=N/A；非query或独立验证。完整训练快照3240阶段/2673911234B已采集；此前2650592222B是写盘尚未结束的观察值，已按完成后stat更正。只读diagnostics extract首次handle51314已退出，无输出；JSON源码字面量超过Python解析限制，尚未进入提取。仅修复压缩数据传输后复用同一完整snapshot，不重复采集或训练；目标ACTIVE。

DIAGNOSTIC_TRANSPORT_READY/VERIFIED：旧AJLR只读提取源码literal超过Python解析长度，完整snapshot和分析结果保留，仅改为完整JSON→gzip→base64数据载荷；远端仍stdlib解码，没有字段删减、eval/pickle、新拟合/求解或数学变更。14项相关合成检查通过（5.97s），4项新增、10项既有回归。root在Git交付后仅重试extract，完整snapshot不重采，目标ACTIVE。

DIAGNOSTIC_EXTRACT_RETRY_STARTED：root已单次启动压缩JSON只读extract，local handle=60237；完整snapshot为2673911234B，不重采、不重训。首次进程读回尚未看到远端extract进程，本地可能仍在解析/压缩载荷；不能从远端未出现推断失败或完成。原失败51314保留。Affine r02保持PID501716/handle2931，原分析不重复。目标ACTIVE。
