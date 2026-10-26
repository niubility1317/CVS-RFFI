# 监督去 SIG-TIME PAMF：功放与参考响应五组消融

- run_id：`20261026-phase1-PAMF-noSIG-supervised-ablation-m15-r02`
- group_id：`phase1-pamf-noSIG-supervised-structure`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

以严格监督去整条路径描述支路为基底，PA/参考响应2×2及同容量参考相位失配；15模型单元，复用基底3个，新增12个scratch50。

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
# 本次已固定范围与执行说明

用户确认以“监督、删除整个 SIG-TIME 路径描述支路”为基底，执行保留功放／参考响应、去功放、去参考响应、同时去两支路及打乱参考模板相位五组。用户随后明确取消“固定打乱局部顺序”，本 run 不包含该项。

每组 3 个模型种子，共 15 个模型单元。基底的 3 个 E50 模型、预测与评分复用原冻结产物，不重复训练或预测；4 个新增结构各从零训练 3 个种子，共 12 个新模型、600 个逻辑 epoch、133200 次更新。不从历史目标表现挑选父模型或轮次。

新增训练与原基底保持一致：L_s=6300、V=27000，U_s IQ 及 TX 标签均不读取；每模型 20＋10＋20 三段，每段重置优化器，累计 E50／11100 更新，L128、每轮 222 次更新。纯分类 CE、LS0，普通增强和原 practical 拼接 CE0.68 保持，原 160 维六类余弦头联合训练。无 CRIC、RDrop、U、EMA、PL 或另训头。

所有模型固定连接既有同契约 DURA-L（266951 参数）。主报告同时列直接识别和恢复后识别。跨日、跨接收机、双跨分别为每景 30000／126000／42000 个物理样本；综合每景 198000 个物理样本。每行完成三群体、七视图、两条读出后，独立 CPU scorer 才能连接真值。综合准确率按实际样本数加权，宏平均 F1 从合并混淆矩阵重算，随后汇总三种子均值、样本 SD 及同种子配对差。完整逐 TX/RX/day 和净纠错保持。

参数量（含原六类头）依次为 177025／165985／164225／153185／177025。删除分支同时改变容量与计算，不将掉分解释为真实硬件参数的因果可辨识性。PA／参考响应四格交互项预先固定为 baseline−noPA−noReference＋neither。

物理 GPU1、5、6、7 各最多 1 个本 run worker，计入其他 owner 后每卡最多 2 个 GPU worker，空闲显存至少 12000 MiB；不替换控制器、不停止或热改其他健康任务。单 CPU scorer；计时仅等待自然空卡，5 结构×2 路径×5 batch×5 轮，共 250 case-round。一次源 checkpoint 无 query smoke 和五结构 roundtrip／backward 先执行，PASS 后立即训练，不另设审批。独立 P0/P1 审查未发现阻断项，协议负测与汇总复算测试已通过。

代码／配置实际执行提交由发布包的 release_commit.txt 和 runtime.json 记录，启动读回后补入 experiment.json。命令、不可覆盖路径、数据句柄、种子、预算及停止规则以 experiment.json 为准。默认队列自动衔接训练、预测、评分、分析、参数／成本／时延与结果报告；当前没有新增性能结果。

## r02 技术恢复

仅补齐检查入口的 input_len=256；方法、数据、预算、种子、增强、DURA 与测试矩阵不变。新增本地回归测试通过，使用新不可覆盖 root，不改旧产物。

## 实际启动核实

VERIFIED：执行提交 `f922fc3fce6d1149c2f0c379050b6da3bba62cdd` 已推送并核对远端 OID；发布归档传输与远端编译通过。真实 checkpoint 与五种结构无 query 检查通过，参数量与预登记一致。已核对 4 个训练进程的实际 PID/CWD/argv/GPU；同 PID 日志增长和首轮训练更新已读回，U_actual、PL_seen、PL_accepted 均为 0。详细状态见 launch_verification.json。当前为 RUNNING，尚无新增测试结果。
