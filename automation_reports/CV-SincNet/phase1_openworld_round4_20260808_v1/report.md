# Phase1开放世界就绪表征第4轮回顾与候选冻结报告

目标模式：`GOAL_MODE=ACTIVE`

状态：`LOCAL_VERIFIED / PREREGISTERED / BLOCKED_EXTERNAL_SSH`

证据边界：`PHASE1_SOURCE_ONLY_DEVELOPMENT_NON_CONFIRMATORY`

## 1.目标

本轮继续优化Phase1地面开放世界就绪表征，不执行真实unknown生命周期、多接收节点协同或新类注册。设计优先级为：高跨TX泛化、已知类与LEO弱信道floor不退化、少做域对齐、尽早发布可证伪实验。

## 2.前三类失败证据

| 路线 | 同行证据 | 结论 |
|---|---|---|
| V30/V31 post-hoc feature repair | V31 `strong_target1_pass=0/80`；最优unknown FAR约54.2% | 后处理或局部feature repair不能修复hard receiver/TX floor |
| 双读出hard disagreement | proxy/held FAR分别改善0.75pp/1.50pp，但source full accuracy下降3.50pp | 二值一致性硬门以known coverage换取极小拒识收益 |
| 冻结C的prototype/kNN密度读出 | 12个primary全部FAR失败；prototype/kNN平均FAR为70.625%/65.792% | 排序信息存在，但source Q98绝对阈值不能跨TX迁移 |
| ManyTx RealOE | F1--F4 proxy AUROC平均+0.13180、proxy FAR改善14.575pp；held FAR反而+13.50pp；known保护仅1/6通过 | 全局OE能量训练产生proxy专化并扰动身份骨干 |

## 3.冻结教训

1.不再发布threshold-only、Q值扫描、hard disagreement或同类prototype/kNN读出实验。
2.不再把ManyTx OE梯度直接作用于完整身份骨干，也不通过减小`lambda`继续同一路线。
3.proxy指标只用于source-only开发晋级信号；外层held-TX完整报告跨TX方向，但不套用Phase3真实unknown的5%强门。
4.下一候选必须把身份分类路径与拒识证据路径解耦，并可导出`z_id/d_class/e_unknown/q`。
5.不新增receiver/day/channel对齐；任何正信号必须来自标签置换等价、类条件几何或尾部建模。

## 4.最小实验框架

- 数据：沿用相同6-fold TX互斥角色与物理RX/day口径，不因方法变化重验数据。
- 配对：每fold保留同seed基线C与唯一新候选；若加入技术消融，只允许共享同一身份路径并拆分拒识头梯度。
- 资源：8卡并行，每卡不超过2个训练进程；不先做长时间超参搜索。
- 选择：六fold完整运行；不按单fold、proxy最优值或中途性能选择。
- 晋级：模型健康、known跨接收机性能无明显退化、最低类别与LEO弱信道floor无严重下降、source proxy相对C产生明确正信号、真实checkpoint bundle导出，共五项。

## 5.设计裁决

独立监督选择简化`GI-EpiOR`，结论为`CHOICE=B / ALLOW_IMPLEMENTATION=YES / ALLOW_RELEASE=NO`。非参数NCT因把inner-LOTO pseudo-unknown尾误写成known conformal p-value，且93%--97% coverage不能满足reject计错后的known drop不超过2pp，作为正式候选拒绝；它只保留一个无阈值连续比值消融。

正式候选只包含：冻结GeoSat-C身份路径、整TX episodic排除、类别置换等价的相对几何descriptor和两层低容量拒识头。删除`z_E`投影、正交协方差、view loss及所有receiver/day/channel对齐。

## 6.唯一方法定义

对单位特征`z`及episode reference中类别`c`定义：

```text
d_c=(1-cos(z,mu_c))/(MAD_c+epsilon)
d_(1),d_(2)=两个最小类条件距离
g=[d_(1),d_(2)-d_(1),d_(1)/(d_(2)+epsilon)]
e_epi=sigmoid(MLP_theta(g))
```

`MLP_theta`固定为`3->8->1`，仅它可训练；输入特征始终stop-gradient。每个inner episode把一个source TX的全部物理记录从reference中排除，该TX query只作`episodic nonregistered negative`。每TX按canonical physical ID `(tx,rx,day,eq,sig)`确定性50/50拆为reference/query；两者必须互斥。

训练固定为balanced full-batch BCE、seed`7281105`、Adam `lr=1e-2`、`weight_decay=1e-3`、200 epoch。唯一二元边界为`e_epi>=0.5 -> unknown`；不得使用quantile、proxy/held校准、阈值扫描或target/query真值。`p_local`恒为冻结C logits argmax。无训练消融只输出`d_(1)/(d_(2)+epsilon)`连续分数，不形成第二候选。

## 7.六foldPhase1晋级门

1.角色互斥、整TX排除、reference/query物理互斥、finite、identity梯度为0且head梯度非零。
2.reject/defer按known错误计后，每fold clean overall、min-class、min-RX、min-day相对C均不低于-2pp。
3.每fold三种LEO的mean、floor、strict mean、strict floor相对C均不低于-2pp。
4.source proxy unknown相对冻结C产生明确正信号；六个外层held-TX作为跨TX方向诊断完整报告，不设置Phase3式`FAR<=5%`强门。
5.真实checkpoint形成单backbone bundle，包含`p_C/e_epi/d_class/mu/rho`，eager与TorchScript最大绝对差不高于`1e-5`，并回执延迟、状态字节和显存。

clean实验先关闭模型健康、known保护、proxy正信号和bundle导出；通过后立即发布三种LEO视图关闭弱信道floor。五项中任一失败即`REJECT`，不扫阈值、不调loss权重、不选择有利fold。发布前两项P1已经由episode采样/梯度闭环和真实bundle/成本smoke关闭。

## 8.当前实现分工

Terra实现者只负责新core/evaluator/tests；主Agent负责设计稿、launcher、报告、diff整合与最终判决。完成focused测试后由原独立监督复核P0/P1，再创建不可覆盖N607 run并交给唯一Luna/max runner。

正式Phase3仍等待可验证的`emission_event_id/satellite_reception_id`绑定资产。本轮不得把source-held TX结果写成真实多卫星unknown性能。

## 9.本地实现与真实feature smoke

| 文件 | 用途 |
|---|---|
| `code/cvsrffi/phase1_gi_epior.py` | 物理互斥split、整TXepisode、类条件几何、固定小head和runtime |
| `code/scripts/eval_phase1_gi_epior.py` | source-only fit、不可变bundle/TS导出及独立score |
| `code/tests/test_phase1_gi_epior.py` | 协议负测、梯度、置换、outer zero-fit、parity和synthetic正例 |
| `code/scripts/launch_phase1_gi_epior_clean6_20260808.sh` | 6个clean fit＋6个clean score的一次性CPU launcher |

本地`ssr-gpu`验证：新测试8项通过；连同`ow_feat`、dual-readout bundle、logits scorer和exporter轴标签测试共35项通过；三个Python文件`py_compile`通过；launcher`bash -n`通过，DRY_RUN精确为6 fit＋6 score。

真实checkpoint衍生feature smoke使用既有只读`E:\type10-7\remote_artifacts\phase2_adv3b02_features\features.npz`，SHA256=`db559d78db305894307851750ef7d698db387f0984ff13c980fea99db85b8532`。只取其中2400条source、6个TX；每TX200 reference＋200 query，物理重叠为0；4000条非source记录排除于fit。结果：identity梯度`0`、head梯度`0.0782326`、eager/TS最大差`0`、CPU 512条平均约`0.280ms`、runtime状态`4028B`。bundle、runtime、receipt SHA256分别为`1cb822578a396aecfc8bd0333b819c8996c5854394ea446e776ef1c31a7e427f`、`fff8b13e6c574ee0dfaaca0d92dc28033802ab4de48994ba487b3dbf00fe99f5`、`cb0f856171ce7ae117769212bd342692e1250e3451b91a0e8c91a159a3095877`。该smoke只关闭接口、梯度、bundle和成本P1，不读取或解释target性能。

拟发布run ID：`phase1_gi_epior_clean6_20260808_v1`。输入只读复用上一轮远端6个C-arm clean NPZ；运行6个小head fit和6个score，均为CPU，不重新运行backbone。clean阶段按六折整体判断模型健康、known保护、proxy正信号和bundle闭环；通过后立即进入三种LEO视图实验。

## 10.独立审查与Git冻结

独立监督复核结论：`VERDICT=APPROVE / P0=0 / P1=0 / ALLOW_CLEAN6_RELEASE=YES`。审查确认35项相关测试通过、Python编译与launcher语法/DRY_RUN通过、六个C-arm NPZ输入闭合为6次fit＋6次score，且正式路径`BACKBONE_CALLS=0`。本轮不追加对齐、阈值搜索、安全包装或报告平台。

实现commit：`d3b5b610987f5ce8f38262875b5bb7ace1ba3143`。

| 文件 | SHA256 |
|---|---|
| `analysis/phase1_gi_epior_design_20260808.md` | `f387d94d5a0d05dccee491b0c81e04260ae09232469437cb3194f905631930e6` |
| `code/cvsrffi/phase1_gi_epior.py` | `09fca031def540715dd52c36ed82ab4822da8988b78ea3c65d624d087b36f458` |
| `code/scripts/eval_phase1_gi_epior.py` | `2821f66f69b5a954cb50c5b6cf10109749d721da97c606c0d1624c981fbc2394` |
| `code/tests/test_phase1_gi_epior.py` | `da512795e3027c77f8a8738c44c76a0c4e09ca051cebcf46e3c5a2f7153d192d` |
| `code/scripts/launch_phase1_gi_epior_clean6_20260808.sh` | `a18bfa57e2807a4b77624371408cc678d9b2f2a0fc7d8e0ebc61b6b44ad83477` |

## 11.N607唯一runner交接

- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_gi_epior_clean6_20260808_v1`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_gi_epior_clean6_20260808_v1`
- release：以本报告commit的Git archive落入不可覆盖目录`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_gi_epior_clean6_20260808_v1_<commit8>`。
- 输入：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1`下六个`F1C`至`F6C`的`features.npz`，只读复用，不下载。
- 环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD=`<release>/code`；CPU-only；每个子进程固定`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。
- 精确入口：`env RUN_ID=phase1_gi_epior_clean6_20260808_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release>/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python INPUT_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_manytx_realoe12_physrx_v2_postfreeze_20260808_v1 bash <release>/code/scripts/launch_phase1_gi_epior_clean6_20260808.sh`。
- 预期：六份`gi_epior_bundle.npz`、六份`gi_epior_runtime.ts`、六份`fit_receipt.json`、六份`clean_metrics.json`、六份`clean_scores.csv`，以及fit/score completion和12份stdout。
- 停止规则：仅协议错误、覆盖风险、输入/hash/checkout错误、确定性异常或零输出触发技术停止；不得按性能停止。任一fit失败则不启动score；不修方法、不调参、不重试。
- 回收：只回收小型receipt、metrics、scores、日志与manifest；不回收输入NPZ、checkpoint或runtime/bundle大文件。runner只报告技术闭环，不解释性能。

clean实验完成后由主Agent读取六折同行结果。模型健康、known保护、source proxy正信号或bundle闭环任一失败即`REJECT`；外层held只作诊断，不作为额外发布门。clean四项通过后立即登记并发布三种LEO视图验证。

## 12.首次发布尝试终态

唯一runner按预注册流程先执行N607直连只读预检，SSH端口返回`Connection refused`；随后仅尝试一次已验证lab bridge，bridge的22端口同样返回`Connection refused`。本地已确认`ssh/scp`进程为0、TCP22连接为0。

本次阻塞发生在release落地之前：远端release、run、log均未创建，GI-EpiOR未启动，没有实验退出码或性能结果。因此状态为`BLOCKED_EXTERNAL_SSH / NO_PERFORMANCE_RESULT`，不是方法失败，也不消耗实验重试。连接恢复后仍从commit`2e400d12b93ba492bf8bba5504095fdaa0b8ccc7`和同一不可覆盖run ID首次发布；不得另建候选、调参或先看局部性能。
