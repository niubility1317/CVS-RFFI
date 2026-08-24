# ADV3B02 Stage2-B CAPTA-P0 Target5实验报告

## 结论状态

- run ID：`adv3b02_stage2b_capta_p0_t5_s713101_20260824_v1`
- 当前状态：`LOCAL_VERIFIED`；尚未发布N607正式矩阵，尚无性能结论
- 设计定位：根据《面向CVS Phase2星上部署的轻型快速／多步域适应设计》实现CAPTA-P0协议安全近似。A3的rank-4域基由合法target support类残差估计，不声称复现设计中的地面跨域`U,V`
- 冻结checkpoint：`ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 协议：`p2_min_v1`、`VALIDATED_ONCE`，同row核对`capsule_id/split_id`
- 阶段：`Stage2-B`、`DA0_REG0`对`DA1_REG0`；REG0新类和unknown指标为`N/A`

## 候选与机制

- `A0`：复用既有冻结checkpoint、冻结原型和同一判决规则的`DA0_REG0`prediction
- `A1`：target support类中心与冻结原型按有效样本量做球面收缩
- `A2`：target support估计类均衡共享平移，再做球面收缩
- `A3`：从target support类残差SVD得到最多rank-4低维基，迁移共享平移后收缩
- `A6`：在support leave-one-out上从`{0,0.25,0.5,0.75,1}`选择source/target分数混合权重，并列时回退更高source权重；权重在query打开前冻结
- 不新增或训练协方差、LDA或持久分类头；原模型参数全部冻结，训练参数`0`，反向传播`0`次，适配更新`0`步
- Phase2运行时source/clean输入为`0`；query只读、逐样本面对全部冻结注册类，不读取query真值、角色、配额或反馈，不更新任何模型/适配状态

## 最小可证伪矩阵

- 配置：`configs/stage2b_capta_p0_target5_s713101_20260824.json`
- 单seed：`713101`
- Target5：receiver=`20-1、3-19、7-14、7-7、8-8`，`K5/new20`，场景=`leo_clear_weak、leo_low_elev_weak、leo_rain_weak`，共15个row
- 候选：`A1/A2/A3`，共45份`DA1_REG0`prediction；每份与同row既有`DA0_REG0`prediction独立配对评分
- 晋级门槛：候选相对`DA0_REG0`旧类等权均值至少`+1.0pp`且全矩阵旧类floor至少`+0.5pp`，同时无协议泄漏且预算达标
- 未达到门槛即记为`SCIENTIFIC_FAILURE_NO_PROMOTION`；本轮不扩大到Target25或多seed

## 输入、输出与运行环境

- 本地工作树：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\stage2b-lateblock-20260824`
- 本地Python：`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`
- N607 CWD：`/home/szu2070436088/2510044040/CV-SincNet`
- N607 checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 复用输入：既有`VALIDATED_ONCE`support/context/prototype、query received IQ package、row binding和`DA0_REG0`prediction；配置逐row登记精确路径
- 不可覆盖输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_stage2b_capta_p0_t5_s713101_20260824_v1`
- 预期artifact：release、remote smoke、45份DA1 prediction、90份paired score和1份矩阵summary、stdout日志
- GPU：预定`GPU0`；发布前preflight显示8张RTX3090均空闲，正式启动前再次核对

正式命令模板：

```text
python code/scripts/run_stage2_capta_target5_matrix.py --config configs/stage2b_capta_p0_target5_s713101_20260824.json --release-root <release/src> --output-root <run-root>/results --device cuda:0
```

系统技术停止规则：仅协议/row绑定错误、输出碰撞、错误checkpoint或checkout、无法产生合法prediction、scorer连接错误、确定性重复异常或进程归属不清时停止；不得因性能差停止。不得影响无关任务。

## 本地验证证据

- 原型传输RED→GREEN：A1/A2/A3、rank上限、输入不变和确定性测试通过
- 运行时RED→GREEN：A6门控、query状态不变、顺序不变、全类逐样本预测和零反向传播审计通过
- row绑定负测：修复前两个各自合法但不同`capsule_id/split_id`的support/query row会打开query；修复后在query IQ打开与prediction前严格失败关闭
- 合并聚焦与邻近回归：`40/40`通过
- Python编译：CAPTA核心与两个入口通过
- 真实checkpoint无query smoke：`PASS`；strict load=`true`，support=`60`，source=`0`，query=`0`，query_loaded=`false`，rank=`4`，trainable=`0`，backward=`0`，model_changed=`false`
- smoke artifact：`E:\type10-7\automation_reports\CV-SincNet\adv3b02_stage2b_lateblock_t5t25_s713101_20260824_v1\local_artifacts\d18_support_smoke\capta_smoke_a3.json`
- 独立审查：初审`P0=0、P1=1`；唯一P1为support/query同row未交叉绑定，已按负测驱动完成修复；唯一一次定点复审`P0=0、P1=0`，允许发布

## 发布状态

- Git分支：`codex/stage2b-capta-p0-20260824`
- 设计与追溯基线提交：`5eb66d0654b7281bba7aecbabde7be92c7ac8c82`，已push并核对远端OID
- 实现提交、release SHA、N607远端编译、PID/CWD/cmdline/GPU/log增长：提交与远端发布后填写

## 结果

当前无正式N607 prediction或评分结果。不得把本地smoke或工程闭合表述为正向收益。
