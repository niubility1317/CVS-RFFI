# Phase1 ADV3B02六折配置等价重训练预注册报告

## 1.状态、目的与边界

- 实验ID：`phase1_adv3b02_clic6_20260813_v1`
- 当前状态：`INDEPENDENT_REVIEW_ALLOW / READY_FOR_N607_F1_SMOKE / NO_PERFORMANCE_RESULT`
- 本地实现操作者：Terra/max；N607唯一runner：待主控完成独立P0/P1复审后指定。
- 方法身份：`ADV3B02_CORE90_SOFT_E200_CLIC_EQ_RHO07_FINAL`。
- 历史机制锚点：`ADV3B02_CORE90_SOFT_E200`。本任务是保持其完整机制和损失配置的配置等价重训练，不是历史checkpoint的字节级复现。
- 目的：为F1—F6各建立一个source-only ADV3B02训练入口，使其`split_mode=tx_rx_day_1_6_3`、`0.07/0.63/0.30`数据角色、seed、epoch与对应CLIC折一致。后续独立入口才会生成目标已注册known的rich reference；本任务不训练、适配、更新、选择或读取任何目标包、truth、query或scorer结果。
- 假设：在逐字段训练/known-test语义配置等价的前提下，六个ADV折可作为每个CLIC折的合法已知类DG比较组件。该假设尚无性能结论。

ADV3B02在本任务中没有独立source-frozen的端点真实unknown决策规则。因此其目标真实unknown指标为`N/A`，不能记为零，不能以历史proxy/vaccept替代，也不能作为70%门的通过证据。

## 2.冻结训练矩阵

|折|候选目录|source训练TX顺序|held-known验证TX|proxy-unknown TX|GPU|seed|
|---|---|---|---|---|---:|---:|
|F1|`F1_ADV3B02_CLIC`|`20-15,20-19,6-15,8-20`|`14-7`|`14-10`|0|392002|
|F2|`F2_ADV3B02_CLIC`|`14-10,20-19,6-15,8-20`|`20-15`|`14-7`|1|392002|
|F3|`F3_ADV3B02_CLIC`|`14-10,14-7,6-15,8-20`|`20-19`|`20-15`|2|392002|
|F4|`F4_ADV3B02_CLIC`|`14-10,14-7,20-15,8-20`|`6-15`|`20-19`|3|392002|
|F5|`F5_ADV3B02_CLIC`|`14-10,14-7,20-15,20-19`|`8-20`|`6-15`|4|392002|
|F6|`F6_ADV3B02_CLIC`|`14-7,20-15,20-19,6-15`|`14-10`|`8-20`|5|392002|

每折训练、held-known和proxy-unknown TX集合两两互斥。启动器在派生命令前检查角色数量和互斥性；`--dry-run`固定按F1→F6输出六条完整命令且不创建run/log根目录。正式模式在创建根目录、PID、日志或启动任一child前，先遍历检查六个`F#_ADV3B02_CLIC`的output_dir和log_path；任一后续折碰撞都必须整体拒绝，不能让F1—F5先启动。

## 3.冻结配置

|类别|固定值|
|---|---|
|数据与时间表|`split_mode=tx_rx_day_1_6_3`；`labeled_ratio=0.07`；`unlabeled_ratio=0.63`；`source_val_ratio=0.30`；`epochs=200`；`label_epochs=130`；`pseudo_epochs=70`；`from_scratch=true`；`seed=392002`。|
|选择边界|`checkpoint_selection=final_only`；`phase1_source_val_selection_only=true`；`best_metric=source_val_sat_hmean`仅保留source validation遥测，不参与checkpoint选择；`enable_joint_safe_guard=false`，避免held-out joint guard路径。|
|历史机制与守卫|保留historical`set_candidate_defaults`和`ADV3B02_CORE90_SOFT_E200`分支的全部有效与零权重旗标：ground prototype、feature mask、TX/RX geometry、prototype memory、open-world feature、z-id compact、proxy unknown、soft unknown mixup、source episode、prototype export/fusion、satellite consistency与半监督/域/Fishr项。|
|关键分支值|`lambda_open_world_feat=0.0024`；`lambda_zid_compact=0.032`；`lambda_proxy_unknown=0.0045`；`proxy_unknown_core_quantile=0.90`；`proxy_unknown_accept_quantile=0.85`；`proxy_unknown_vaccept_cvar_alpha=0.30`；`lambda_soft_unknown_mixup=0.0045`；`lambda_source_episode=0.0035`；`lambda_sat_cls=0.68`；`lambda_sat_cons=0`。|
|其余历史损失项|`lambda_u=0.16`；`lambda_ent=0.01`；`lambda_domain=1`；`lambda_adv=0.35`；`lambda_group_ce=0.16`；`lambda_fishr=0.04`；`tau_min=0.92`；`tau_max=0.97`；`pseudo_quantile=0.86`；`use_ema_teacher=true`。完整逐旗标receipt由启动器`--print-contract`给出，并可用`--validate-contract-file`逐字节校验。|
|LEO训练压力|训练场景固定为`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`；schedule=`1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。这属于source-only训练时的固定弱扰动，不引入目标接收机输入。|

## 4.本地实现与验证

|文件|用途|状态|
|---|---|---|
|`code/scripts/launch_phase1_adv3b02_clic6_v1_20260813.sh`|六折训练入口、冻结合同、全六fold预检、根目录防覆盖、PID表与限定技术停止逻辑|已实现|
|`code/scripts/smoke_phase1_adv3b02_clic_f1_v1_20260813.sh`|从正式启动器机械提取F1命令，在独立根执行恰3个完整source-only forward/backward/optimizer batch；不读取source-V、target、query或test row|已实现|
|`code/SSDG/train_ssdg.py`|新增默认关闭的三批技术烟测生命周期；flag=0不改变正式训练，flag=3仅接受冻结F1合同并在第三个有限、有效optimizer step后不可覆盖地写技术receipt|已实现|
|`code/tests/test_phase1_adv3b02_clic6_baseline.py`|六折命令/角色/全部关键旗标、真实训练器解析与运行时dry-run、无目标侧接口、全六fold预检、根目录保护和合同漂移拒绝|已实现|
|本报告|预注册、N607交接与结果表模板|已建立|

|验证命令|结果|
|---|---|
|`python -m pytest code/tests/test_phase1_adv3b02_clic6_baseline.py -q`|29/29通过。除原7项六fold合同外，覆盖F1独立入口、仅0/3 batch、错误方法在数据构建前拒绝、完整正式F1 parser profile逐字段绑定、方法/loss/零权重/布尔/场景schedule漂移拒绝、真实1/2 batch首轮不足时在任何source-V评估前拒绝、三次有限有效optimizer step receipt、非有限batch拒绝、receipt不可覆盖、`BASH_ENV`伪造formal profile和PATH伪bash拒绝，以及flag=0不约束正式F2—F6。|
|`bash -n code/scripts/launch_phase1_adv3b02_clic6_v1_20260813.sh`|通过。|
|`bash code/scripts/launch_phase1_adv3b02_clic6_v1_20260813.sh --dry-run`|通过；恰六条F1→F6源侧命令。|
|真实`train_ssdg.build_arg_parser().parse_args(...)`和`train(args --dry_run)`|六条命令均接受`--from_scratch true`、`--checkpoint_selection final_only`、`best_metric=source_val_sat_hmean`和全部训练旗标；逐条追加`--dry_run`后，真实`train_ssdg.train(args)`均返回0且未构造数据/模型。|
|F6碰撞行为测试|预建`RUN_ROOT/F6_ADV3B02_CLIC`后，启动器在任何PID、日志、根目录或F1—F5目录写入前以`refusing to overwrite planned fold output: F6_ADV3B02_CLIC`退出。|
|历史旗标逐项差异审计|历史ADV3B02命令157个旗标与新命令162个旗标比较：除预注册的split/比例/路径/身份、`best_metric=source_val_sat_hmean`、`enable_joint_safe_guard=false`变化及5个新增final-only／三角色旗标外，`unexpected_changed=[]`、`missing_historical=[]`、`unexpected_added=[]`。|
|`python -m py_compile code/SSDG/train_ssdg.py`|通过。|
|`git diff --check`|通过。|

Git实现谱系：初始Task1入口提交为`72640e30fe713d5aca365e2cb07fa52522fa4d02`；独立审查NO-GO后的代码/测试修复提交为`d86926470065c0d9a68b9d94e9f6a79e2032ea47`。修复版本文件SHA256：launcher=`FCABDA8ABA3A29D8DEE81D1E16C90A9E211402B10270B1C80FB9B56B832DAD22`；测试=`FB825FDECC75E7F0F2E5B447454E51FC91C4FA280E230E71118CD6231F086372`。未完成独立P0/P1复审、N607 preflight和唯一runner handoff前不得同步或启动。

## 5.N607发布合同

- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`。
- 待建只读release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_clic6_20260813_v1_<Task1Commit>`。
- 环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 工作目录：release内项目根；`CODE_ROOT=<release>/code`。
- 启动器：`<release>/code/scripts/launch_phase1_adv3b02_clic6_v1_20260813.sh`。
- 单次正式命令：`RUN_ID=phase1_adv3b02_clic6_20260813_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release>/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_clic6_20260813_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_clic6_20260813_v1 bash <release>/code/scripts/launch_phase1_adv3b02_clic6_v1_20260813.sh`。
- 正式launch次数：1；retry：`NO`；不复用、不恢复或覆盖任一旧run。
- 本地→远端映射：launcher、测试和本报告均来自上述Git commit；正式同步仅发送launcher和经审核的release archive，不修改远端源文件。

正式六fold启动前有且只有一次强制F1技术烟测门：runner必须从同一release以`bash <release>/code/scripts/smoke_phase1_adv3b02_clic_f1_v1_20260813.sh`启动，固定写入`runs/.smoke_phase1_adv3b02_clic6_20260813_v1_F1/F1_ADV3B02_CLIC`与对应logs根。PID文件记录的是前台烟测wrapper PID；训练子进程及其CWD/cmdline须由runner在运行期另行核验。技术PASS严格要求receipt schema=`cvs.phase1.adv3b02_technical_smoke.v1`、`batches=forward=backward=optimizer_attempts=optimizer_effective_steps=3`、nonfinite=0、source-val/target/query/test row访问和selection反馈全为0。烟测失败、缺receipt或进程绑定不闭合时，`FORMAL_INVOCATION=0`，不得调用正式六fold启动器；烟测通过后才允许唯一formal invocation=1。烟测不读取性能，也不构成checkpoint或候选选择。

首次Task2提交`fc6e4146`经独立审查判定`P0=0/P1=2/NO-GO`：直接训练器调用可漂移`lambda_proto`等方法profile，且真实loader一轮不足3批时会进入source validation。修复版本在任何通用校验或数据构建前，从同一release的正式launcher F1 dry-run机械解析完整parser namespace，除隔离output/export/control字段外逐字段精确比较，`wisig_pkl`亦不得漂移；同时在第一轮batch loop结束、任何source-V评估之前对实际batch数不足3立即fail-closed且不写receipt。该修复须重新获得独立`P0=0/P1=0`后方可进入N607。

第二轮独立攻击发现，继承的`BASH_ENV`可重定义`printf`并伪造formal F1 dry-run，PATH也可替换用于profile恢复的bash。最终实现只接受平台系统bash的绝对解析路径，并以最小环境运行formal dry-run：固定`PATH=/usr/bin:/bin`，仅保留locale及Windows启动所需系统变量，显式排除`BASH_ENV`、`ENV`、`SHELLOPTS`、`BASHOPTS`、`WSLENV`和`BASH_FUNC_*`。这是Task2最后一轮release-engineering修复；关闭真实入口后，不再以同进程恶意猴补、重复签名或额外authority扩大阻断。

最终独立终审基线：`edacc54d1d9fe8b5477e6da71d5f40ca4e8104af`；结论`P0=0/P1=0/ALLOW`。fresh证据为29/29聚焦测试、环境攻击2/2、正式六折真实解析并调用`train(...--dry_run)`6/6、训练器`py_compile`、两份`bash -n`、最小POSIX环境formal dry-run=6行/smoke=1行及diff-check全部通过。冻结SHA256：训练器=`5BBC34C8204E93076F297A524D89294D8265EEADD03E8513A6BE116569221D10`；正式launcher=`FCABDA8ABA3A29D8DEE81D1E16C90A9E211402B10270B1C80FB9B56B832DAD22`；烟测wrapper=`80B2D3AA70FEF909B7FD027C01A5E5197B8EB37308605D5338A8754C354F2F7D`；测试=`8432F5F7773FAB11FA4C069BEA4B52437C172161106ED18451D849061C181FBE`。该ALLOW只授权同release的N607 F1技术烟测；烟测receipt闭合前正式六fold调用次数必须保持0。

## 6.预期工件、健康与停止规则

每折预期工件为`final_ssdg.pth`、训练器默认`metrics_epoch.csv`/`metrics_epoch.jsonl`、`phase2_zid_prototypes.pt`、完整`F#_ADV3B02_CLIC.out`日志、`F#_ADV3B02_CLIC.pid`和`status/F#_ADV3B02_CLIC.status`。外层预期工件为`frozen_contract.txt`、`outer.pid`、`pids.tsv`及必要时的`systemic_stop.status`。训练器当前ADV路径不承诺不存在的CLIC专属terminal/config JSON；完成性以final checkpoint、嵌入checkpoint的args、完整log、PID/GPU绑定和本任务冻结合同共同核验。

N607 runner在启动前必须确认release哈希、训练器可编译、数据可见、run/log根均不存在、六个planned output/log路径均不存在、GPU0—5资源可用及每卡训练进程数不超过2。启动后立即核验outer PID、六个child PID、CWD/cmdline/run-root、GPU映射和日志增长。停止仅限协议/hash/覆盖错误，或至少两折在有效`final_ssdg.pth`前出现相同确定性异常指纹；停止时仅向已登记的本run child PID发送有限`TERM`，保留所有部分工件。accuracy、loss、DG、proxy或任何性能值绝不构成停止、重试或选择依据。

## 7.风险与后续接口

- 训练入口保留历史方法机制，不等于历史`0.10/0.70/0.20`原件，也不保证性能与历史数值相同。
- 本任务只闭合训练配置。六个checkpoint完成后，独立Task2/Task3接口才能按既定权限生成3120条盲态预测、开启truth-side rich known metrics并严格ingest six-fold ADV reference。
- ADV与CLIC不要求使用同一目标capsule字节；比较前必须重新核验双方训练数据语义配置、known-test语义配置和指标定义逐字段等价。
- 缺任何fold的合法ADV rich reference时，原combined scorer必须维持`FAIL/CANNOT_ESTABLISH`。

## 8.结果表模板

|折|候选|机制类别|训练TX／known／proxy|K-shot|seed|known DG overall|三scene已知指标|macro/min-class/min-RX|min-day|真实unknown拒识|checkpoint／log|结论|
|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|
|F1|`F1_ADV3B02_CLIC`|source-only ADV3B02 DG comparator|见第2节|N/A|392002|N/A|N/A|N/A|N/A|`N/A—no independently source-frozen endpoint rule`|待运行|NOT_RUN|
|F2|`F2_ADV3B02_CLIC`|source-only ADV3B02 DG comparator|见第2节|N/A|392002|N/A|N/A|N/A|N/A|`N/A—no independently source-frozen endpoint rule`|待运行|NOT_RUN|
|F3|`F3_ADV3B02_CLIC`|source-only ADV3B02 DG comparator|见第2节|N/A|392002|N/A|N/A|N/A|N/A|`N/A—no independently source-frozen endpoint rule`|待运行|NOT_RUN|
|F4|`F4_ADV3B02_CLIC`|source-only ADV3B02 DG comparator|见第2节|N/A|392002|N/A|N/A|N/A|N/A|`N/A—no independently source-frozen endpoint rule`|待运行|NOT_RUN|
|F5|`F5_ADV3B02_CLIC`|source-only ADV3B02 DG comparator|见第2节|N/A|392002|N/A|N/A|N/A|N/A|`N/A—no independently source-frozen endpoint rule`|待运行|NOT_RUN|
|F6|`F6_ADV3B02_CLIC`|source-only ADV3B02 DG comparator|见第2节|N/A|392002|N/A|N/A|N/A|N/A|`N/A—no independently source-frozen endpoint rule`|待运行|NOT_RUN|
