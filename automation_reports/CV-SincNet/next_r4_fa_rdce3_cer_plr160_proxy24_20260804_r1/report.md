# NEXT-R4 FA-RDCE3×CER-PLR160 Proxy24实验报告

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

## 1.实验身份

- 实验ID：`next_r4_fa_rdce3_cer_plr160_proxy24_20260804_r1`
- 日期：2026-08-04
- 主agent：`gpt-5.6-sol/high`
- 科学实现/审查：`gpt-5.6-terra/max`
- 唯一N607 runner：冻结提交、命令和路径后由`Luna/max`接管；当前尚未发布
- 目标：验证共享3维Fisher锚定域位移`FA-RDCE3`与轻量残差头`CER-PLR160`能否同时改善域适应和新类注册，而不恢复D92的不必要稠密计算
- 对照：同一row、同一query、同一K下的qKNN基座Q，以及四态内的配对差值

## 2.统一性能状态

|状态码|唯一中文主名称|主指标|
|---|---|---|
|`DA0_REG0`|域适应前/新类注册前|old BA、old-floor、总正确数；seen-new/H=`N/A`|
|`DA1_REG0`|域适应后/新类注册前|old BA、old-floor、总正确数；seen-new/H=`N/A`|
|`DA0_REG1`|域适应前/新类注册后|old BA、seen-new、H、all-floor、总正确数|
|`DA1_REG1`|域适应后/新类注册后|同上；联合主结果|

日志、prediction/score artifact、CSV/JSON、表格和结论必须同时保存`state_id`与中文主名称。REG0的`seen_new_acc`和`H_old_new`写`N/A`而不是0。禁止只写before/after；任何提升必须给出同row或同聚合层的起点、终点和差值。

## 3.冻结假设与矩阵

- 候选：`NEXT-R4-FA-RDCE3-CER-PLR160`
- 协议：`p2_min_v1`
- 语义：`phase1_seen_class_loco_directional_proxy`，不构成正式新类注册声明
- receiver：`1-1`、`18-2`
- held-class：每个receiver 6类，运行时唯一排序
- K：1、5；K1是K5逐类support前缀
- 逻辑行：`2×6×2=24`
- prediction：144个唯一预测；artifact：192个含K1 alias的arm记录
- DA1_REG1必须逐字节复用DA1_REG0的FA状态；K1 H必须逐logit精确alias Q
- 禁止参数、seed、receiver、K、阈值或矩阵搜索
- Phase1资产统计锁：`D_v=tau`、`D_F=1/spectrum`、`rho=sqrt(3)`、`kappa=spectrum/(spectrum+tau)`；只使用D106 canonical rank-3 receiver-day nuisance basis和旧类等权聚合，非正/非有限即scientific reject

主要比较：`DA1_REG0−DA0_REG0`、`DA1_REG1−DA0_REG1`、固定DA下的`REG1−REG0`、K5的`H−Q`及DA×head交互。

## 4.版本与本地验证

当前实现提交：`dab44012`。已落地的相关提交包括：

- `5179c541`：CER-PLR160核心
- `09a07ccf`、`8cb723f9`：矩阵与动态计数一致性
- `45e148c1`：FA-RDCE3核心及四态指标命名
- `a5d80db0`：设计到实现追踪
- `66d14379`：关闭CER的R0/R1表示契约与alias前query闭合P1
- `e356c15a`：独立truth-side scorer及四态明确指标输出
- `54f0723d`：新类按held-class宏平均；注册遗忘改为固定DA下注册前减注册后，并补齐总体与逐receiver聚合
- `6f779325`、`b8b26e90`：prepare→predict→score CLI、动态capsule和行身份闭合
- `8dcfdd69`：预测侧query绑定升级为v2扁平physical/observation ID，递归拒绝按类query字段
- `b849383b`：冻结FA-RDCE3的Phase1统计公式和来源
- `744a0b5d`：12个FA-RDCE3聚合资产构建器
- `cf9b3d65`：CLI适配扁平query绑定并闭合prepare→predict→score权威哈希链
- `b1aab4cb`、`c6e83488`：兼容合法`day_ids`并修正同源Phase1 ID只与当前row support/query隔离
- `b02738f8`、`74991fb5`、`dab44012`：24行capsule构建、一次性validator receipt、正式qKNN lock适配及predictor-safe metadata

当前R4合并验证为58项全部通过，并完成两个真实输入检查：12个Phase1资产的in-memory构建，以及正式落盘资产的逐文件SHA/checkpoint绑定复算。提交`8dcfdd69`已由独立Terra复审为`P0=0/P1=0`：predictor package只含全局扁平query ID，matrix/runtime/artifact/scorer均递归拒绝旧v1和任意嵌套的按类query字段；24行、144个唯一prediction和192个arm闭合。CLI随后由`cf9b3d65`完成foreign-result绑定拒绝及prepare receipt到completion/score的完整权威链。58项验证仍是功能/协议证据，不是real-checkpoint smoke或性能证据。

## 5.发布前最小信息

以下字段在唯一runner接管前填入，不作为当前研发阶段的额外gate：

|字段|冻结值|
|---|---|
|Git commit/文件SHA|已独立复审的source baseline`568ceafaf58e91977d14198be7a9cce69aba8aea`；runtime archive SHA256=`350f56641513e61d077daff562c4152806dfed65c49fff466ab15650be698de7`，6,528,947B|
|本地验证命令与结果|R4九组聚焦测试`58 passed`；三入口`py_compile`通过；`git diff --check`通过|
|N607工作目录|`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r4_fa_rdce3_cer_plr160_proxy24_20260804_r1/source`；run root创建前必须`ABSENT`|
|Conda/Python环境|`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；runner在preflight复核Python/torch/CUDA|
|prepare/predict/score精确命令|见下方§5.1；仅package/prepare/truth文件SHA由同次prepare后`sha256sum`机械填入|
|GPU分配|物理GPU0；`CUDA_VISIBLE_DEVICES=0`、进程内`cuda:0`，以发布时preflight为空闲为条件|
|run root/log/prediction/score路径|`<root>/{prepare,output,logs}`；主日志`<root>/logs/run.out`，PID`<root>/logs/main.pid`，score=`<root>/output/score.json`|
|PID/CWD/cmdline证据|唯一PID=`1343721`；启动时CWD=`<root>/source`且cmdline绑定本run/package/checkpoint/receipt；异常后PID与run树均退出|
|预期artifact|prepare package、truth sidecar、prediction、manifest、resource receipt、score、complete log|

技术停止仅允许协议/安全违规、错误checkout/hash、覆盖风险、缺prediction闭合或至少两个不同row出现同一确定性异常指纹；不得按运行中性能停止。

本地真实性能输入准备：已确认D105 checkpoint为`E:\type10-7\automation_reports\CV-SincNet\d105_feature_tap_real_checkpoint_smoke_20260731\input\best_joint_safe_ssdg.pth`，SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。规定的只读N607 preflight已通过，8张RTX 3090当时均空闲；随后只读取回既有received-IQ到本报告目录`input/d106_ls_received_iq.npz`，大小1,509,068B，远端与本地SHA256同为`e32708214eaedaf39af532c572e16045f173422d63110e4022778f3ad0252ede`。同时取回receipt，SHA256=`a18bd5d610c9874bd0d6b50d34e845d85229d5892453bce9ff5bfeaa8ee82d59`；其声明`protocol_schema=p2_min_v1`、`row_count=588`、`formal_query_access=false`、`clean_iq_access=false`。SCP结束后本地`ssh.exe=0`且到N607/bridge的TCP22连接为0。

正式Phase1资产已由strict tap SHA256=`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`构建到`input/fa_assets_v1/`。方法锁使用冻结目标文档原始字节SHA256=`35530428ecfe77982043a3b29f3f2275c5bfb66fa1da523f64c8c01030bc7311`；manifest SHA256=`0dedb5ae2c6052820f44c1d9d986ff29222ac16765618cd470529671cbcb6fd8`。共12个资产，每个只含5类×84条的聚合统计、Phase1 fit count=420；逐资产wire SHA和checkpoint SHA复算全部一致，且`phase1_member_ids_written=false`、`phase1_per_row_features_written=false`。

新的候选无关qKNN锁不复用任何旧候选/row SHA：Phase1 LODO authority SHA256=`b49cdc9f99094372412fd76d647cec58495a486eeb978fbd72f10e85f0f0e26a`，INT8 margin audit SHA256=`024a5024c06d710fbf4ddfee5326aacc89dc2ab2c74d3a9b866af09957efd9e3`，K1/K5锁文件SHA256=`13b5496c3580b16a6660dee4fc8cd0f41874a41144b357c7c08c99b4d80e91fc`。数值只机械继承D106的`ν=3,d_eff=12,h0=0.35,temperature=0.85`，K1/K5仅`active_k`不同；Phase1-only量化一致性为K1/K5 top1各588/588，未计算准确率、未读取Stage2 truth、未据此改参。当前direct qKNN score不实际应用temperature，该字段仅保留在继承锁中。

正式capsule v2 SHA256=`5b30f6dda514797beb984b5ed01995cfc64b8cb5d1a7367da241a468b6ab8272`，一次性validator receipt SHA256=`d0ac04930dd02a7d4c2dfe98c41d8933301e89095cc3cc5afad028f3bc499c64`，`capsule_id=9df82b4af19898748bc5a27c039cd7e04d1f7c53fc3aa4e082c6308a3eb32a26`，`split_id=a5ccbba48980228a6dfb42b86116262a33184877d5ed4cafe11d406b74d05d96`。固定salt为`cvs.stage2.next_r4.proxy24.opaque_physical_id_order.v1`、metadata seed=0，均在读取NEXT-R4性能前冻结；每receiver×class 14条中K5前5、K1为其首条、其余9条为K1/K5共享query。v1 metadata保留了安全扫描禁止的重复`global_reassignment=false`字段，因此真实prepare在任何forward前失败并完整保留；v2仅删除该重复字段，validator receipt仍明确保存访问为false。

真实D105 checkpoint no-query smoke已在CPU上用两条真实received-IQ完成，两次forward的160维R0逐字节一致，`truth_loaded=false`、`query_truth_access=false`。真实prepare v2随后成功，输出`local_prepare_v2/`：predictor package SHA256=`250bf38c4ff26be960a2f41b59af3cbf9b2cf78b4e5cf4b863d49a40fe463c5b`，truth sidecar SHA256=`48c7291271fb79da1fa9236acc5282c4aeb20b062ce591a44af733825237731b`；递归扫描未发现按类query、Phase1成员ID、truth、role、quota或global reassignment字段，24行闭合且query fit/update/selection均为0。这些仍是发布就绪证据，不是性能结果。

### 5.1 N607冻结执行

- runtime archive：本地`release/next_r4_runtime_568ceafa.tar.gz`，远端`<root>/input/next_r4_runtime_568ceafa.tar.gz`。
- 远端FA manifest：本地`release/remote_fa_asset_manifest.json`，SHA256=`dd602359d9ff28aaf9084a09c2d2e4fc9d6daf3383bc7268492b2eb58ede196d`；其中12个`asset_path`固定指向`<root>/input/fa_assets_v1/`。
- received-IQ沿用只读路径`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output/selected_ls_iq/d106_ls_received_iq.npz`；checkpoint沿用只读路径`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`。
- prepare：在`<root>/source`用冻结Python执行`code/scripts/run_next_r4_proxy24.py prepare`，参数为上述received-IQ/SHA、`<root>/input/next_r4_capsule_metadata_v2.json`/SHA=`5b30f6dda514797beb984b5ed01995cfc64b8cb5d1a7367da241a468b6ab8272`、远端FA manifest/SHA及checkpoint SHA，输出`<root>/prepare`。
- predict：同一CWD与Python，`CUDA_VISIBLE_DEVICES=0`，执行`... predict --run-id next_r4_fa_rdce3_cer_plr160_proxy24_20260804_r1 --run-root <root>/output --received-iq <received> --received-iq-sha256 e327... --package <root>/prepare/predictor_package.json --package-sha256 <同次文件SHA> --fa-asset-manifest <root>/input/remote_fa_asset_manifest.json --fa-asset-manifest-sha256 dd602... --checkpoint <checkpoint> --checkpoint-sha256 2699... --prepare-receipt <root>/prepare/prepare_receipt.json --prepare-receipt-sha256 <同次文件SHA> --device cuda:0`。
- score：仅在prediction/completion闭合后执行`... score --run-root <root>/output --truth <root>/prepare/truth.json --truth-sha256 <同次文件SHA> --prepare-receipt <root>/prepare/prepare_receipt.json --prepare-receipt-sha256 <同次文件SHA> --output <root>/output/score.json`。

predict以`nohup`独立启动；runner必须在启动后核对PID/CWD/cmdline、GPU映射、日志增长、首row/首wave计数，并用短SSH连接监控。fresh-run retry未授权。

### 5.2 r1正式执行结果

|阶段|状态|证据|
|---|---|---|
|preflight/落地|`PASSED / LANDED`|直连N607普通账号；root创建前`ABSENT`；GPU0为0%/1MiB且compute-apps为空；archive、capsule、validator、qKNN receipts/locks、remote manifest及12个FA wire远端SHA全部匹配|
|compile/import|`PASSED`|Python3.10.19、torch2.1.0+cu121、CUDA可用且8卡可见；R4 import、`py_compile`和CLI help通过|
|prepare|`PASSED`|exit=0；package SHA=`af9201c90ac5cb76c856bea9a5e350aacfbf73254567d76a62ecbe0f31e6df2b`，prepare receipt SHA=`13295bb712a4d47c33b614f2ebfde9ac92eb29d894b030d6dcf2afc9288fcf6e`，truth SHA=`48c7291271fb79da1fa9236acc5282c4aeb20b062ce591a44af733825237731b`；24 rows，query fit/update/selection=0，truth未进predictor|
|predict|`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`|唯一PID在首logical row、0 prediction时自然退出；异常`TIE_UNRESOLVED: direct qKNN final float32 top tie`；无prediction/manifest/completion，未score、未重启|
|资源与取回|`CLOSED`|GPU compute-apps为空、8卡0%/1MiB；partial已取回`E:\type10-7\automation_reports\CV-SincNet\next_r4_fa_rdce3_cer_plr160_proxy24_20260804_r1\retrieved\`并复算SHA；本地`ssh.exe=0`，N607/bridge TCP22连接=0|

`run.err`共23行、SHA256=`c911ecf0ebb4bb7905b9cc785cca9d4ceae4de1cf1ed724f2b08b98536069238`；同一traceback内异常文本出现4次，不表示4个不同row。output仅有`plan.json`、`preregistration.json`和`smoke.json`；其中24/144/192为预登记目标，实际完成数为0。r1不产生域适应前/后或新类注册前/后的任何性能结论。

## 6.结果表

当前没有真实性能结果，不填写估计值或单元测试代理值。

|receiver|held-class|K|状态码|中文状态|arm|old BA|old-floor|seen-new|H|all-floor|总正确数|判定|
|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|
|—|—|—|—|—|—|—|—|—|—|—|—|`NO_PERFORMANCE_RESULT`|

## 7.完成后检查与裁决

完整矩阵返回后检查same-row四态、per-class old accuracy、forgetting、seen-new、H、all-floor、总正确数和receiver×K聚合。r1在0 prediction时因确定性top tie退出，当前先研究只作用于精确tie、无truth/role/quota/阈值的确定性决策规则；若科学复核通过，使用新run ID发布，不覆盖或重启r1。真实性能弱或为负时才按冻结阈值关闭相应组件/路线；不扩大矩阵或盲调参数。
