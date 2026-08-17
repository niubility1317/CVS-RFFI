# D92 E0 FULL QIC Hard9+K1实验报告

状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`

## 1.目标与冻结方法

|字段|值|
|---|---|
|run ID|`d92_e0_full_d42_qic_hard9k1_20260817_v2`|
|runtime commit|`6969796f7af6ceb5624c989fbe1f4405fcaf4a83`|
|reference|`E0_FULL_ONLY`|
|candidate|`E0_FULL_D42_QUANTIZATION_INTERCEPT_CLOSURE`|
|selection SHA|`b07a7fdc43fa08934c4cbdccea01cb2e36bdee09f2148a1c3e4acc4495dff975`|
|协议|`p2_min_v1`；复用`VALIDATED_ONCE`sealed输入|
|claim scope|`DEVELOPMENT_ONLY_DISJOINT_FROM_G0_HARD_SCREEN`|
|fresh-run retry|`false`|

QIC只在一次E0 FULL与既有D42量化后，用同一support均值闭式重算全类FP16截距；不改变系数、scale、log-diagonal、query MAC/state格式，不增加FULL/BLOCK/LOO/Fisher/scan。K≤2保持`K1_K2_EXACT_D92_FULL_ALIAS`。

技术先决条件已经由独立G0 v2实测通过：三scene均active/nonfallback、bit change=11、残差严格下降；wall为63.808–64.912ms，peak为8–92KiB；query MAC=3168、state=8583B，与E0完全一致。G0未读取性能。

## 2.冻结矩阵与裁决

|项目|冻结值|
|---|---|
|矩阵|9个performance outer+1个K1 liveness outer|
|scene|`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`|
|总量|10 jobs、30 scene-arm、8 shards|
|G0排除|`rx_7_7__seed_713106__k_10__new_5`|
|smoke|`rx_7_7__seed_713104__k_5__new_20`，真实checkpoint、truth-free|
|资源硬门|registration wall P90≤150ms、candidate/E0≤1.50、candidate incremental peak≤1MiB|
|实时推理|query MAC/state与E0精确相等；七项query访问全false|

完整artifact取回后才运行独立truth-last analyzer。performance outer按同一row比较八项：`H_old_new`、old BA、`c_old`、old floor、seen-new均严格上升；forgetting、new→old、old→new均严格下降。任一tie或反向即`REJECT_ROUTE`。K1只作liveness，不进入性能均值。

报告和分析使用四态`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`；REG0中的新类与old/new调和指标记为`N/A`。

## 3.本地实现与验证

QIC matrix/runner科学实现commit为`d4a986b6`，truth-last analyzer commit为`ddb715cb`；v2运行身份与归档字节闭包commit为`6969796f`。matrix/runner聚焦39项通过，v2归档独立锁定文件检查1项通过；生产入口`py_compile`、runner/analyzer`--help`、JSON和`git diff --check`均通过。

v1在prepare阶段因归档运行文件字节与冻结SHA不一致而技术停止，未进入smoke且没有性能结果。v2仅启用全新不可覆盖路径，并按冻结配置逐项封装17个运行文件的精确工作树字节；方法、矩阵、阈值与裁决均未改变。

release：

|文件|大小|SHA256|
|---|---:|---|
|`runtime/d92_qic_hard9_k1_source_6969796f_20260817_v2.tar.gz`|320955B|`46cd187aeb902eaa5adc8ab79777e6f2a9b94bf0aa85009e85cadd65593175e1`|
|`launch.sh`|6692B|`57c787f0f06968bf565d34d02b526af50bac2d3dd3df09005e758c817313c025`|

archive共50成员，其中source manifest列49项；路径安全、source manifest、extracted runner/analyzer help、config JSON和launch`bash -n`均通过。release不包含数据、checkpoint、truth sidecar或测试。

## 4.N607路径与唯一命令

```text
archive=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_qic_hard9_k1_source_6969796f_20260817_v2.tar.gz
launch=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_qic_hard9_k1_driver_d92_e0_full_d42_qic_hard9k1_20260817_v2.sh
source=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_qic_hard9_k1_source_6969796f_20260817_v2
output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_qic_hard9_k1_20260817_v2
logs=/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_d42_qic_hard9k1_20260817_v2
local_retrieval=E:/type10-7/local_artifacts/d92_e0_full_d42_qic_hard9k1_20260817_v2
python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
gpu=0,1,2,3,4,5,6,7
```

唯一detached command：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_qic_hard9_k1_driver_d92_e0_full_d42_qic_hard9k1_20260817_v2.sh >./d92_qic_hard9_k1_driver_d92_e0_full_d42_qic_hard9k1_20260817_v2.out 2>./d92_qic_hard9_k1_driver_d92_e0_full_d42_qic_hard9k1_20260817_v2.err </dev/null &
```

launch顺序固定为prepare→truth-free smoke→8 shards；不自动运行analyzer。预期产物包括`matrix_manifest.json`、smoke receipt、20组正式before/after prediction闭包、10个job receipt、10个score、8个shard summary和日志。

## 5.健康停止与证据边界

只允许因协议/安全错误、输出覆盖、错误source、query泄漏、缺失prediction closure、确定性异常或系统性零预测停止；禁止按accuracy、H、BA、floor或forgetting停止。若技术停止，不重试同run，保留并取回partial artifacts并标记`NO_PERFORMANCE_RESULT`。

sole runner不得解释性能、不得调参、不得改矩阵。只有技术健康且artifact完整后，主代理才在本地运行冻结analyzer并更新本报告。

## 6.N607运行证据与收尾

|字段|结果|
|---|---|
|RULES_READ|完成；live`AGENTS.md`、`项目.md`、Git Bash skill及完整failure catalog、`cv-sincnet-n607-automation`skill均已重新读取|
|release identity|release commit=`93111be81a218a46fe71ee58cb76fe6df86e9f72`；runtime identity=`6969796f7af6ceb5624c989fbe1f4405fcaf4a83`；archive与launch内容/SHA未改变|
|direct preflight|通过；ordinary`N607`、项目根目录/Python可见、8张RTX3090可见；v2同run远端路径与进程均ABSENT|
|SYNC|archive与driver按冻结顺序SCP成功；archive=320955B，SHA=`46cd187aeb902eaa5adc8ab79777e6f2a9b94bf0aa85009e85cadd65593175e1`，50 members；driver=6692B，SHA=`57c787f0f06968bf565d34d02b526af50bac2d3dd3df09005e758c817313c025`，`bash -n`通过；archive config SHA=`6a8fc7942a2820b5779e5cf74e1222df22d800791864ce312db5708d7a47e8a8`与冻结manifest一致|
|唯一launch|已执行1次，使用本报告第4节冻结detached command；未重试、未重启、未覆盖|
|prepare|通过；生成`matrix_manifest.json`，状态`QIC_HARD9_K1_MATRIX_PREPARED`，job_count=10、scene_arm_count=30|
|smoke|失败；`smoke.err`：`D92 QIC Hard9+K1 failed: candidate fit is not finite`；未生成`smoke_receipt.json`，固定PASS marker未出现，8 shards未启动|
|smoke技术partial|保留before/after各一组技术闭包：`COMMIT.json`、`execution_receipt.json`、`fit_audit.json`、`resource_audit.json`和`prediction_artifact.npz`；未读取其性能内容|
|正式技术计数|formal prediction before/after=0/0；formal COMMIT/fit/resource/execution=0/0/0/0；job_receipt=0；score=0；shard_summary=0|
|query/truth边界|未产生truth sidecar、score或smoke receipt；未读取accuracy、H、BA、floor、forgetting或confusion；未运行analyzer|
|取回|`E:/type10-7/local_artifacts/d92_e0_full_d42_qic_hard9k1_20260817_v2`：94个source文件、13个output文件、8个技术日志、driver out/err及输入archive/launch；truth sidecar=0|
|清理|远端同run PID=0、GPU compute为空；source/output/log/driver保留在远端；每次SSH/SCP均已断开，本地无N607 TCP22 ESTABLISHED|

本次运行在truth-free smoke阶段发生launcher-wide确定性技术失败，按冻结规则标记`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，`fresh_run_retry=false`。后续应由主代理在本地定位candidate fit非有限问题，完成窄验证和独立P0/P1复核后，以新的不可覆盖run ID重新注册；本runner不修复、不重试本run。
