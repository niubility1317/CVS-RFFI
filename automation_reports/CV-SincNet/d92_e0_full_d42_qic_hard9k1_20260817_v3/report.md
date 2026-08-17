# D92 E0 FULL QIC Hard9+K1实验报告

状态：`LOCAL_VERIFIED / READY_FOR_N607_HANDOFF / NO_PERFORMANCE_RESULT`

## 1.目标与冻结方法

|字段|值|
|---|---|
|run ID|`d92_e0_full_d42_qic_hard9k1_20260817_v3`|
|runtime commit|`fa75cf8e4cb4235e09ef3d77b3f6091e4ef31663`|
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
|执行策略|资源超限不再伪装成启动技术故障；完整记录后由truth-last analyzer执行资源门并裁决|

完整artifact取回后才运行独立truth-last analyzer。performance outer按同一row比较八项：`H_old_new`、old BA、`c_old`、old floor、seen-new均严格上升；forgetting、new→old、old→new均严格下降。任一tie或反向即`REJECT_ROUTE`。K1只作liveness，不进入性能均值。

报告和分析使用四态`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`；REG0中的新类与old/new调和指标记为`N/A`。

## 3.本地实现与验证

QIC matrix/runner科学实现commit为`d4a986b6`，truth-last analyzer commit为`ddb715cb`；真实收据修复commit为`7aba0926`，v3运行身份commit为`fa75cf8e`。matrix/runner聚焦39项通过，真实v2 smoke fit-audit在修复后可闭合并准确记录资源门失败；生产入口`py_compile`、runner/analyzer`--help`、JSON和`git diff --check`均通过。

v1在prepare阶段因归档运行文件字节与冻结SHA不一致而技术停止。v2在truth-free smoke暴露runner误要求6个非QIC字段，随后又发现smoke outer的E0本身峰值已超过1MiB。v3删除错误字段要求，并把资源超限保留为analyzer裁决而非启动故障；方法、矩阵、资源阈值及最终裁决均未改变。

release：

|文件|大小|SHA256|
|---|---:|---|
|`runtime/d92_qic_hard9_k1_source_fa75cf8e_20260817_v3.tar.gz`|320797B|`61d424bda548ec04b49a1763b83080a0dff04c11b94d6a707e72d98cd00bddbe`|
|`launch.sh`|6692B|`737a638b978430713599fbf75135c9802d408eb4e101948130762dfc743c6658`|

archive共50成员，其中source manifest列49项；路径安全、source manifest、extracted runner/analyzer help、config JSON和launch`bash -n`均通过。release不包含数据、checkpoint、truth sidecar或测试。

## 4.N607路径与唯一命令

```text
archive=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_qic_hard9_k1_source_fa75cf8e_20260817_v3.tar.gz
launch=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_qic_hard9_k1_driver_d92_e0_full_d42_qic_hard9k1_20260817_v3.sh
source=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_qic_hard9_k1_source_fa75cf8e_20260817_v3
output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_qic_hard9_k1_20260817_v3
logs=/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_d42_qic_hard9k1_20260817_v3
local_retrieval=E:/type10-7/local_artifacts/d92_e0_full_d42_qic_hard9k1_20260817_v3
python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
gpu=0,1,2,3,4,5,6,7
```

唯一detached command：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_qic_hard9_k1_driver_d92_e0_full_d42_qic_hard9k1_20260817_v3.sh >./d92_qic_hard9_k1_driver_d92_e0_full_d42_qic_hard9k1_20260817_v3.out 2>./d92_qic_hard9_k1_driver_d92_e0_full_d42_qic_hard9k1_20260817_v3.err </dev/null &
```

launch顺序固定为prepare→truth-free smoke→8 shards；不自动运行analyzer。预期产物包括`matrix_manifest.json`、smoke receipt、20组正式before/after prediction闭包、10个job receipt、10个score、8个shard summary和日志。

## 5.健康停止与证据边界

只允许因协议/安全错误、输出覆盖、错误source、query泄漏、缺失prediction closure、确定性异常或系统性零预测停止；禁止按accuracy、H、BA、floor或forgetting停止。若技术停止，不重试同run，保留并取回partial artifacts并标记`NO_PERFORMANCE_RESULT`。

sole runner不得解释性能、不得调参、不得改矩阵。只有技术健康且artifact完整后，主代理才在本地运行冻结analyzer并更新本报告。
