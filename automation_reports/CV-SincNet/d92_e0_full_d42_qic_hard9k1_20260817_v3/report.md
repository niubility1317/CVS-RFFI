# D92 E0 FULL QIC Hard9+K1实验报告

状态：`ANALYZED / REJECT_ROUTE / NO_TARGET125`

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

## 6.N607运行证据与收尾

本节由唯一runner Luna/max在2026-08-17完成。规则栈、运行边界和v3资源语义均已重读；v1/v2及其他run未触碰。

|阶段|证据|
|---|---|
|RULES_READ|live`AGENTS.md`、`项目.md`、`using-git-bash-on-windows`及完整failure catalog、`cv-sincnet-n607-automation`均已读取。|
|PRECHECK|direct普通`N607`只读预检通过；8卡可用；v3远端archive/driver/source/output/logs/driver.out/err、同run进程及本地retrieval均满足初始ABSENT。|
|SYNC|archive按冻结SHA`61d424bda548ec04b49a1763b83080a0dff04c11b94d6a707e72d98cd00bddbe`、driver按冻结SHA`737a638b978430713599fbf75135c9802d408eb4e101948130762dfc743c6658`顺序落地；远端size、archive 50 members、embedded config SHA、`bash -n`、Python/CUDA核验通过。|
|COMMAND/LAUNCH|唯一冻结detached command执行1次；启动后driver记录shard 0–7分别绑定GPU 0–7，driver.err为0B。未重试、未重启、未覆盖。|
|SMOKE|`D92_QIC_HARD9_K1_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS`；`truth_open=false`；七项query访问标志均false；truth仅在immutable prediction之后加入；prediction/scorer进程隔离；before/after closure的commit、fit和prediction哈希字段齐全。|
|健康停止判定|未发现错误source、输出覆盖、query泄漏、缺失prediction closure、launcher-wide确定性异常或系统性零预测；resource wall/ratio/absolute peak字段按v3策略仅记录，不作为技术停机条件。|
|收尾|运行结束后同run PID=0、8卡compute进程为空；远端产物保留；每个SSH/SCP完成后本地无活动SSH/SCP及TCP22连接。|

### 6.1技术artifact计数

|artifact|本地retrieval证据|
|---|---:|
|source目录文件|95|
|output目录文件|220|
|logs目录文件|24|
|正式before/after prediction closure|20/20；另有smoke 1/1，总`prediction_artifact.npz`=22|
|COMMIT/fit/resource/execution|22/22/22/22；正式部分各20|
|job receipt|10|
|`diag_cosine_score.json`|10（仅计数，未打开内容）|
|`score_binding.json`|10（仅计数，未打开内容）|
|shard summary/event|8/8（`shard_0`–`shard_7`）|
|driver|3个文件：script 6692B、out 2714B、err 0B|
|truth sidecar|10个；按manifest路径复制到本地并逐个SHA256匹配，未打开或解析内容|

完整本地取回根目录为`E:/type10-7/local_artifacts/d92_e0_full_d42_qic_hard9k1_20260817_v3`，其中`input/`保留冻结archive与launch，`source/`、`output/`、`logs/`、`driver/`和`truth_sidecars/<outer_key>/`分别保存对应证据。远端文件未删除。

本runner未运行analyzer，未读取score内容或任何accuracy、H、BA、floor、forgetting、confusion字段；当前不构成performance result。下一步由primary在本地按冻结truth-last analyzer执行分析并更新本报告。

## 7.truth-last分析与唯一裁决

主代理在完整取回后运行冻结analyzer。最初版本沿用了远端绝对`score_binding`路径、未识别runner扁平取回的truth sidecar目录，并误要求旧CSOAS/CCOC收据字段；这些问题只产生空指标的`REJECTED_EVIDENCE_CLOSURE`诊断包，不属于性能结果。修复commit为`48e5723a`和`49cb7f6a`，10项聚焦测试、`py_compile`和`diff-check`通过。有效分析输出位于`local_artifacts/.../analysis_v5/`，其artifact闭包为10个paired rows、60个逐旧类rows、30个scene rows，其中9个performance outer对应27个scene；K1仅作为1个liveness row。

### 7.1九个performance outer的同排均值

|指标|E0|QIC|QIC-E0|严格方向|
|---|---:|---:|---:|---|
|`H_old_new`|72.2166%|38.8552%|-33.3614pp|失败|
|old balanced accuracy|73.6420%|47.6235%|-26.0185pp|失败|
|`c_old` accuracy|73.6420%|47.6235%|-26.0185pp|失败|
|old-class floor|42.7778%|19.4444%|-23.3333pp|失败|
|seen-new accuracy|71.0000%|36.3704%|-34.6296pp|失败|
|average forgetting|13.2716%|39.2901%|+26.0185pp|失败；应下降|
|new→old|14.8241%|33.4074%|+18.5833pp|失败；应下降|
|old→new|15.9877%|31.6358%|+15.6481pp|失败；应下降|

QIC没有在均值层面改善任何一项冻结指标。receiver、scene、K/new slice与逐旧类稳定性门也全部失败；这不是边缘tie，而是量化自洽截距把注册后判别边界整体推离E0。

### 7.2逐outer结果

下表八项差值单位均为百分点；前五项应为正，后三项应为负。

|outer|K/new|ΔH|Δold BA|Δc_old|Δfloor|Δnew|Δforget|Δnew→old|Δold→new|peak MiB|wall P90 ms|ratio P90|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`rx_7_7__seed_713104__k_5__new_20`|K5/new20|-27.48|-18.89|-18.89|-26.67|-31.08|+18.89|+35.92|-6.94|2.27|107.49|1.124|
|`rx_7_7__seed_713103__k_10__new_5`|K10/new5|-37.26|-23.89|-23.89|-18.33|-46.67|+23.89|+38.33|+8.89|1.36|60.32|0.959|
|`rx_8_8__seed_713103__k_5__new_20`|K5/new20|-32.91|-35.28|-35.28|-15.00|-30.33|+35.28|-8.00|+38.06|2.84|117.51|1.133|
|`rx_8_8__seed_713103__k_10__new_5`|K10/new5|-40.33|-48.61|-48.61|-43.33|-23.33|+48.61|-5.67|+46.94|1.36|61.58|0.919|
|`rx_8_8__seed_713106__k_5__new_20`|K5/new20|-33.09|-32.78|-32.78|-28.33|-33.42|+32.78|+1.00|+34.17|2.27|95.99|0.893|
|`rx_7_14__seed_713104__k_10__new_10`|K10/new10|-23.61|-20.56|-20.56|-21.67|-26.33|+20.56|+4.83|+23.61|2.13|76.34|1.098|
|`rx_3_19__seed_713102__k_10__new_5`|K10/new5|-14.85|-11.39|-11.39|-26.67|-16.67|+11.39|-4.33|+16.39|1.96|72.67|1.269|
|`rx_7_7__seed_713105__k_10__new_20`|K10/new20|-27.01|-13.33|-13.33|-6.67|-34.50|+13.33|+34.50|-12.50|2.82|156.43|1.572|
|`rx_7_7__seed_713104__k_10__new_5`|K10/new5|-63.71|-29.44|-29.44|-23.33|-69.33|+29.44|+70.67|-7.78|1.36|73.42|1.147|

### 7.3资源与实时推理

|门|结果|证据|
|---|---|---|
|query MAC/state|PASS|30/30 scene均与E0精确相等；无新增实时推理计算或持久状态|
|registration wall P90|PASS|115.503ms≤150ms|
|registration wall ratio P90|PASS hard、FAIL target|1.2687≤1.50，但高于1.25目标|
|candidate incremental peak|FAIL hard|最大2,977,792B（2.84MiB）>1MiB|
|resource integrity|PASS|scene-keyed收据和E0同排绑定闭合|

用户允许把注册资源空间放宽到原目标的两倍，但实时推理必须保持低开销。QIC满足query exact门，也满足wall硬门；它仍超过放宽后的1MiB注册峰值门。即使忽略资源失败，八项性能全部反向，裁决不会改变。

### 7.4结论

唯一裁决为`REJECT_ROUTE`。不运行Target125，不对QIC增加任务偏置、强度扫描或后验调参，也不把K1 liveness或G0技术通过解释为性能收益。该结果否证了“仅按量化后自能量闭合全类截距即可改善new→old且保持旧类”的假设：在真实同排Hard9中，它同时损害旧类、新类、floor、遗忘和两向混淆。
