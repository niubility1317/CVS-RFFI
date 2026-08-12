# D92 E0 FULL CSOAS Hard9+K1 v2实验报告

状态：`ANALYZED / REJECT_ROUTE / TARGET125_NOT_AUTHORIZED`

## 1.目标与冻结范围

|字段|值|
|---|---|
|run ID|`d92_e0_full_csoas_hard9k1_20260812_v2`|
|科学commit|`b8ebd4f4522fcc3e9e6b7dd18d722c329021f181`|
|机械实现/P1修复|`ac811820`/`7a434080`|
|v1递归修复|`1fab89eb15d66970f4725e897da585f477a791bd`|
|候选|`E0_FULL_CSOAS`；candidate=`d92_e0_full_csoas`；mode=`csoas_full`|
|矩阵|与G0不重叠的9个最难performance outer+1个K1 liveness；3 scenes；10 jobs；8 shards|
|声明|development-only Hard9；完整artifact和冻结analyzer前无性能结论|
|retry|`false`；v2唯一detached launch；不得resume/覆盖v1|

v1已固定为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`：prepare完成后，smoke在预测前因runner verifier动态自引用触发`RecursionError`，8 shards未启动。v2唯一变化是冻结原始base verifier引用；方法、矩阵、阈值、数据、scorer和analyzer均不变。

## 2.本地修复与发布门

- TDD RED：新增回归测试在父实现上复现同一`RecursionError`。
- GREEN：runner focused `12 passed`；`py_compile`与`git diff --check`通过。
- 独立复审：`P0=0/P1=0，APPROVE`；确认context内调用冻结原始helper且`finally`恢复base属性。
- selection SHA256=`a851590bc6d502ddbe326a936096d95f5bb382e4cb235b61b0121d98c0b87b5d`。
- 数据继续复用`p2_min_v1/VALIDATED_ONCE`，不重验；query fit/update/selection/truth/role/quota/global reassignment必须全false。

## 3.性能与资源裁决

9个performance outer必须逐row同E0比较，K1只做liveness。八项总体均值全部严格优于E0才允许晋级：`H_old_new`、old balanced accuracy、`c_old_acc`、old floor、seen-new accuracy升高；average forgetting、new→old、old→new降低。任何tie/反向、K>2 fallback/codec retry、稳定性或资源硬门失败均`REJECT_ROUTE`，不跑target125。

大胆目标依次为`+1.0pp,+1.5pp,+1.0pp,+4.0pp,+0.5pp,-1.5pp,-0.5pp,-0.5pp`。资源硬门：wall P90≤150ms、paired wall ratio median≤1.50、peak delta≤512KiB、query MAC/state逐row等同E0；120ms/1.25是目标门。

## 4.交付与远端路径

|交付|size|SHA256|
|---|---:|---|
|`d92_csoas_hard9_runtime_1fab89eb.tar.gz`|6,184,420|`de74fe49d8d24432898e44fddfc3c8a9f2f2444b2d70421e7d69d786c9a25d78`|
|`stage2_d92_csoas_hard10_v1.json`|6,293|`6fcd29dfab77c99745df336f32425dfdc0a0a0a99469c92766a4751fa92e427e`|
|`launch.sh`|3,717|`c8e87a3d75e2d6ac76c50f21bc3fb0826d8ed5967522d605be05740f92fe7bed`|

archive共1466 members，runner SHA256=`623c7138e7e70bde6e4ef49bfd0dcd6f66d1b7203e5ef09e876bc458f1d8c08c`，必需入口齐全，无绝对/`..`/`code/code`路径；launch以LF内容通过`bash -n`。

|用途|远端路径|
|---|---|
|source|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_hard9_source_1fab89eb_20260812_v2`|
|output|`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_csoas_hard9k1_20260812_v2`|
|logs|`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_csoas_hard9k1_20260812_v2`|
|context|`/home/szu2070436088/2510044040/CV-SincNet/runs/d131_d92_lite160_qtie_target125_20260804_r3/prepared/target125_context.json`|

唯一命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_csoas_hard9_source_1fab89eb_20260812_v2 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

环境=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；8 shards映射GPU0–7。prepare和真实K5/new20 truth-free smoke通过后才允许shards。

## 5.健康门、artifact与后续

只按wrong hash/CWD、覆盖风险、协议/安全违规、launcher异常或两个distinct outer同一prediction前指纹停止；不得按任何性能值停止。健康完成期望：10 job receipts；正式before/after prediction/COMMIT/fit/resource/execution各20；scores 10；summaries 8；完整取回source/output/logs及10 truth sidecars。

primary只在artifact闭环后运行冻结analyzer。若八项严格Pareto及稳定/资源门全部通过，立即发布完整target125；否则记录失败指标并停止该路线。

## 6.运行与分析结果

### 6.1技术闭包

v2唯一launch完成。prepare生成10 jobs/30 scene rows，manifest SHA=`be0dd760dce70c20db4f10bca4a30bc5058450c55cf9fda907f36d84f5ffe808`；K5/new20真实checkpoint truth-free smoke通过；8/8 shard summaries为PASS，failed job=0。正式产物为10 job receipts、before/after prediction/COMMIT/fit/resource/execution各20、score 10；另有smoke before/after各2份结构产物。stderr非空=0，systemic stop=0，query禁用字段全false。

完整取回根为`E:\type10-7\local_artifacts\d92_e0_full_csoas_hard9k1_20260812_v2`。远端=本地树：source 1438 files/73,811,362B/SHA `8c76ed6036388a1df04ee98890f41187bc5bb408ff823a1aad4cbc1bbbbac464`；output 190/16,527,849B/`c7edae513b9e52278ff55f949959b4887d4f9f7bc3f6974f85ee2fbd14473e93`；logs 22/9,554B/`3cc185ca127665c802a2510bd977b46d0194f672fc408226cd4ccd2e361171cc`。10份truth sidecar逐SHA取回；runner未读取性能，最终run/GPU/SSH/TCP22均清零。

### 6.2同排九个最难outer结果

下表每格为`CSOAS/E0_FULL_ONLY（Δpp）`；forgetting与两向混淆的负Δ为改善。

|outer|K/new|H|old BA|old acc|old floor|seen-new|forgetting|new→old|old→new|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`rx_7_7__seed_713104__k_5__new_20`|K5/20|70.27/69.78（+0.48）|81.94/75.56（+6.39）|81.94/75.56（+6.39）|68.33/41.67（+26.67）|61.50/64.83（-3.33）|9.44/15.83（-6.39）|14.08/13.67（+0.42）|13.61/17.22（-3.61）|
|`rx_7_7__seed_713103__k_10__new_5`|K10/5|77.85/82.69（-4.84）|86.67/81.11（+5.56）|86.67/81.11（+5.56）|76.67/53.33（+23.33）|70.67/84.33（-13.67）|6.67/12.22（-5.56）|26.00/13.67（+12.33）|8.89/12.22（-3.33）|
|`rx_8_8__seed_713103__k_5__new_20`|K5/20|67.63/66.38（+1.25）|72.78/66.94（+5.83）|72.78/66.94（+5.83）|31.67/28.33（+3.33）|63.17/65.83（-2.67）|11.11/16.94（-5.83）|13.00/13.33（-0.33）|20.56/22.78（-2.22）|
|`rx_8_8__seed_713103__k_10__new_5`|K10/5|77.70/77.49（+0.21）|83.06/76.67（+6.39）|83.06/76.67（+6.39）|63.33/53.33（+10.00）|73.00/78.33（-5.33）|2.22/8.61（-6.39）|22.67/17.67（+5.00）|8.61/10.83（-2.22）|
|`rx_8_8__seed_713106__k_5__new_20`|K5/20|63.37/64.38（-1.01）|65.56/63.61（+1.94）|65.56/63.61（+1.94）|35.00/36.67（-1.67）|61.33/65.17（-3.83）|13.33/15.28（-1.94）|11.83/9.75（+2.08）|26.39/19.44（+6.94）|
|`rx_7_14__seed_713104__k_10__new_10`|K10/10|75.54/73.11（+2.43）|82.50/73.06（+9.44）|82.50/73.06（+9.44）|36.67/33.33（+3.33）|69.67/73.17（-3.50）|5.56/15.00（-9.44）|14.50/12.50（+2.00）|11.39/17.22（-5.83）|
|`rx_3_19__seed_713102__k_10__new_5`|K10/5|54.90/54.16（+0.74）|59.44/58.61（+0.83）|59.44/58.61（+0.83）|28.33/30.00（-1.67）|51.00/50.33（+0.67）|10.56/11.39（-0.83）|29.00/29.67（-0.67）|25.56/19.17（+6.39）|
|`rx_7_7__seed_713105__k_10__new_20`|K10/20|77.72/76.98（+0.73）|85.83/79.44（+6.39）|85.83/79.44（+6.39）|70.00/50.00（+20.00）|71.00/74.67（-3.67）|9.17/15.56（-6.39）|9.92/7.83（+2.08）|13.06/16.67（-3.61）|
|`rx_7_7__seed_713104__k_10__new_5`|K10/5|81.15/84.97（-3.81）|87.50/87.78（-0.28）|87.50/87.78（-0.28）|68.33/58.33（+10.00）|75.67/82.33（-6.67）|8.89/8.61（+0.28）|22.33/15.33（+7.00）|7.78/8.33（-0.56）|

### 6.3总体八项与资源

|指标|CSOAS均值|E0均值|Δpp|冻结方向|结果|
|---|---:|---:|---:|---|---|
|`H_old_new`|71.7933%|72.2166%|-0.4233|升高|FAIL|
|old balanced accuracy|78.3642%|73.6420%|+4.7222|升高|PASS|
|`c_old_acc`|78.3642%|73.6420%|+4.7222|升高|PASS|
|old floor|53.1481%|42.7778%|+10.3704|升高|PASS|
|seen-new accuracy|66.3333%|71.0000%|-4.6667|升高|FAIL|
|average forgetting|8.5494%|13.2716%|-4.7222|降低|PASS|
|new→old rate|18.1481%|14.8241%|+3.3241|降低|FAIL|
|old→new rate|15.0926%|15.9877%|-0.8951|降低|PASS|

CSOAS通过资源完整性、硬门和目标门：query MAC/state逐row等同E0；registration wall P90=`30.895252ms`，paired wall ratio median=`0.2324×`，peak delta max=`520,192B`≤512KiB；相对原D92的component-fit reduction最差=`95.8333%`。K1为精确alias，不进入性能均值。

稳定性不通过。严格方向逐row命中数为：H 6/9、old BA 8/9、old acc 8/9、floor 7/9、seen-new 1/9、forgetting 8/9、new→old 2/9、old→new 7/9。三个scene的seen-new均下降：clear -7.0278pp、low-elev -3.7778pp、rain -3.1944pp。逐旧类也发生明显分化：TX `14-7`均值+25.37pp，但TX `14-10`均值-5.93pp且最差-23.33pp，TX `20-15`均值-2.96pp且最差-21.67pp。

### 6.4解释与裁决

同排证据表明CSOAS成功把协方差几何推向旧类保留：old BA/old acc各提高4.72pp，floor提高10.37pp，forgetting降低4.72pp；代价是新类边界系统性收缩，seen-new降低4.67pp，new→old增加3.32pp，最终H下降0.42pp。该结果否证了“仅用D81 Cauchy权重和OAS重估共享FULL协方差即可同时改善旧类floor、遗忘与新类准确率”的假设。这里可以确认的是同一冻结矩阵上的关联模式；不能把退化归因到某一个公式项而不做新的受控实验。

冻结analyzer输出`REJECT_ROUTE`：artifact closure、performance closure、resource integrity、resource hard/target和compute reduction通过；strict Pareto、magnitude和stability失败。CSOAS不进入target125，不调权重、不扫参、不挑outer重跑。完整analysis位于`E:\type10-7\local_artifacts\d92_e0_full_csoas_hard9k1_20260812_v2\analysis`；`summary.json` SHA=`c55b9c197f6cc44f798b0cdcb7b499a08a734f2a49dba1f491265994d4a2efb2`，`paired_rows.csv` SHA=`fb187bd1a3cdddbe942e49223d05850a9ebb015260e117f095262d44079eb7a4`。
