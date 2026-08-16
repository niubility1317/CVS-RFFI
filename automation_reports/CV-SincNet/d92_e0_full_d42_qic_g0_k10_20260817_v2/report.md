# D92 E0 FULL QIC K10 G0预登记与发布报告

状态：`ARTIFACTS_COMPLETE / TECHNICAL_G0_PASS / NO_PERFORMANCE_RESULT`

本报告只登记一次K10/new5、三scene、truth-free技术G0发布。当前未执行N607、未启动driver、未读取truth/scorer或性能结果。

## 1.冻结身份

|字段|值|
|---|---|
|run ID|`d92_e0_full_d42_qic_g0_k10_20260817_v2`|
|runtime commit|`82ce747643af71ac3737bc0a89d18114be96f27e`|
|outer|`rx_7_7__seed_713106__k_10__new_5`|
|reference arm|`E0_FULL_ONLY`|
|candidate arm|`E0_FULL_D42_QUANTIZATION_INTERCEPT_CLOSURE`|
|scene|`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`|
|协议|`p2_min_v1`；复用已验证sealed package|
|GPU|GPU0|
|环境|远端`CVS-RFFI` Python；本地验证环境`ssr-gpu`|
|expected marker|`D92_QIC_G0_ACTIVE_QUANTIZATION_INTERCEPT_CLOSURE_RESOURCE_PASS`|
|fresh-run retry|`false`|

## 2.本地变更与验证

本次release只封装已提交runtime，不修改科学core、Slim、Query或runner门：

- `code/scripts/run_d92_qic_g0.py`
- `tests/test_run_d92_qic_g0.py`
- 本目录的`report.md`、`launch.sh`、`DELIVERY_MANIFEST.txt`和runtime archive
- `analysis/d92_qic_traceability_20260817.md`的QIC-04状态更新

已完成或待交接前复核的窄验证：

|检查|结果|
|---|---|
|focused runner|`8 passed`|
|`py_compile`|通过|
|runner`--help`|通过；无truth/scorer开关|
|`bash -n launch.sh`|通过|
|archive整体SHA/成员数/路径安全|通过：`218553`bytes、38members；整体SHA为`802a52557657d6d415992192fd546fe564495ff71354a7890807645bb071113a`|
|extracted runner`--help`|通过|
|外部镜像`cmp`|通过；四个release文件逐字节一致|

## 3.冻结sealed输入与路径

四个sealed文件沿用CCOC G0 v6的同一job和SHA：

```text
before_enrollment=e3da38668a1e6ec4053e65e669e2a1845bb43198891644d830950e4550b5cea9
before_apply=736852188c32255647b8105bc7a68d4cc92ca73615e4734d0ed5f4bdd0f04473
after_enrollment=2600a21ee9a2f95a8d17fa1f4d2263b0e04d243424e3257474502953ed6d9286
after_apply=afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a
ground_manifest=15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c
```

```text
source_root=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_qic_g0_source_82ce7476_20260817_v2
source_archive=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_qic_g0_source_82ce7476_20260817_v2.tar.gz
remote_launch=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_qic_g0_launch_82ce7476_20260817_v2.sh
reference_output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_d42_qic_g0_k10_20260817_v2/reference_e0
candidate_output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_d42_qic_g0_k10_20260817_v2/candidate_qic
validation=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_d42_qic_g0_k10_20260817_v2/g0_validation.json
logs=/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_d42_qic_g0_k10_20260817_v2
local_retrieval=E:/type10-7/local_artifacts/d92_e0_full_d42_qic_g0_k10_20260817_v2
```

## 4.archive与launch

archive从runtime commit`82ce747643af71ac3737bc0a89d18114be96f27e`的Git对象生成，复用CCOC G0 v6的package依赖成员集合，移除CCOC专属entry并加入当前QIC core/runner。未包含数据、checkpoint、truth或scorer；不建立逐成员SHA体系。

|字段|值|
|---|---|
|relative path|`runtime/d92_qic_g0_source_82ce7476_20260817_v2.tar.gz`|
|size|`218553`bytes|
|SHA256|`802a52557657d6d415992192fd546fe564495ff71354a7890807645bb071113a`|
|members|38|
|required entries|`__init__.py`、D42、QIC core、Slim、Query、QIC runner|
|tar safety|禁止绝对路径、`..`路径和`code/code`嵌套|
|launch SHA256|`fe9c708735c6416dec0958ece5cca1c3c26fd27a242e28e9211359fc9eed467e`|

launch只执行必要的archive整体SHA/size/member/path检查、四sealed SHA检查、fresh source/output/log检查、解包、入口存在、py_compile、竞争import、一次QIC driver和固定marker检查。它不包含truth/scorer参数，不执行第二次driver，不读取性能。

## 5.G0技术门

三scene均必须通过以下冻结门，任一失败即写`D92_QIC_G0_REJECTED`并以非零退出：

|类别|硬门|
|---|---|
|矩阵shape|两臂每scene`k_shot=10`、`old_class_count=6`、`registered_class_count=11`|
|QIC状态|`active=true`、`fallback=false`、`fallback_reason=null`、`final_state_sha256!=e0_state_sha256`|
|QIC闭合|intercept FP16 bit change>0、candidate residual严格小于E0、reduction>0、仅`intercept_fp16`修改|
|生命周期|decode=1；additional full/block/loo/fisher/scan/requantize全为0；actual FULL=1|
|查询|base与QIC各七项query访问字段全为false|
|状态/MAC|两臂query MAC均为正整数且等于`11*288`；两臂persistent state bytes均为正整数且相等|
|资源|candidate registration wall≤150ms、candidate/reference wall ratio≤1.50、incremental peak≤1MiB；512KiB为记录目标|

## 6.唯一detached command

完成本地验证和sole runner handoff后，唯一允许的远端启动命令为：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_qic_g0_launch_82ce7476_20260817_v2.sh >./d92_qic_g0_launch_82ce7476_20260817_v2.out 2>./d92_qic_g0_launch_82ce7476_20260817_v2.err </dev/null &
```

运行前必须确认source archive、launch、sealed输入、同run source/output/log和launcher stdout/stderr不存在；运行后只核验精确run-root/CWD/cmdline、GPU0、日志增长、两臂输出、`g0_validation.json`和固定marker。`fresh_run_retry=false`。

v2仅修复v1的发布层自引用缺陷：外层`nohup`重定向会在脚本开始前创建driver out/err，因此v2不在脚本内部重复要求这两个文件不存在；该fresh检查仍由sole runner在启动前完成。QIC科学实现、输入、三scene和全部G0门保持不变。

## 7.证据边界

本文件在运行前不声明G0通过、不声明方法性能、不读取accuracy/H/BA/floor/forgetting、truth或scorer。只有sole runner返回完整技术artifact后，主代理才可更新同一报告的最终状态和逐scene技术表。

## 8. Sole runner最终技术交接

状态：`ARTIFACTS_COMPLETE / TECHNICAL_G0_PASS / NO_PERFORMANCE_RESULT`

本run只执行一次冻结detached command，完成K10/new5、old6/registered11、三scene truth-free技术G0。未运行analyzer，未读取accuracy、H、BA、floor、forgetting、truth或scorer；prediction artifact仅随run完整取回，未读取其内容。

### 8.1 RULES_READ / PRECHECK / SYNC / COMMAND

|项目|结果|
|---|---|
|RULES_READ|`VERIFIED`；live AGENTS.md、项目.md、Git Bash skill、failure catalog、N607 automation skill均完整读取|
|PRECHECK|`VERIFIED`；普通direct N607、项目根/Python/8GPU可见，v2七个fresh路径及同run进程ABSENT，GPU0空闲|
|SYNC|`VERIFIED`；archive 218553B、SHA256=`802a52557657d6d415992192fd546fe564495ff71354a7890807645bb071113a`；launch 6623B、SHA256=`fe9c708735c6416dec0958ece5cca1c3c26fd27a242e28e9211359fc9eed467e`；远端`bash -n`通过|
|release/runtime|release HEAD=`2392a8b79444036d66edb85bab27b2cc827ebc5b`；runtime=`82ce747643af71ac3737bc0a89d18114be96f27e`|

唯一执行命令：

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_qic_g0_launch_82ce7476_20260817_v2.sh >./d92_qic_g0_launch_82ce7476_20260817_v2.out 2>./d92_qic_g0_launch_82ce7476_20260817_v2.err </dev/null &
```

启动后短连接观察到source-root、run-root、logs-root和两臂输出均生成；进程已自然退出，无需要健康停止的异常。最终检查时间为2026-08-17 06:55:04（Asia/Hong_Kong），同run PID/PPID/CWD/cmdline匹配为0，GPU0–GPU7均无compute app。

### 8.2 固定marker与逐scene技术表

固定marker：`D92_QIC_G0_ACTIVE_QUANTIZATION_INTERCEPT_CLOSURE_RESOURCE_PASS`。`g0_validation.json`的`status`、`marker`和`pass`均匹配，三个`scene_gates`均为true。

|scene|K/类形状|QIC active/fallback|E0→final state SHA256（缩略）|FP16 bit change|E0 residual→candidate residual→reduction|modified fields|decode/additional-full/block/LOO/Fisher/scan/requantize|query MAC candidate/reference/delta|state bytes candidate/reference/delta|wall candidate/reference/ratio|peak bytes|query/protocol gates|
|---|---|---|---|---:|---|---|---|---|---|---|---:|---|
|`leo_clear_weak`|10/6/11|true/false|`e473eeef…04366`→`6f3853af…60894`|11|229.1639949631→0.0364577573→229.1275372059|`intercept_fp16`|1/0/0/0/0/0/0|3168/3168/0|8583/8583/0|64.911634ms/61.707970ms/1.051917|8192|base/QIC七项全false；全部gate=true|
|`leo_low_elev_weak`|10/6/11|true/false|`6258f924…bee71f`→`b7909151…c181b`|11|173.3684354920→0.0131944239→173.3552410681|`intercept_fp16`|1/0/0/0/0/0/0|3168/3168/0|8583/8583/0|64.196002ms/61.590854ms/1.042298|94208|base/QIC七项全false；全部gate=true|
|`leo_rain_weak`|10/6/11|true/false|`7d367a41…775b`→`52f66c1d…1689`|11|251.8618226068→0.0219630476→251.8398595592|`intercept_fp16`|1/0/0/0/0/0/0|3168/3168/0|8583/8583/0|63.808285ms/61.644737ms/1.035097|32768|base/QIC七项全false；全部gate=true|

逐scene完整SHA、残差和字段均保存在取回的`fit_audit.json`中；表中SHA仅为可读缩略，不构成新的hash gate。

### 8.3 资源与协议收据

- QIC support上界：`34848`MACs，transient bytes上界`357024`，复杂度`O(C*K*288)+O(C*288)`。
- 三scene candidate registration wall均≤150ms，candidate/reference wall ratio均≤1.50，incremental peak均≤1MiB且均低于512KiB目标。
- 三scene均为一次coefficient decode；additional FULL、block、LOO、Fisher、candidate scan、requantize均为0，且`actual FULL once`为true。
- 仅`intercept_fp16`修改；coefficient/scale/log_diag/intercept_fp32/state shape/class registry均byte-exact，direct state publish、all-class shared formula、class/row permutation invariance均为true。
- base与QIC七项query访问字段全部false；query MAC为`3168=11*288`，QIC query MAC delta为0；persistent state bytes delta为0；clean/source sample access为false，support-only为true。

### 8.4 ARTIFACTS / CLEANUP / NEXT_ACTION

完整取回至`E:/type10-7/local_artifacts/d92_e0_full_d42_qic_g0_k10_20260817_v2/`：source root、reference/candidate run root、logs、driver out/err。`g0_validation.json`、两臂`fit_audit.json`、execution/resource receipts和prediction artifacts均存在；prediction artifacts未读。driver out仅含固定PASS marker，driver err为空。

远端source/run/logs/driver保留未删；最终同run进程为0、GPU0–GPU7已释放。每次SSH/SCP后均核验本地主机无存活SSH/SCP客户端及到N607的ESTABLISHED连接。

下一动作：主代理可基于这份truth-free技术G0收据决定是否推进预注册的完整矩阵；本run不得重试、重启、改写或被当作性能结果。
