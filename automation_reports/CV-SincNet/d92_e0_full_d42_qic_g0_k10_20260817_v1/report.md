# D92 E0 FULL QIC K10 G0预登记与发布报告

状态：`LOCAL_VERIFIED / NOT_RUN / NO_PERFORMANCE_RESULT`

本报告只登记一次K10/new5、三scene、truth-free技术G0发布。当前未执行N607、未启动driver、未读取truth/scorer或性能结果。

## 1.冻结身份

|字段|值|
|---|---|
|run ID|`d92_e0_full_d42_qic_g0_k10_20260817_v1`|
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
|archive整体SHA/成员数/路径安全|通过：`218585`bytes、38members；整体SHA为`05aab57ce47f4dfcce23aaef805959c4b6fc5278901dce09ba9e68a15af7a14b`|
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
source_root=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_qic_g0_source_82ce7476_20260817_v1
source_archive=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_qic_g0_source_82ce7476_20260817_v1.tar.gz
remote_launch=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_qic_g0_launch_82ce7476_20260817_v1.sh
reference_output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_d42_qic_g0_k10_20260817_v1/reference_e0
candidate_output=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_d42_qic_g0_k10_20260817_v1/candidate_qic
validation=/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_d42_qic_g0_k10_20260817_v1/g0_validation.json
logs=/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_d42_qic_g0_k10_20260817_v1
local_retrieval=E:/type10-7/local_artifacts/d92_e0_full_d42_qic_g0_k10_20260817_v1
```

## 4.archive与launch

archive从runtime commit`82ce747643af71ac3737bc0a89d18114be96f27e`的Git对象生成，复用CCOC G0 v6的package依赖成员集合，移除CCOC专属entry并加入当前QIC core/runner。未包含数据、checkpoint、truth或scorer；不建立逐成员SHA体系。

|字段|值|
|---|---|
|relative path|`runtime/d92_qic_g0_source_82ce7476_20260817_v1.tar.gz`|
|size|`218585`bytes|
|SHA256|`05aab57ce47f4dfcce23aaef805959c4b6fc5278901dce09ba9e68a15af7a14b`|
|members|38|
|required entries|`__init__.py`、D42、QIC core、Slim、Query、QIC runner|
|tar safety|禁止绝对路径、`..`路径和`code/code`嵌套|
|launch SHA256|`505fc632c987822fd698634aa1a927cb883e35323ae8cddf8c50dd0e2a1d0c8e`|

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
cd /home/szu2070436088/2510044040/CV-SincNet/runs && nohup bash ./d92_qic_g0_launch_82ce7476_20260817_v1.sh >./d92_qic_g0_launch_82ce7476_20260817_v1.out 2>./d92_qic_g0_launch_82ce7476_20260817_v1.err </dev/null &
```

运行前必须确认source archive、launch、sealed输入、同run source/output/log和launcher stdout/stderr不存在；运行后只核验精确run-root/CWD/cmdline、GPU0、日志增长、两臂输出、`g0_validation.json`和固定marker。`fresh_run_retry=false`。

## 7.证据边界

本文件在运行前不声明G0通过、不声明方法性能、不读取accuracy/H/BA/floor/forgetting、truth或scorer。只有sole runner返回完整技术artifact后，主代理才可更新同一报告的最终状态和逐scene技术表。
