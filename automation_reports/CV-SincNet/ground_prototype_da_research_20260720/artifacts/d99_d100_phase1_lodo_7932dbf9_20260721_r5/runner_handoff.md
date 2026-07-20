# D99/D100 Phase1 nested-LODO r5 runner handoff

## 最终状态

- run ID：`d99_d100_phase1_lodo_7932dbf9_20260721_r5`
- 状态：`LOCAL_VERIFIED -> PREFLIGHT_BLOCKED_MODULE_SHA_MISMATCH`
- child：未启动
- PID/log/exit：不存在
- 远端run/log/output：均未创建
- 配置：未上传到N607
- 自动重试：无，未授权
- 方法、配置、候选：未修改
- target访问：无

## 本地预登记

|项|值|
|---|---|
|Git HEAD|`245b9beba14e417517e7a4cb6108efa566f8aceb`|
|配置路径|`E:\type10-7\code\snapshots\ground_proto_da_rd_wt\automation_reports\CV-SincNet\ground_prototype_da_research_20260720\preregistered_inputs\d99_d100_phase1_lodo_7932dbf9_20260721_r5.json`|
|配置大小|`3008 bytes`|
|配置SHA256|`6df14d7d17fc4fe9d3eb786001eeeedcf92d640aa190d6974a60793bf6b8e30e`|
|seed|`991`|
|metric seed|`713101`|
|候选数|`64`|
|execution mode|`development_diagnostic`|

本地Git工作树在preflight前为clean。

## 已通过的远端门

- N607 direct preflight：`PASS`，时间`2026-07-21 03:37:37 CST`。
- r5 run根：创建前不存在。
- r5 log根：创建前不存在。
- r5 output：创建前不存在。
- GPU5及其余GPU：`0%`,`10/24576 MiB`，无compute process。
- r4 archive NPZ：`cdd8747d267336b48e8c555ce7e010206f042ff07c695af351541a97187fad03`，匹配。
- r4 archive manifest：`5f363bc09503f882c66aa92805657199ca57484f627c5d2805254cad07bffa15`，匹配。
- r1 ground NPZ：`e69409268ad1215d440fd0555a4f8e2903214e95062f22a0c5f436fb2b799bd4`，匹配。
- r1 ground manifest：`f92a1bd6e936c4e661517da73ee39cc5e94f77b796f50587dd33f6c0d2da8f0d`，匹配。
- r1 base lock：`7481c351a3114432df0c06c8527b1f625c15c05b1e3aed89000098b26f022a9e`，匹配。
- D19 component manifest：`15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c`，匹配；文件为`int8_component/manifest.json`。

## 阻断门：5模块SHA全部不匹配

规定源码根：

```text
/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/source_7932dbf9
```

|配置键/文件|配置期望SHA256|r4只读源码实际SHA256|结果|
|---|---|---|---|
|`run_d99_d100_phase1_lodo` / `code/scripts/run_d99_d100_phase1_lodo.py`|`1c625411439e20671516206400a38ec05835732237d90d11367be87bec7f8ab7`|`110295caa83ab0d7717e26b17b1d4ac33423337afaa8877067f64649d06c7ea1`|不匹配|
|`stage2_d100_ra_cgspr_lgf`|`2c8ce8f7063256624c0ce740510b61e0527e3540ce7d41dd99ca4b4d49fb64af`|`86c185ee13222bc0c97c4576984b9cd07f981201da4f0b62f8d4bc66970b4714`|不匹配|
|`stage2_d81_phase1_episode_scorer`|`05dc600ea169ce9deb629ff4c764179cffda7eded16280ae6914bf4d950c0ef4`|`54ee742c81b60e00b6c1c36d2d6bf1f0409ad10f72a25e01c2dcd589093be55d`|不匹配|
|`stage2_d99_d100_phase1_lodo`|`56cc705f3568b67242d657bc1e922d2aebc3b2d27041915420dd65f9ab0005f7`|`aa99b3d726338481ed7f22f4acc5cdf2cfe4b2ef420e44da6f2ff2f674841e0e`|不匹配|
|`stage2_d99_ra_cgtmk_d81`|`e3f7e4990c842c60c9b879334a8f99ecfff7f95ce6afaccbb321827a1f9c89f7`|`c166a5e375b0b8be5c95e678e63a6f04526474cd1a01544616829106af52f56f`|不匹配|

因此，配置绑定的模块字节并不是指定r4只读源码中的实际模块字节。按明确发布规则，“模块SHA不一致即停”，runner没有：

- 将Git提交`245b9beb`的新模块同步进r4旧源码目录；
- 改写配置以接受r4旧模块SHA；
- 创建r5远端目录或上传配置；
- 启动child或产生任何LODO结果/receipt。

## 需要主线重新统一的选择

后续新预登记必须在以下两种版本身份中明确选择一种，不能混用：

1. 保持只读复用r4`source_7932dbf9`，则配置中的5模块SHA必须绑定上表实际值，并确认旧模块确实实现当前r5方法与输出合同；或
2. 使用Git提交`245b9beb`对应的5个新模块及其当前期望SHA，则需要新的隔离源码包/源码根、远端落地、compile/import/test证据和新run ID，不能覆盖r4源码或本r5。

本runner不替主线作此版本选择。

## SSH终态

```text
ssh_processes=0
n607_established=0
```
