# NEXT-R5 FA-RDCE3→qKNN Target125实验报告（r6）

## 身份与状态

- run ID：`next_r5_fa_rdce3_q_target125_20260805_r6`；日期：2026-08-05；状态：`LOCAL_VERIFIED`。
- 目标：完成同一sealed输入下FA-RDCE3+qKNN完整125四状态矩阵，联合验证域适应与新类注册。
- 主agent：`gpt-5.6-sol/high`；科学实现/独立复核：`Terra/max`；唯一N607 runner：`Luna/max`。
- Git branch=`codex/next-r5-fa-q-target125-20260805`；科学commit=`8ae765fd2107db477233b8e27af5f91a69e633c6`。

## r5失败与r6唯一修复

- r5因两个不同shard在REG1 support出现相同`z_id rows contain a zero-norm vector`而技术停止；678个partial prediction保留，但无manifest、truth或score，状态为`NO_PERFORMANCE_RESULT`。
- r6正常非零R0行继续使用原SHA-bound sealed runtime输出；同一fixed received IQ和同一checkpoint的pre-ReLU side tap只用于逐行ReLU绑定，并仅在sealed行精确为零、同IQ ReLU也精确为零且pre-ReLU有限非零时提供signed unit fallback。
- 不读取truth、role、quota或其他query，不更新状态；FA、RDCE、qKNN参数/logit/tie、K1 alias和125矩阵不变。
- `ssr-gpu`六入口`py_compile`、五份聚焦测试`25 passed`、`git diff --check`通过；独立Terra复核`P0=0，P1=0`。

## 版本闭包

- closure：`E:\type10-7\code\snapshots\next_r5_fa_rdce3_q_target125_20260805_r6_closure_8ae765fd.tar`；大小=`73154560`bytes；SHA256=`4227bccfbd614ba5dd57f1bd75efe1738539c2f671c5e68be3e53908e742b113`。
- method lock/builder/core/runtime/CLI SHA256：`3b9059f545620bf2a47e8bd79b537ede15a1eb7fdce4be3fee952d4a27dcc6b9`/`9b6b938d87fdfa603f6e0c8be374c77ca7399430ea898cd37c0c37a530880e38`/`88777df6cdaf352bea37c9b4bd36a78a9b32a24947bf5404b32dad2037b7c2ac`/`994412c0c1bed7ede9c98b13132c1c3a22eed381c6698bdff9a2a6f4d7219336`/`d0811d699629aa71b75d9d6f111a48f2d2cfc0468788d14e95b3df62bcc0cca5`。

## 冻结矩阵与输入

```text
receiver={20-1,3-19,7-14,7-7,8-8}
seed={713102,713103,713104,713105,713106}
(K,new)={(10,5),(10,10),(10,20),(5,20),(1,20)}
125 outer×3 scene×4 state=375 scene rows/1500 logical surfaces
=1350 unique predictions+150 K1 aliases
```

- D106 strict tap SHA=`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`；receipt SHA=`24badfa3f56c8f1b98a35768ea102a6c8e13267fcff80d59060ec6f2f13e0665`。
- checkpoint SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- D108 plan/context SHA=`13665ce5404c8ba34b3b05b7fd161baad05a96cec6d542416e115b3a9d6bd348`/`067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f`。
- capsule/split未变化，复用`p2_min_v1/VALIDATED_ONCE`，不重验数据。

## N607冻结执行

- RUN_ROOT=`/home/szu2070436088/2510044040/CV-SincNet/runs/next_r5_fa_rdce3_q_target125_20260805_r6`，落地前必须`ABSENT`；r1至r5禁止触碰。
- Python=`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD=`RUN_ROOT/source`；GPU0至7各一固定shard。
- 顺序：direct preflight→输入/GPU/RUN_ROOT核验→SCP closure→SHA/compile→用新method lock重建asset→prepare→truth-free smoke→8 shards→merge→truth-open→score→取回。
- asset命令固定使用`build_next_r5_fa_target125_asset.py`、上述strict tap/checkpoint、新method-lock绝对路径及SHA`3b9059...c6b9`；其余阶段固定使用`run_next_r5_fa_target125.py`对应子命令，实产asset/plan/context SHA逐阶段封存。
- 成功闭合：125/125 outer、375/375 scene、1500/1500 logical、1350 unique、150 alias、8/8 shard、完整manifest、truth和score。
- 停止仅限P0/安全/hash/覆盖故障或至少两个不同row在prediction前出现相同确定性异常；不得因性能停止。fresh retry authority=`无`。

## 结果待填

|outer|scene|logical|unique|truth|score|结论|
|---:|---:|---:|---:|---|---|---|
|0/125|0/375|0/1500|0/1350|未打开|未产生|`LOCAL_VERIFIED`|

