# D92 TCRA_SAFE_DIRECTIONAL_v2 Hard9+K1实验报告

## 1.实验身份

|字段|冻结值|
|---|---|
|run ID|`d92_e0_full_d42_tcra_safe_v2_hard9k1_20260812_v1`|
|候选|`E0_FULL_D42_TAIL_CLASS_ROW_ASCENT`；final gate=`safe_directional_v2`|
|代码commit|`824a6744`（科学实现`6a74c410`）|
|状态|`LOCAL_VERIFIED / READY_TO_LAND`|
|矩阵|9个performance outer+1个K1 liveness；30 scene-arm；8 shards|
|比较目标|完整125实验中的同outer`E0_FULL_ONLY`|
|声明|`DEVELOPMENT_ONLY_DISJOINT_FROM_G0_HARD_SCREEN`|

## 2.目标与假设

本实验只验证最难子集。用于修改final gate的G0 outer`rx_7_7__seed_713106__k_10__new_5`被强制排除，不参与性能统计。候选只在E0 FULL发布后的真实D42 state上修改`coef2_qint8`真类行；无query拟合、选择或truth访问。

八项同排均值必须全部严格优于E0：`H_old_new`、old balanced accuracy、`c_old_acc`、old floor、seen-new提高；average forgetting、new→old、old→new降低。任一指标持平或反向立即拒绝路线。只有全部同向后，才检查预注册幅度门和资源门。

## 3.冻结矩阵

performance outer依次为：

1. `rx_7_7__seed_713104__k_5__new_20`
2. `rx_7_7__seed_713103__k_10__new_5`
3. `rx_8_8__seed_713103__k_5__new_20`
4. `rx_8_8__seed_713103__k_10__new_5`
5. `rx_8_8__seed_713106__k_5__new_20`
6. `rx_7_14__seed_713104__k_10__new_10`
7. `rx_3_19__seed_713102__k_10__new_5`
8. `rx_7_7__seed_713105__k_10__new_20`
9. `rx_7_7__seed_713104__k_10__new_5`

K1 liveness为`rx_20_1__seed_713106__k_1__new_20`，仅验证精确E0 alias，不进入性能均值。smoke固定第1个performance outer，真实激活TCRA后才能启动shard。

## 4.本地验证与发布输入

- TCRA core/slim/query/Hard9 runner相关回归81项通过；Hard9聚焦8项通过。
- `py_compile`、runner CLI、config canonical identity及`git diff --check`通过。
- selection SHA=`4fc836fbe3960cf95bfdf9fdb9eba1d311fb47fa4cc2ff89b64acab7e88f8e61`。
- config SHA=`9740ebd8f7368ea73bf8bdfb1ff57735e7407f89dab7b51a834988c4d6f9f13e`。

|文件|大小|SHA256|
|---|---:|---|
|`d92_tcra_safe_v2_hard9_runtime_824a6744.tar.gz`|5,118,339B|`017a3c1c8137061cdd616c4548f6ec1f23c78d5dacb95250a6df4ca5b411e582`|
|`stage2_d92_tcra_safe_v2_hard10_v1.json`|7,213B|`9740ebd8f7368ea73bf8bdfb1ff57735e7407f89dab7b51a834988c4d6f9f13e`|
|`launch.sh`|3,576B|`867b5b9e58d81a8bc10a47f9a36486e0681c8680741dff07d49945fe90a78a90`|

远端冻结路径：

- source：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_tcra_safe_v2_hard9_source_824a6744_20260812_v1`
- output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_d42_tcra_safe_v2_hard9k1_20260812_v1`
- logs：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_d42_tcra_safe_v2_hard9k1_20260812_v1`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

唯一detached命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_tcra_safe_v2_hard9_source_824a6744_20260812_v1 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

## 5.健康门与裁决

smoke要求K5真实TCRA active、无fallback、safe-directional收据/fit/prediction/COMMIT闭合且query七项全false。系统停止只限协议/安全、wrong hash/overwrite、launcher确定性错误，或至少两个distinct outer产生同一pre-prediction异常；绝不按中间性能停止。同run不重试。

预期正式artifact：10 job receipts、20组before/after prediction/COMMIT/fit/resource/execution、10 score、8 shard summaries；smoke另有2组诊断artifact。完整取回后再由主代理独立做truth-last同排分析。

资源硬门：wall P90≤150ms、同排wall ratio中位≤1.50、peak≤E0+512KiB、query MAC/state exact；目标为P90≤120ms、ratio≤1.25。性能大胆幅度门为H≥+1pp、old BA≥+1.5pp、`c_old_acc`≥+1pp、old floor≥+4pp、seen-new≥+0.5pp、forgetting≤-1.5pp、两向混淆各≤-0.5pp。
