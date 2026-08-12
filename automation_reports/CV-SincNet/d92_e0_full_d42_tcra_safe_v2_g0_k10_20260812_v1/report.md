# D92 TCRA_SAFE_DIRECTIONAL_v2 K10 G0报告

## 1.身份与边界

|字段|冻结值|
|---|---|
|run ID|`d92_e0_full_d42_tcra_safe_v2_g0_k10_20260812_v1`|
|候选|`E0_FULL_D42_TAIL_CLASS_ROW_ASCENT`；final gate=`safe_directional_v2`|
|科学commit|`6a74c410`（父commit`b2934f62`）|
|状态|`ARTIFACTS_COMPLETE / G0_ACTIVE_RESOURCE_PASS`|
|outer|`rx_7_7__seed_713106__k_10__new_5`；三场景|
|目标|truth-free复核v2三场景active、state非E0、协议闭合和wall P90≤150ms|
|声明|`DEVELOPMENT_ONLY_SUPPORT_G0 / NO_PERFORMANCE_RESULT`；禁止scorer/truth/accuracy读取|

v1同outer未读取truth，三场景wall为121.518–137.051ms，但严格七组正增益门导致fallback。v2保留E0固定tail、原子、排序、prefix真实D42守卫和query边界，仅冻结统一final门：七组及new-all在`-tol`以上、双hinge不超过`tol`、至少一个旧类严格正、六旧类gain总和严格正、selected>0且state非E0。该修订不得表述为新类收益；clear的pooled-new support gain仍为负但在tol内。

## 2.本地证据

- TDD先RED，core/query和相邻TPCE/E0D-slim共79项GREEN。
- query独立重算revision、old gain sum、strict-positive count和safe-directional pass，active/fallback/alias均fail-closed。
- C26/K10交错基准v2/v1为28.145/28.016ms，比值1.005。
- `py_compile`、`git diff --check`通过；独立审查P0=0、P1=0、APPROVE。
- v2设计commit=`5890e5d1`；实现commit=`6a74c410`。

## 3.发布输入

|文件|大小|SHA256|
|---|---:|---|
|`d92_tcra_safe_v2_g0_runtime_6a74c410.tar.gz`|6,073,468B|`24ea05944806503085755fccbaa2c6e451653ecaedc1d95843c332eb95fc00fc`|
|`launch.sh`|8,057B|`5aab845495da243c522ea0039ce3b79040bd2987feec753231bb7cf5402c286d`|

- source：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_tcra_safe_v2_g0_source_6a74c410_20260812_v1`
- output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d92_e0_full_d42_tcra_safe_v2_g0_k10_20260812_v1`
- logs：`/home/szu2070436088/2510044040/CV-SincNet/logs/d92_e0_full_d42_tcra_safe_v2_g0_k10_20260812_v1`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；GPU0；CPU threads=2。

唯一detached命令：

```text
cd /home/szu2070436088/2510044040/CV-SincNet/runs/d92_tcra_safe_v2_g0_source_6a74c410_20260812_v1 && nohup bash ./launch.sh >./launch_driver.out 2>./launch_driver.err </dev/null &
```

## 4.通过门与后续

三场景必须active、无fallback、revision=`safe_directional_v2`、state非E0、仅`coef2_qint8`改变、safe-directional门闭合、fit=2/1且postfit=0、query七项全false、P90≤150ms。launcher已修正v1验证器路径为`output/after/fit_audit.json`。

通过后才建设Hard9+K1：从原Hard10排除本G0 outer，剩余9个最难performance outer一次性运行。八项同排均值任一平或负直接`REJECT_ROUTE`，不得再改门、挑scene或同run重试。

## 5.G0实测结果

唯一冻结命令执行一次，三场景均完成truth-free预测；`g0_validation.json`标记为`D92_TCRA_SAFE_V2_G0_ACTIVE_RESOURCE_PASS`。未运行scorer，以下仅是机制与资源证据，不是性能结果。

|场景|active/fallback|generated/selected|旧类最小gain|旧类gain和|pooled-new gain|双hinge delta|wall|
|---|---|---:|---:|---:|---:|---|---:|
|`leo_clear_weak`|`true/false`|27/21|-0.000616|0.012791|-0.000231|0/0|126.191ms|
|`leo_low_elev_weak`|`true/false`|33/24|-0.000418|0.012804|0.000819|0/0|138.106ms|
|`leo_rain_weak`|`true/false`|27/21|-0.000153|0.010555|0.000616|0/0|122.312ms|

- 三场景`final_gate_revision=safe_directional_v2`、`support_guard_pass=true`、`safe_directional_pass=true`。
- 三场景final state SHA均不同于E0 state SHA，修改字段仅`coef2_qint8`；query七项访问均为`false`。
- wall P90（3场景nearest-rank）为138.106ms，低于150ms硬门。
- clear的pooled-new support gain仍为负但在冻结容差`-0.000806`以上，因此该G0只能说明候选机制可执行；是否提升新类准确率必须由与本G0 outer不重叠的Hard9 truth-last实验决定。

结论：进入一次性Hard9+K1验证，不扩大方法、不回看本G0 truth、不在同run调参。

## 6.回收与清理

完整artifact已回收到`E:\type10-7\local_artifacts\d92_e0_full_d42_tcra_safe_v2_g0_k10_20260812_v1`，远端与本地逐树一致：

|目录|文件数|字节数|canonical tree SHA256|
|---|---:|---:|---|
|`source`|1,417|73,355,925|`29cd5a741073b7bfc325cf1f26a5eae3c549b0a02938a02ba234c355ad29e62a`|
|`output`|10|729,571|`059a23fedeb0ef0b4482bfbb45585d3b4b403c6741d8361150d04a20ed68cce4`|
|`logs`|5|1,553|`bc21a7969ee0962c7e26da95eb43e75972bea32f2012ac9792efcbba8121b42f`|

终态同run进程为0，8卡均释放；本地`ssh.exe`及N607/bridge TCP22连接均为0。远端artifact保留，未删除或覆盖。
