# D92 TCRA_SAFE_DIRECTIONAL_v2 K10 G0报告

## 1.身份与边界

|字段|冻结值|
|---|---|
|run ID|`d92_e0_full_d42_tcra_safe_v2_g0_k10_20260812_v1`|
|候选|`E0_FULL_D42_TAIL_CLASS_ROW_ASCENT`；final gate=`safe_directional_v2`|
|科学commit|`6a74c410`（父commit`b2934f62`）|
|状态|`LOCAL_VERIFIED / READY_TO_LAND`|
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
