# Phase1双读出bundle build one-shot报告

## 0.状态

- 目标模式：`ACTIVE`
- run ID：`phase1_dualreadout_bundle_v2_build_oneshot_20260808_v1`
- 状态：`LOCAL_VERIFIED / P0_P1_CLOSED / READY_FOR_N607`
- 证据等级：`TECHNICAL_BUNDLE_NOT_PERFORMANCE_PROMOTED`
- 时间：2026-08-08
- 实现commit：`c5a8caa8bab5ebbc1735dae8528073255b5992a4`

## 1.目的与唯一修复

本run只修复真实NPZ的physical ID作用域：裸`sig_id`在2400行中只有913个唯一值；`(TX,RX,day,sig)`在全体2400行及source 1600行中均完全唯一。build保持“每条校准物理记录ID唯一”的门，不放宽或删除检查，只用canonical JSON构造全局作用域ID；`eq_id`、channel view和sat scenario不进入ID，因此不会把数学view计为新物理样本。

本地新增同sig跨TX/RX/day合法、完整行重复失败、`None/NaN`缺失失败测试。`ssr-gpu`共14项focused tests、语法与`git diff --check`通过。

独立定向复审：`P0=0`、`P1=0`、`ALLOW_BUILD_ONESHOT=YES`。固定commit的module/script/test归档SHA256分别为`c177dc87d547bf2f74b11808cec31343805151e80c472744fe8e4e2440d55896`、`a988af44fc5e90fb345a91d4f693b9daa819a045a2047ffcc2e5d5a63b42de25`、`7ef7548d2c5bdf13026835ab339e9d57246784a7ee8085f3e7f293d44749428b`。

## 2.只读技术输入

新run不恢复失败run，只把上一run已完整通过的技术子产物作为只读输入，并在执行前验证哈希与parity receipt：

| 输入 | SHA256 |
|---|---|
| angular runtime | `7d40592a2a720aa4b7c6fa6f4a66c2019db1114310feee5d0ee6194c2dd1b93c` |
| robust runtime | `a23da1e479990d3f711eba667591eb5886f70b8617ac55eb1bc38b0ca7728d97` |
| angular parity | `62e2f6c5abc5c5846e6038d2478af0baa8bd74c5fd279fd519a89e6b2219ae85`；`max_abs=0.0` |
| robust parity | `c7ba2896f93979eaac3edfe270039c8444ac35399049530a57be4c71832cb20a`；`max_abs=0.0` |
| C `z_dom` NPZ | `52585d273fa245dbe57314eba7eb24bb1cd3e32e7deb7183b89e8972cfd9b6bd` |

B/C `z_id` NPZ和checkpoint SHA沿用已冻结值。上述5项位于`runs/phase1_dualreadout_bundle_v2_cpu_oneshot_20260808_v1/inputs/`，只读引用，不移动、不覆盖。

## 3.冻结执行与路径

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_dualreadout_bundle_v2_build_oneshot_20260808_v1_c5a8caa8`。
- run/log：`/home/szu2070436088/2510044040/CV-SincNet/{runs,logs}/phase1_dualreadout_bundle_v2_build_oneshot_20260808_v1`。
- 资源：CPU单进程；不占GPU。
- 命令：严格串行build→emit 2400→proxy score→held score，参数、source-only角色、class handles、content root外部传入和评分边界与原冻结矩阵相同。
- retry：`NO`。

任何输入hash/parity、逐行metadata、全局physical ID、source-only fit、bundle allowlist/content root、evidence role/truth、行数或执行错误均停止且无性能结果。不得调方法、阈值或复用scorer结果。

## 4.结果表（待回收）

| candidate | category | receiver/TX split | K-shot | seed | known/unknown | coverage/defer | bundle summary | verdict |
|---|---|---|---:|---:|---|---|---|---|
| P1-DUALREADOUT-BUILD-ONESHOT | source-calibrated bundle build | 4/1/1 | N/A | 7281105 | 待回收 | 待回收 | C class/domain+B continuous JS | `NO_PERFORMANCE_RESULT_YET` |

## 5.科学边界

proxy/held指标仅为`SOURCE_HELD_PROXY_NONDEPLOYMENT_DIAGNOSTIC`；本run不提供Phase3真实unknown、same-event多节点或真实在轨结论。
