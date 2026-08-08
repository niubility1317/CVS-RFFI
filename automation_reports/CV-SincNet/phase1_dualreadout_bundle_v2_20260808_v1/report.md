# Phase1双读出deployment bundle v2报告

## 0.状态

- 目标模式：`ACTIVE`
- run ID：`phase1_dualreadout_bundle_v2_20260808_v1`
- 状态：`LOCAL_VERIFIED / WAITING_INDEPENDENT_P0_P1`
- 证据等级：`TECHNICAL_BUNDLE_NOT_PERFORMANCE_PROMOTED`
- 时间：2026-08-08
- 实现commit：`5a78b37b495c3268771530bfca35031d483fd373`

## 1.目标与冻结假设

目标是把已经完成训练的B角度几何checkpoint和C LEO一致性checkpoint变成不可变双runtime bundle。C独占registered类别、`z_id`和`z_dom`；B只提供连续JS陌生度，不能覆盖类别或作为硬一致性门。

校准只读取source-known。proxy/target/held行即使存在于输入NPZ，也不得影响任何中心、半径、尺度或阈值；它们只允许在bundle content root封存后由独立scorer打开。该run不产生Phase3真实unknown或same-event多节点结论。

## 2.输入与版本

| 输入 | 远端路径/哈希 |
|---|---|
| B checkpoint | `runs/phase1_geosat_lite_4arm_20260808_v1/B_ANGULAR_Z0/final_ssdg.pth`；`f0f89b9251f6ada33778975b08ced3d9d407623b91a16d685d9d3bad9fa2070f` |
| C checkpoint | `runs/phase1_geosat_lite_4arm_20260808_v1/C_LEO_CONS_Z0/final_ssdg.pth`；`9a1be4c739275f3c623f0df3d049f4a8b99b1ac51b21f4a50dca809fb23727e0` |
| B z_id NPZ | `postfreeze_audit_v1/B_ANGULAR_Z0/features.npz`；`31fc239ac7705488d1999b103902a04165ba4f4ccbfb1fe230f89a2a9f507c02` |
| C z_id NPZ | `postfreeze_audit_v1/C_LEO_CONS_Z0/features.npz`；`b4e980a5495f2d297d61d461d30c6a510f9eb8bb9c51e31b2f16ebe7e247e4c6` |

| 本地文件 | SHA256 |
|---|---|
| `code/cvsrffi/phase1_dualreadout_bundle_v2.py` | `91cf72c86dac0a6d2d4625390689a4e780dd72c342a6af0987bf92114a12aa59` |
| `code/scripts/phase1_dualreadout_bundle_v2.py` | `b18d9524b808289c460a95f7b8afd25f02d9fbf890e1c389e4bc0d33e1575c7c` |
| `code/tests/test_phase1_dualreadout_bundle_v2.py` | `86ee0d7ca619e3787a526a02b3f3a9284d45c4e3a653adc151962a44f3a5431d` |

本地`ssr-gpu`验证：9项focused tests通过；CLI compile/help通过。覆盖source-only fit、proxy数值不影响、physical ID唯一、两runtime加载、B/C职责、member篡改/额外文件、外部content-root、无role/truth evidence和defer评分语义。

## 3.N607冻结发布

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_dualreadout_bundle_v2_20260808_v1_5a78b37b`
- run：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_dualreadout_bundle_v2_20260808_v1`
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_dualreadout_bundle_v2_20260808_v1`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- GPU0：B TorchScript export；GPU1：C TorchScript export；GPU2：C `z_dom` actual-IQ export。每卡1个进程；其余卡不占用。
- retry：不授权自动重试。

三条独立导出可并行；随后严格串行build→emit→proxy score→held score。C `z_dom`导出必须复用既有2400行选择参数：source TX四项、target_old=`6-15`、proxy_unknown=`8-20`、days=`0,1`、rx=`0..6`、每TX 400、seed=`7281105`、全部clean view、raw IQ不持久化。

bundle成员固定为两个TorchScript、`calibration.npz`、source-only receipt和manifest；禁止`.pth`、raw IQ、role、truth和物理样本ID。emit输出2400条`proxy_unverified`逐reception本地证据，不构造多节点event。

## 4.成功与停止

成功：B/C eager↔TorchScript在batch 1/8/64 parity通过；C `z_dom`与B/C `z_id`逐行元数据相同；bundle exact allowlist/content root验证通过；source-known smoke可运行；2400条证据不含role/truth；两次scorer只在bundle封存后运行；小artifact和哈希回收完整。

停止：checkpoint/NPZ/代码hash不匹配、旧run/log/release已存在、runtime parity失败、物理行绑定不一致、proxy/target进入fit、bundle出现禁用成员、证据出现truth/role或覆盖风险。不得根据proxy/held性能决定重试或调参。

## 5.结果表（待回收）

| candidate | category | receiver/TX split | K-shot | seed | known/unknown | coverage/defer | bundle summary | verdict |
|---|---|---|---:|---:|---|---|---|---|
| P1-DUALREADOUT-BUNDLE-V2 | source-calibrated technical bundle | 4/1/1 | N/A | 7281105 | 待回收 | 待回收 | C class/domain+B continuous JS | `NO_PERFORMANCE_RESULT_YET` |

## 6.科学边界

proxy=`8-20`和held=`6-15`只能形成source-held非部署诊断。即使达到数值目标，也不能替代Phase3合法物理event/reception绑定、真实unknown或同步多星结论。

