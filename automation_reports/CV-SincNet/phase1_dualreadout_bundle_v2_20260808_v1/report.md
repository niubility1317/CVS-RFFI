# Phase1双读出deployment bundle v2报告

## 0.状态

- 目标模式：`ACTIVE`
- run ID：`phase1_dualreadout_bundle_v2_20260808_v1`
- 状态：`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`
- 证据等级：`FAILED_TECHNICAL_LANDING_NO_BUNDLE`
- 时间：2026-08-08
- 实现commit：`ebf9764caeb31562c41f4fb520d969e542803ee0`

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
| `code/cvsrffi/phase1_dualreadout_bundle_v2.py` | `257ac635b0b7b400848b5c9a9896b9c27b69f06e416c92d7598d433207f69a29` |
| `code/scripts/phase1_dualreadout_bundle_v2.py` | `38176226d160622317809a0e1d033246043d9f8b12e17d7a395d51ace4358c56` |
| `code/tests/test_phase1_dualreadout_bundle_v2.py` | `f2446998131a73be0f495c5712e26453be7104197b603cda618675fa58daa666` |

固定commit的`git archive`CRLF字节SHA256依次为`c177dc87d547bf2f74b11808cec31343805151e80c472744fe8e4e2440d55896`、`3baf32d9cb49c664c9a56bc5de92e2563da187444644e4ba7bb394af2faec12a`、`cfa51dfb4b345e4c83b4f3e2914ab007e3c1a0d8cffdfe4ac66314aff08560c3`，N607按该归档口径核验。

本地`ssr-gpu`验证：12项focused tests通过；CLI compile/help通过；`git diff --check`通过。新增覆盖严格source-only calibration role、receipt exact allowlist，以及parity receipt与runtime/checkpoint/shape/batch/numerical gate绑定。

独立定向审查：`P0=0`、`P1=0`、`ALLOW_N607_REAL_BUNDLE=YES`。审查仅复核上述3个原P0，不增加P2发布门。

## 3.N607冻结发布

- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_dualreadout_bundle_v2_20260808_v1_ebf9764c`
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

### 4.1实际终态

三条GPU命令均只执行一次并exit=1；CPU build、emit和两次score均未启动。angular exporter因默认forward返回二维logits张量却按`result['z_id']`读取而失败；robust和`z_dom`导出因runner同时设置`CUDA_VISIBLE_DEVICES`并传物理`cuda:1/2`造成`invalid device ordinal`。三条任务均未生成runtime、parity receipt或`z_dom` NPZ。

该run不重试、不覆盖。完整命令、PID、completion、三份stdout和manifest已回收到`artifacts/logs/`，8项小文件逐项哈希匹配；GPU0–7、Python进程和SSH连接均已释放。修复只允许进入新的不可覆盖v2 run。

## 5.结果表（待回收）

| candidate | category | receiver/TX split | K-shot | seed | known/unknown | coverage/defer | bundle summary | verdict |
|---|---|---|---:|---:|---|---|---|---|
| P1-DUALREADOUT-BUNDLE-V2 | source-calibrated technical bundle | 4/1/1 | N/A | 7281105 | N/A | N/A | 未生成bundle | `STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE` |

## 6.科学边界

proxy=`8-20`和held=`6-15`只能形成source-held非部署诊断。即使达到数值目标，也不能替代Phase3合法物理event/reception绑定、真实unknown或同步多星结论。
