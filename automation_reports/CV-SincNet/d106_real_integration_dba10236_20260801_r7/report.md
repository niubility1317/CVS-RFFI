# D106真实集成r7预登记报告

状态：`ARTIFACTS_COMPLETE / TECHNICAL_CLOSURE_PASS / NO_PERFORMANCE_RESULT`

## 1.身份、目标与假设

- run ID：`d106_real_integration_dba10236_20260801_r7`
- 时间：2026-08-01
- operator：主agent本地release；N607专属Terra Max runner为唯一launch owner
- release source commit：`dba10236889a45b11f2f10dab3596aff7e218df0`
- 目标：修复r6正式RDCE资产结果发布字段合同后，执行no-query真实DATA→checkpoint→RDCE闭环。
- 假设：serialized/reloaded资产的`basis_codes_qint8.shape=(3,160)`能够闭合权威`RDCE_RANK=3`，正式结果和`COMPLETED.json`可发布。
- 比较目标：r6只作技术失败对照；本run不访问或比较性能。

r4、r5和r6均为已封存技术失败，r7不修改或复用其路径。

## 2.本地变更与验证

|文件|目的|
|---|---|
|`code/scripts/run_d106_real_integration.py`|从roundtrip真实资产形状验证秩，并以权威常量发布`rdce_rank`|
|`configs/d106_candidate_runtime_manifest_20260801.json`|绑定新入口SHA|
|`tests/test_run_d106_real_integration_cli.py`|以无`.rank`替身覆盖正式合同，并对missing/wrong shape fail closed|
|`tests/test_d106_real_integration_handoff.py`|机械核验r7 fixture、绝对路径、四映射、archive SHA/entry和秩修复闭包|

专项16/16通过；完整D106闭包126通过1跳过；独立终审为`P0=0/P1=0/P2=0 / GO`。方法数学、DATA选择、Phase1 tap、查询权限和method lock均未改变。

### 2.1 精确验证命令

```powershell
conda activate ssr-gpu
python -m pytest -q tests/test_d106_real_integration_handoff.py tests/test_run_d106_real_integration_cli.py
python -c "import zipfile; p=r'automation_reports/CV-SincNet/d106_rdce_gtsm_20260801_r1/artifacts/d106_real_integration_source_dba10236.zip'; z=zipfile.ZipFile(p); names=[n for n in z.namelist() if n.endswith('.py')]; [compile(z.read(n),n,'exec') for n in names]; print('R7_ARCHIVE_COMPILE_PASS',len(names))"
python -c "import sys; sys.path.insert(0, r'automation_reports/CV-SincNet/d106_rdce_gtsm_20260801_r1/artifacts/d106_real_integration_source_dba10236.zip/source/code'); import baseline_origin_sat_view, model, model_dual_cvsincnet; from cvsrffi.stage2_d105_phase1_bundle import D105_CANDIDATE_RUNTIME_MODEL_FILES; assert tuple(D105_CANDIDATE_RUNTIME_MODEL_FILES)==('baseline_origin_sat_view.py','model.py','model_dual_cvsincnet.py'); print('R7_ARCHIVE_THREE_MODEL_ZIPIMPORT_SMOKE_PASS')"
```

结果分别为25/25、`R7_ARCHIVE_COMPILE_PASS 200`和`R7_ARCHIVE_THREE_MODEL_ZIPIMPORT_SMOKE_PASS`。直接zipimport RDCE模块会被其真实文件路径execution-closure主动拒绝，故不作为release smoke；runner须解压后在真实`source/code`路径执行RDCE import。

## 3.Release资产与同步目标

|资产|SHA256|远端目标|
|---|---|---|
|`d106_real_integration_source_dba10236.zip`|`1eae03c8a63ede8241c4b3cb7331994ffb32e571608774e1dd874d30c928a585`|`<run-root>/input/release/source.zip`|
|`d104_split_4a1e23cc.zip`|`b1884cf1a7e287aa489a2b591fc5688a7e655c6b541f6f90eafcf71cf476372e`|`<run-root>/input/release/d104_split.zip`|
|`d106_real_integration_fixture_dba10236_r7.json`|`d8c3475dca9cdd82450a63b6b8a4097dc96a98ca8849d55e2df3cf51c59ba669`|`<run-root>/input/d106_real_integration_fixture_dba10236_r7.json`|
|`d106_train_held_disjoint_receipt.json`|`ee7005fcc99d703dac2f3e529e39426587ffa8967d19c15cf848c98f5295d961`|`<run-root>/input/d106_train_held_disjoint_receipt.json`|

完整本地绝对路径、entry SHA和远端绝对映射见`d106_real_integration_runner_handoff_dba10236_r7.md`。

## 4.N607预登记

- run root：`/home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7`
- Conda/Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- CWD：`<run-root>/source`
- GPU：物理GPU0；`CUDA_VISIBLE_DEVICES=0`；进程内`cuda:0`
- log：`<run-root>/logs/run.out`
- PID：启动后由唯一runner记录
- output：`<run-root>/output`，启动前`ABSENT`
- expected：selected-IQ、strict-tap、RDCE wire、`d106_real_integration_result.json`、`COMPLETED.json`
- retry：`NOT_AUTHORIZED`

精确命令冻结在handoff §4。启动成功不等于技术完成；只在result/marker canonical、SHA、roundtrip binding、`rdce_rank=3`及四个禁用访问/性能标志全部闭合后标为`ARTIFACTS_COMPLETE`。

唯一允许的服务器子命令为：

```text
CUDA_VISIBLE_DEVICES=0 /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/source/code/scripts/run_d106_real_integration.py --fixture /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/input/d106_real_integration_fixture_dba10236_r7.json --output-dir /home/szu2070436088/2510044040/CV-SincNet/runs/d106_real_integration_dba10236_20260801_r7/output --device cuda:0
```

## 5.数据、配置与判定

- dataset：D104冻结source split；L_s=588；checkpoint SHA=`2699eedc…`
- protocol：`p2_min_v1`
- candidate：`D106-RDCE/GTSM-r3-SCATTER02`
- method lock SHA=`e7a1982b…`
- runtime SHA=`ba8e96a9…`
- source split SHA=`4a1e23cc…`
- seed/K/scenario：本run仅构建固定Phase1 aggregate，不执行Target25或performance matrix
- source-held truth、formal query、Target：全部禁止

成功标准是技术闭环完整且无越权；不存在accuracy、H、BA、floor或其他性能早停。P0协议/安全错误、wrong hash/path、overwrite风险、两行同指纹零prediction错误或确定性入口异常触发精确run-owned停止和`NO_PERFORMANCE_RESULT`。

## 6.风险与完成后检查

主要风险为旧r6路径污染、archive/runtime SHA漂移、checkpoint依赖缺失、roundtrip rank门失败和partial误报完成。完成后必须检查PID/CWD/cmdline/GPU、log增长、五类expected artifact、canonical result/marker、全部SHA、禁用访问标志、run-owned进程归零、GPU释放和SSH/TCP22清理。只回收小型控制证据，不读取或拉回大型IQ/tap/wire内容。

## 7.N607执行结果

r7由唯一N607 runner于2026-08-01执行。direct preflight通过；run root创建前为`ABSENT`。四份传输资产、7个关键source entry、2个D104 entry、canonical 22字段fixture、9个绝对路径/SHA绑定、r7 scope、200个Python文件编译和checkpoint/source-pool/salt现场SHA均通过。真实解压路径下的三模型import smoke精确为`baseline_origin_sat_view.py`、`model.py`和`model_dual_cvsincnet.py`；RDCE入口导出`RDCE_RANK=3`、`Z_DIM=160`和`_validated_roundtrip_rdce_rank`，不存在旧`asset.rank`读取。

唯一launch PID为`3065521`。启动3秒后进程存活，CWD和cmdline与r7绝对路径合同完全一致，物理GPU0映射为进程内`cuda:0`。进程随后自然结束，形成`selected_ls_iq`、`strict_tap`、`rdce_asset`、`d106_real_integration_result.json`和`COMPLETED.json`五类完整artifact；日志无异常指纹。

| 完成证据 | SHA256 |
|---|---|
| `d106_real_integration_result.json` | `7b3c2b73a8c7c9bfeab85d99b68f70007996d44451f88b63a529bf7d8fd140cb` |
| `COMPLETED.json` | `2c35385f56b4c596840c4c9420cf4294746c25fc4b5297a2f59315dccb041880` |
| `logs/run.out` | `b0fbf86f8f736067679fbd132a3eea667898c7d60b3b04b2d056a1232179fdbe` |
| `logs/runner_completion_receipt.json` | `e29e26abcfd281513fc0f6acdffba7e585825e2faa6270667693e1993f73df94` |
| `logs/sha256_manifest.txt` | `509164493f82d7fb616c0c5b24fa4be0a07eb8196c942dbf133e7ec5ad7604ac` |

result与marker均为canonical JSON，marker对result SHA绑定通过；7项artifact SHA、RDCE wire反序列化roundtrip、asset receipt与binding digest均闭合，`rdce_rank=3`。`source_held_truth_access`、`formal_query_access`、`target_access`和`performance_metrics_computed`均为false，result中无性能字段。这是真实Phase1 aggregate的技术闭环，不是Target25、125矩阵或性能结论。

16份小型证据已回收到`automation_reports/CV-SincNet/d106_rdce_gtsm_20260801_r1/artifacts/remote_dba10236_r7/`。大型IQ、tap和wire仅保留在原远端run中，未拉回或用于性能解读。最终run-owned进程为0，GPU0释放，本地`ssh.exe`及N607/bridge TCP22连接均为NONE。
