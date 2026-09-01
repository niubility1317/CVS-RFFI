# ADV3B02-FCR Task2报告

- Status:`DONE_WITH_CONCERNS`

## 修改文件

- `code/dataset_wisig.py`：为每个WiSig样本增加物理ID、crop offset和label可见性；U_s返回`y=-1`并删除所有可逆TX元数据，改用仅用于同步的HMAC不透明ID。
- `code/baseline_origin_sat_view.py`：`transform(...,batch_meta=None)`保持旧调用兼容；传入批元数据时LEO视图继承clean物理ID和crop offset。
- `code/cvsrffi/tensors.py`：增加不改变旧batch形状的`extract_batch_meta(...)`。
- `code/cvsrffi/phase1_fcr_interventions.py`：实现元数据清理、物理ID、严格三轴索引、能力状态和`-1`失效索引。
- `code/scripts/audit_phase1_fcr_interventions.py`：实现只读JSON元数据审计。
- `code/tests/test_phase1_fcr_pairing.py`、`code/tests/test_phase1_fcr_interventions.py`：增加同步、权限、严格pair和审计测试。
- `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`：更新FCR-01、FCR-13和FCR-20。

## TDD证据

### 红测

1. `conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_pairing.py -v`
   - 预期失败：U_s元数据同时泄漏`true_tx_i`、`tx_i`和`tx`；FCR配对模块不存在。
2. `conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_pairing.py code/tests/test_phase1_fcr_interventions.py -v`
   - 预期失败：同一TX泄漏仍存在，`cvsrffi.phase1_fcr_interventions`尚不存在。
3. `conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_interventions.py::test_read_only_audit_reports_missing_strict_capabilities -v`
   - 预期失败：审计脚本文件不存在。

### 绿测与编译

- `conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_pairing.py code/tests/test_phase1_fcr_interventions.py code/tests/test_baseline_origin_sat_view.py -v`
  - 结果：`14 passed`。
- `conda run --no-capture-output -n ssr-gpu python -m py_compile code/dataset_wisig.py code/baseline_origin_sat_view.py code/cvsrffi/tensors.py code/cvsrffi/phase1_fcr_interventions.py code/scripts/audit_phase1_fcr_interventions.py`
  - 结果：exit0。
- `git diff --check`
  - 结果：通过。

## 兼容性与权限边界

- 未传`batch_meta`的卫星视图继续返回旧字段和旧语义；既有`test_baseline_origin_sat_view.py`8项通过。
- 传入元数据时LEO只继承clean的`physical_sample_id`和`crop_offset`，不进行第二次裁剪；同步测试通过。
- U_s元数据返回`y=-1`、`label_visible=false`，不含`true_tx_i`、`tx_i`、`tx`或`base_index`；其同步ID是不随元数据暴露密钥的HMAC摘要。严格Fingerprint Pair只使用`label_mask=true`的可见标签，绝不从U_s读取TX真值。

## Pair能力与只读审计

- Nuisance Pair：已实现且由同位置clean/已生成LEO、相同物理ID和相同crop严格验证。
- Content Pair：仅在同一`content_record_id`的不同有效crop窗口中成立；缺字段或无第二窗口时返回`-1/false`，无随机替代。
- Fingerprint Pair：仅在显式共同前导、receiver、day、view/link条件、excitation bin一致且可见TX不同的情况下成立；不满足时返回`-1/false`。
- 审计夹具已验证：无共同前导和第二窗口时写入`common_preamble_configured=false`、0个候选及明确原因，且以exit0结束。
- Live read-only audit：未运行。仅发现若干实验配置JSON，未发现可只读、无需猜测的WiSig元数据索引或公共前导配置；真实Content/Fingerprint能力因此未测量。

## 追踪与自检

- FCR-01=`verified`；FCR-20=`implemented`，限已验证的U_s元数据边界；FCR-13=`blocked`，原因是无本地真实能力输入，不能从合成夹具晋级。
- 自检未发现随机、最近邻或标签依赖fallback；无额外独立P0/P1审查或额外gate。

## Git闭合

- Commit OID：本报告随Task2交付提交，权威OID在提交后最终交接中独立读取，避免报告内容与其自身Git对象ID形成自引用。
- Push结果和remote OID readback：在提交后记录于最终交接。

## 后续接口

- Task7/Task8/Task10可消费`FCRPairBatch`、`InterventionCubeBatchBuilder.build(...)`、`pair_valid_mask`和`InterventionCapability`；必须把无效pair保留为`-1/false`。
- 只有提供已登记的真实WiSig索引和共同前导/窗口元数据后，才可运行只读审计并重新测量FCR-13；不得猜测路径或以随机异TX配对代替。
