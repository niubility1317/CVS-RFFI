# Task 1报告：冻结FCR-V2类型契约和回归测试

日期：2026-09-03
工作树：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\adv3b02-fcr-20260901`
状态：待提交

## 改动

- `code/cvsrffi/phase1_fcr_types.py`
  - 新增`FCRV2Metadata`，冻结V2 metadata字段、`eta_schema_version`、`eta`和`eta_valid_mask`的最小契约。
  - 新增`FCRV2Metadata.from_mapping(...)`，对缺字段、batch维度不一致、未知schema和valid eta非有限值做fail-closed校验。
  - 新增`FCRV2Metadata.flip_batch()`，冻结后续pairing测试依赖的批次翻转接口。
  - 新增`FCRV2FactorOutput`、`FCRV2CapabilityState`和`FCRV2LossOutput`，并保留V1 dataclass不变。
- `code/tests/test_phase1_fcr_v2_contracts.py`
  - 先写红灯契约测试，覆盖metadata成功路径、shape mismatch fail-closed、unknown schema fail-closed、factor decoder输入合同、capability reasons和loss active/weight合同。
  - `cross_decode`测试按brief要求保留为显式skip，等待后续Task 4/5落地，不让默认套件无条件失败。

## 测试命令与结果

1. `conda run -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_v2_contracts.py -q`
   - 首次结果：`6 failed, 1 skipped`
   - 失败原因：`phase1_fcr_types.py`中缺少`FCRV2Metadata`、`FCRV2FactorOutput`、`FCRV2CapabilityState`和`FCRV2LossOutput`
2. `conda run -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_v2_contracts.py -q`
   - 最终结果：`6 passed, 1 skipped`
3. `conda run -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_transplant.py code/tests/test_phase1_fcr_cross_losses.py -q`
   - 最终结果：`15 passed`

## 提交与远端核对

- 代码提交：`d162cb063a8070bd310a5c6b1e6dadd421b22303`（`test: freeze FCR-V2 contracts`）
- 代码Push：`VERIFIED`
- 代码远端OID核对：`origin/codex/adv3b02-fcr-r1r8-s392005-20260903 = d162cb063a8070bd310a5c6b1e6dadd421b22303`
- 报告提交：本文件以单独提交纳入分支；为避免Git提交对象自引用递归，本报告不在正文中写入“包含本报告自身内容的最终报告提交哈希”，该哈希以后续`git log`和分支头OID为准。

## 遗留关注

- `cross_decode(source, destination, decoder)`仍未在V2 loss/schedule路径落地；本任务只把它固定为后续任务必须满足的显式契约，不在本任务内实现。
- `phase1_fcr_types.py`工作树已提示LF/CRLF转换警告；本次未改仓库换行策略，只保留Git当前行为。
