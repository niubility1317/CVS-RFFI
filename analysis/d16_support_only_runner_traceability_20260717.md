# D16-FCAR strict-K10 enrollment-only runner追踪

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|D16R-01|`项目.md` 7.1/7.1.1|只接受外部SHA固定的before/after `enrollment_only`密封包，并复用既有D14 pre-open与strict-K10 loader|`code/scripts/run_d16_support_only_fcar.py`|verified|源码复用边界测试；真实包pre-open及外部seal SHA通过|不复制package安全业务规则|
|D16R-02|`项目.md` 7.1.1|三场景`physical_sample_id`与`parent_received_iq_sha256`并集两两不交；Before旧类在After中精确复用|同上|verified|互斥正反单测；真实三组pair的ID/SHA overlap均为0|每个物理support只对应一个固定LEO_weak IQ|
|D16R-03|Phase2无Oracle边界|runner无query/truth/prediction/scorer输入参数或loader；只做support-only development selection|同上、`tests/test_run_d16_support_only_fcar.py`|verified|CLI源码测试；receipt记录所有query/scorer open均为false|正式query绝不在本runner开放|
|D16R-04|D16统一候选|仅比较true Z0与`margin_band=.02/.04`；positive arm固定rank8/shrink.5/ridge.01及FCAR固定幅度grid|同上|verified|候选锁确定性测试；真实lock SHA=`0d4c65e13575cfb8e600a4b854230bac6d24b481295e6d5f092734a67bb43041`|同一候选用于三个LEO场景|
|D16R-05|旧类与新类同等优先|调用`stage2_fcar.evaluate_joint_leave_two_out`并报告Before old、After old、seen-new、joint、H、forgetting、逐类和vs-Z0|同上|verified|真实3 candidate×3 scenario×5 fold完整审计|outer held2不得参与拟合|
|D16R-06|floor优化|分别汇总old/new floor handles与floor accuracy，并纳入三场景正arm门|同上|verified|严格floor门单测；真实每foldfloor审计与失败原因完整保存|floor由FCAR support OOF自动识别，不硬编码TX|
|D16R-07|避免旧类遗忘|正arm要求Before/After/new逐类不劣于Z0、After old逐类不劣于Before、old score逐位锁、joint/H不劣于Z0|同上|verified|防fold平均抵消测试；真实所有15 fold分别判门|任何场景失败即不能统一选择|
|D16R-08|true zero回退|所有positive arm失败时选择rank0/force-zero Z0，且仍不开放query|同上|verified|选择单测；真实状态=`SUPPORT_ONLY_D16_DEVELOPMENT_TRUE_Z0_NO_QUERY_OPEN`|Z0不算D16性能提升|
|D16R-09|完整证据|输出JSONL完整trace、JSON审计、Markdown报告、候选锁、逐场景资源、enabled/幅度与输入provenance摘要|同上|verified|真实artifact 145/145条JSONL可解析；receipt绑定training/audit/report SHA|authority固定development diagnostic|
|D16R-10|轻量部署审计|报告0参数、0epoch、state bytes、head ops、单support-row相对identity single-qKNN MAC/延迟/状态|同上|verified|真实Z0 state 12,859B；3168 MAC vs 31,680；0参数/0epoch/无dense图|包装层延迟仅作support-row诊断，不作正式query结论|

## 验证记录

- `conda run -n ssr-gpu python -m py_compile code\scripts\run_d16_support_only_fcar.py tests\test_run_d16_support_only_fcar.py`
- `conda run -n ssr-gpu python -m pytest -q tests\test_stage2_fcar.py tests\test_run_d16_support_only_fcar.py`：`19 passed`
- `conda run -n ssr-gpu python code\scripts\run_d16_support_only_fcar.py --help`
- `git diff --check -- analysis\d16_support_only_runner_traceability_20260717.md code\scripts\run_d16_support_only_fcar.py tests\test_run_d16_support_only_fcar.py`
- 真实artifact：`E:\type10-7\automation_reports\CV-SincNet\d8_second_block_dev_20260717_020200\d16_fcar_strict_k10_v1`
