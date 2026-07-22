# D4a单观测独立模块实现Traceability

日期：2026-07-17
范围：只新增独立模块与最窄单元测试；不接入或修改现有runner
设计源：`analysis/d4a_single_observation_floorlock_design_20260717.md`

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|D4M-01|设计§2.2|只接受已接收的单一LEO_weak IQ约定`float32 [N,2,L]`或其固定二维特征，不生成额外LEO状态|`code/cvsrffi/stage2_single_observation_floorlock.py`|verified|`py_compile`、focused pytest|独立模块不读取dataset、clean或source|
|D4M-02|设计§2.2|为base/plus/minus计算view输出`parent_received_iq_sha256/operator_id/view_seed`lineage，并声明view不增加K|同上|verified|`test_received_iq_hash_and_lineage_views_share_one_physical_observation`|plus/minus由固定feature确定性派生|
|D4M-03|设计§4|仅用registered support闭式拟合class-balanced对角能量等化状态|同上|verified|`test_support_only_equalizer_audits_parameters_and_loo_floor`|不实现完整D4a分类head|
|D4M-04|设计§4.1–4.3|输出leave-one-physical-sample-out floor统计；K1使用leave-one-view-out|同上|verified|K≥2与K1聚焦测试|统计含overall、逐类accuracy与margin floor|
|D4M-05|设计§2.3|query API为inference-only，不接收标签且不更新状态|同上|verified|signature/state immutability/batch-local test|无query fit或自适应|
|D4M-06|设计§7|显式审计可拟合参数并fail closed保证`<=80,000`|同上|verified|正常参数与80,001维拒绝测试|当前仅D维log-scale|
|D4M-07|用户要求|增加最窄单元测试且不修改现有runner|`tests/test_stage2_single_observation_floorlock.py`|verified|`5 passed`、`py_compile`|只新增一个测试文件|
|D4M-08|交付纪律|不提交Git，检查仅预期新增文件|本traceability及上述文件|verified|`git status`、no-index `diff --check`|现有工作树其它改动不属于本任务|

## 明确不在本次范围

- Stage2-B/C完整训练循环、新类注册head和guard offset；
- package builder、pre-open validator、prediction artifact、scorer或runner接线；
- N607同步与实验；
- 性能达标声明。

因此本次实现是D4a的独立接收后表示与LOO floor统计原语，不是完整D4a设计的严格全链路落地。
