# Phase1 export-only在线几何聚合可追溯记录

日期：2026-07-17

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|PGS-001|用户要求|第一遍仅在线累计归一化`z_id`的域×类sum/count|`code/cvsrffi/phase1_geometry_streaming.py`|verified|中心解析测试通过|不保存样本级feature|
|PGS-002|用户要求|冻结最终centroid后提供第二遍P90接口|同上|verified|解析角度测试通过|P90定义为cosine distance|
|PGS-003|用户要求|P90计算使用有界内存|同上|verified|固定直方图与内存尺寸断言通过|内存只依赖`D×C×bins`|
|PGS-004|用户要求|严格shape、finite、单位范数、索引、cell coverage和两遍一致性检查|同上|verified|参数化负向测试通过|第二遍同时核对count与聚合sum|
|PGS-005|用户要求|最终只返回供codec消费的full-precision内存对象，无文件写入或样本级接口|同上|verified|API与无sample字段测试通过|count仅为Phase1内存审计，不进入bundle|
|PGS-006|用户要求|不修改`train_ssdg.py`、v2 codec或远程文件|Git diff范围|verified|`git diff --check`通过|本模块独立实现|

## P90近似

P90取固定直方图中经验nearest-rank P90所在bin的上沿，因此为保守上界；余弦距离误差确定性不超过`2/radius_histogram_bins`。默认4096 bins时上界为0.00048828125。直方图本身不序列化。

## 验证记录

- `conda run -n ssr-gpu python -m pytest -q tests/test_phase1_geometry_streaming.py tests/test_phase1_center_lowrank_prototype_bundle.py`：21项PASS。
- `py_compile`与`git diff --check`：通过。
- 追溯状态：`verified=6`、`deferred=0`、`rejected=0`、`blocked=0`。
