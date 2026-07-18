# D55中心化LOO难度补偿追踪

|需求|实现|验证|状态|
|---|---|---|---|
|D46底座|系数完全继承D46|D46＋D55 23项|VERIFIED_PRE_RUN|
|类难度|`d_c=sum_g w_gc CE_gc`|数值重算、类置换|VERIFIED_PRE_RUN|
|中心化截距|`Delta b=d-mean(d)`|补偿和为0|VERIFIED_PRE_RUN|
|K1/K2|证据读取前精确D46 fallback|参数化测试|VERIFIED_PRE_RUN|
|无调参/角色/query|无alpha/threshold/clip/branch|源码audit|VERIFIED_PRE_RUN|
|完整性能|105行完成后解析|`full_performance_summary.json`＋报告第5—13节|VERIFIED_COMPLETE|
|协议闭合|query/source/clean/role/quota均禁用|receipt/resource audit|VERIFIED_COMPLETE|
|性能判定|after70.56%、new69.33%、H68.46%、forget12.78pp|与D46同折15/15变化|DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE|

D55为support-only开发探针，不具有formal/125权限。完整结果证明raw LOO-CE不能直接作为判别logit截距；该路线停止，不做alpha/clip/threshold扫描。
