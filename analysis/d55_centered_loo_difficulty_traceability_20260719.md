# D55中心化LOO难度补偿追踪

|需求|实现|验证|状态|
|---|---|---|---|
|D46底座|系数完全继承D46|D46＋D55 23项|VERIFIED_PRE_RUN|
|类难度|`d_c=sum_g w_gc CE_gc`|数值重算、类置换|VERIFIED_PRE_RUN|
|中心化截距|`Delta b=d-mean(d)`|补偿和为0|VERIFIED_PRE_RUN|
|K1/K2|证据读取前精确D46 fallback|参数化测试|VERIFIED_PRE_RUN|
|无调参/角色/query|无alpha/threshold/clip/branch|源码audit|VERIFIED_PRE_RUN|
|完整性能|105行完成后解析|summary/report|PENDING_RUN|

D55为support-only开发探针，不具有formal/125权限。
