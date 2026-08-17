# Stage2-C MRIOR-SDA预适应类增量追踪

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|CI-01|用户请求|CSIL先做MRIOR-SDA目标域预适应再注册|`adv3b02_mrior_preadapt_ci.py`、CI predictor|pending|focused pytest|保持CSIL类增量参数不变|
|CI-02|用户请求|MoPC-HR先做MRIOR-SDA目标域预适应再注册|`adv3b02_mrior_preadapt_ci.py`、CI predictor|pending|focused pytest|保持MoPC-HR类增量参数不变|
|CI-03|公平性|预适应artifact跨方法/new-count复用但绑定同一old support|plan builder、artifact loader|pending|identity negative tests|键不含method/new-count|
|CI-04|项目协议|新类support/query保持正式LEO固定观测|existing package loader、runner|pending|package identity tests|不触发数据重验|
|CI-05|query边界|query不参与MRIOR或类增量训练|predictor、receipts|pending|query-open ordering tests|truth独立评分|
|CI-06|完整矩阵|300 preadapt job、800 CI cell、2400 scene row|plan、runner、analysis|pending|closure tests和N607 artifacts|smoke不是性能证据|
|CI-07|报告|同方法同row比较预适应与v7无预适应|analysis、report|pending|exact join tests|跨方法仅描述性比较|
|CI-08|用户请求|ERTB-IDR无预适应注册反事实|D92独立分支与报告|pending|D92 focused tests|不在本分支实现|
