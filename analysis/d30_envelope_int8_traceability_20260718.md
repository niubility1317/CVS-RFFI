# D30双包络int8组内校准追溯表

|ID|目标/协议要求|D30实现要求|当前状态|证据|
|---|---|---|---|---|
|D30-01|单IQ LEO_weak|每个physical sample只有一个已叠加LEO_weak观测；z160/FFT96/RF32仅为同一IQ确定性拼接描述|complete_code|support audit与geometry audit已串接|
|D30-02|域适应与注册同等重要|B3拼接几何+D27-B旧类适应；DALI旧类组内重排；max-new新类注册校准；同run记录注册前后|complete_code|90行runner与before/after字段|
|D30-03|无跨组代价的floor优化|逐样本精确保持`max_old`、`max_new`，只允许old-old/new-new变化|complete_code|58回归+2,000-state压力测试|
|D30-04|int8真实使用|不可变Phase1旧类聚合原型真实进入old-old重排，不更新、不落盘反量化bank|complete_code|DALI resource/runtime audit|
|D30-05|K1与统一K-shot|K1精确旁路；K2~4 fail closed；K>=5 shot-rank OOF；开发K10一次选参|complete_code|单测、candidate lock v8|
|D30-06|query=test only|CLI/API无query truth/role/quota输入；query 0行；逐样本全注册类argmax|complete_code|support audit、receipt、接口检查|
|D30-07|clean/source不可达|只复用已密封support与授权int8模型知识；不读取clean/source/衍生信号|complete_code|protocol contract与runtime字段|
|D30-08|极轻资源|峰值≤80k参数、≤30epoch、≤50step、≤256KB；无dense query图|complete_code|fold/full resource与58测试|
|D30-09|floor与可达上界|逐类输出组间/组内错误及max-envelope可达上界，明确哪些floor可被组内校准修复|complete_code|校准前后confusion artifact|
|D30-10|完整证据|合法TX/receiver/support清单、完整日志、逐类/逐receiver、资源、自动报告、Git提交|complete_development_negative|90行、artifact SHA、完整报告、Git；正式独立矩阵仍pending|
