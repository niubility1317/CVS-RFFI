# D32训练期内生安全cap追溯表

|ID|要求|落地|状态|证据|
|---|---|---|---|---|
|D32-01|单一LEO_weak IQ|沿用唯一密封support；z160/FFT96/RF32为同一IQ确定性描述|implemented|support audit待run回填|
|D32-02|域适应与注册同等重要|15步Stage2-B+10/15步Stage2-C，同run before/after|verified|72项相邻测试|
|D32-03|训练部署同分数面|step0及每步重算非正安全cap，部署保存同一bias|verified|核心单测与72状态压力测试|
|D32-04|floor与遗忘|group-balanced CE、top20% CVaR、有限bias恢复、逐步回滚与support checkpoint|verified|方法锁单测|
|D32-05|轻量部署|参数峰值≤2,016，总25/30步，无dense query图|verified|resource audit单测|
|D32-06|无query/Oracle|query=0，逐样本全注册类，无角色/配额/全局分配|implemented|runner/receipt待run回填|
|D32-07|K1与多新类|K1质心+cap零更新；2/5/10/20新类压力覆盖|verified|72状态压力测试|
|D32-08|int8原型|固定medoid只做旧类内部rerank；完整bundle与slim投影双口径|implemented|full resource待run回填|
|D32-09|自动化证据|candidate lock v10、selection/receipt统一positive helper、90行矩阵|verified|runner定向测试28项|
|D32-10|实验结果|逐类、场景、floor、trace、资源、artifact闭环|pending|N607未启动|

D32是回顾后的第3轮。完成后在启动D33前必须重新阅读目标与`项目.md`、复查D30-D32完整日志并记录回顾。
