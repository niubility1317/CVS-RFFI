# D103-R2需求—实现—测试—artifact追踪

状态：`IMPLEMENTED_LOCAL / RELEASE_REVIEW_GO / PREREGISTERED_INPUTS_BOUND / N607_GPU_STACK_BLOCKED / N607_LAUNCH_NO_GO / TARGET25_NO_GO`

|ID|需求|设计依据|实现文件|验证或artifact|当前状态|
|---|---|---|---|---|---|
|R2-01|全局精确0.07/0.63/0.30正式分离归档|重入卡§3|`code/cvsrffi/rxid_metabias4_source_archive.py`、`code/scripts/export_d103_r1_source_splits.py`|本地真实8400行smoke：588/5292/2520；42组各L=14；leave-day min/max=10/12；定向测试通过|implemented-local-verified|
|R2-02|跨receiver K1/K5/K10元任务|重入卡§2|`code/cvsrffi/rxid_metabias4_phase1_trainer.py`|本地真实特征首step完成；support receiver=`1-1`、query receiver=`1-19`、query4/class、source-val=0|implemented-local-verified|
|R2-03|R1其余Phase1机制保持|重入卡§2|现有trainer|TX-null/MMD/selfsup/VICReg测试|implemented|
|R2-04|M0/D102/D103 matched scorer|重入卡§4|held execution、D102 builder、predictor、truth-side scorer|同support/query；49个L_s fold-specific D102诊断bundle；63行truth-free预测；独立评分；合成全路径测试通过|implemented-local-verified|
|R2-05|4 leave-day实际shift方向门|重入卡§4|held execution、predictor、gate|196个day fit计划；49个K1 outer逐项计算`cos(B_day a_day,B_outer a_outer)`；近零fail closed；测试通过|implemented-local-verified|
|R2-06|K1数值、INT8和双TX probe门|R1卡§5–7；R2卡§4|Stage2、bundle、prepare、falsifier|失败TX探针保留真实诊断预测但拒绝部署序列化；7fold×9capacity TX探针；INT8和K1门测试通过|implemented-local-verified|
|R2-07|246fit/98,400step非覆盖runner|R2卡§5|fit matrix、完整held pipeline|每GPU 1–2 lane；两行同规范化异常指纹停止dispatch；终止前绑定PID/CWD/cmdline/run-root，先定向SIGTERM、后仅对仍存活的已绑定PID升级；保存前后进程树、GPU快照、逐fit日志和资源receipt；本地静态/定向验证通过|implemented-awaiting-release-review|
|R2-08|Target25严格25行单seed|用户当前口径|计划gate后matrix|5receiver×1seed×5slice；held gate前禁止|blocked|

首轮独立release审查结论为`P0=1/P1=4/P2=2`，已逐项修正：K1活动性和INT8一致性只读support，不再读held query；246个fit逐一验证outer身份、输入SHA、访问ledger和teacher数组聚合SHA；D102父method lock和原拒绝receipt按冻结值精确匹配；truth-side scorer在打开truth前创建唯一事件并要求63行共同绑定；异常指纹规范化且停机保存精确进程/GPU证据；磁盘按当前树加16MiB后续分析凭据预留计费。

验证汇总：实现commit=`59978e44`；独立复审index SHA256=`30c8c98ff8fcdf2915f4c2e797c605cecc98d138e95ad3b0bc6e542faf9fdc9b`；`REVIEW_GO / P0=0 / P1=0 / P2=2`。D103定向61项通过，`python -m py_compile`覆盖36个新增/修改Python文件并通过，`git diff --cached --check`通过；其中资源终结器实测按当前run-root加16MiB后续分析凭据预留计费；解析系数与既有D102闭式FP16逐bit一致。当前代码development-only真实checkpoint特征400step无query-truth smoke再次通过（39.125s、K1 support=6、query=354、D103 ACTIVE、未计算性能）。`source_train`cache SHA已只读绑定为`d719808ceaed07c13f6c8d8053acf910a61904243ec1d47c28cc4e4b679cffd2`，三场景和五项冻结输入全匹配，18个远端release目标全部`ABSENT`。但N607内核驱动535.309.01与用户态NVML580.173.02不匹配，`nvidia-smi`exit18，逐卡进程/util/显存不可取信；未sync或启动，保持`N607_GPU_STACK_BLOCKED / N607_LAUNCH_NO_GO / TARGET25_NO_GO`。
