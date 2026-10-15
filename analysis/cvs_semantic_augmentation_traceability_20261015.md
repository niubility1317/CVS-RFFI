# 最新双增益主线：真实标签跨视图身份排序

来源为用户31cc8a22附件。当前代码／本地管线已实现，源校准及r01发布已执行；D0健康训练，D3/D4首步设备错误后退出。HQ源反事实不是此方案训练结果；旧八位置族失败不改写。配置与矩阵见 experiments/cvs_urban_semantic50/registration.json 和对应自动化报告。

|ID|需求与实现位置|状态|证据／未完成|
|---|---|---|---|
|S01|固定SIG_TIME／默认头，原联合训练|verified|实际模型和CE梯度测试，863730/195041，W正常CE更新|
|S02|实际物理5580/720折及两个旧控制来源|verified|真实A0/A-J默认头CP及原fold/provenance smoke PASS，无query|
|S03|D0保留旧U/EMA并关闭training satellite|verified|实际args及native D0管线测试|
|S04|三场景真实residual及正式IQ保持|verified|无改动原引擎／config/seed；新channel薄封装，既有物理检查复用|
|S05|只真实L标签，同TX不作负样本|verified|objective mask/隐藏U负例通过|
|S06|正负参照同RX/day、anchor跨RX|verified|逐triplet条件测试|
|S07|margin满足停止、有界与每anchor归一化|verified|.1 hinge、最大2.1、无参考／满足即0测试|
|S08|两端anchor梯度，D4为两个方向半和|verified|单向clean无梯度、双向两端非0测试|
|S09|额外W/gain为0，CE仍联合训练|verified|真实joint/native D0/D3/D4图测试|
|S10|相同source epoch5、全局0.1强度规则|verified|实际720L_dev/2160前向/0更新，D3 .6863719838，D4 1.1577435581；状态和grad未变|
|S11|E50/11100、旧U和额外EMA前向匹配|pending|配置／早期管线测试通过，待完整真实训练计数|
|S12|3新模型＋2精确旧控制，禁止重复训练测试|verified|五条件配置及直接predict/preflight拒绝复用row测试|
|S13|逐row原头七视图＋CPUtruth-last|pending|沿原实现自动接线，待冻结预测和独立评分|
|S14|clean对D0、low对D1分别报告及跨seed|pending|报告／源条件预定；目前无新性能或多seed证据|
|S15|实际梯度、clip、更新、分支信息与使用量归因|pending|训练记录已接；完整冻结机制与图待产物|
|S16|Bridge-CE/S同成本局部教学|deferred|HQ物理源诊断已完成；按最新顺序留待语义证据后|
|S17|独立跨RX教师库／学生KD|deferred|两教师core草稿独立保留，未发布；不并入语义损失|
|S18|三方共享GPU锁、所有owner、不可覆盖与读回|pending|接线已写，待真实发布前后容量／PID／GPU／日志证据|

计数：verified11、pending5、deferred2、rejected0、blocked0。6项相关测试PASS（6.793s），compile PASS；一次独立P0/P1审查及原问题定向复核 PASS。D0 显式关闭 labeled generic augmentor，回归实际 native post-concat 路径，原 U 强视图保持。该计数是实现阶段覆盖，不是训练完成或用户批准。最高风险仍是相对身份目标能否改善固定默认头的clean与low，不能从正负关系正确或局部梯度非0推导双增益。

历史对照核查：local-order是full-L clean的t1/t2额外cohort，soft-relation是EMA软图且源梯度条件FAIL。本轮是相同内部折的真实标签clean/practical双向joint160排序，不重复其已完成训练，也不声称身份排序是新提出的通用方法。

实际r01 e8f51edab：archive/compile/真实CP smoke PASS。D3/D4首步CUDA标签与CPU元数据mask技术失败；本地修复前复现、后7测试PASS6.435s，设备问题定向P1复核PASS。D0仍健康，不热改；修复尚未发布，新run仅恢复失败两行，旧D0和D1/D2复用。不得把r01标为完整训练或正式性能完成。
