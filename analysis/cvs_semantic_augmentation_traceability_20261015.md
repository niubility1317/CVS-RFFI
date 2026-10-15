# 最新双增益主线：真实标签跨视图身份排序

来源为用户31cc8a22附件。当前修复后的r02三条训练已发布并保持运行；原r01失败记录和不匹配D0诊断保留。HQ源反事实不是此方案训练结果；旧八位置族失败不改写。配置与矩阵见 experiments/cvs_urban_semantic50/registration.json 和对应自动化报告。

|ID|需求与实现位置|状态|证据／未完成|
|---|---|---|---|
|S01|固定SIG_TIME／默认头，原联合训练|verified|实际模型和CE梯度测试，863730/195041，W正常CE更新|
|S02|实际物理5580/720折及两个旧控制来源|verified|真实A0/A-J默认头CP及原fold/provenance smoke PASS，无query|
|S03|D0保留旧U/EMA和原通用增强，仅关闭training satellite|verified|r02 actual post-augmentor L capture／CE／EMA测试；r01 no-generic D0仅独立诊断|
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
|S15|实际梯度、clip、更新、分支信息与使用量归因|pending|r02实际非零关系梯度/W-gain额外0已读回；独立冻结源probe已实现/5检查及定向P1复核PASS，完整冻结机制与图待实际产物|
|S16|Bridge-CE/S同成本局部教学|deferred|HQ物理源诊断已完成；按最新顺序留待语义证据后|
|S17|独立跨RX教师库／学生KD|deferred|两教师core草稿独立保留，未发布；不并入语义损失|
|S18|三方共享GPU锁、所有owner、不可覆盖与读回|verified|r02真实发布63f6bd461、实际GPU1/2/3；独立PID/CWD/argv/日志增长，无健康任务修改|

计数：verified12、pending4、deferred2、rejected0、blocked0。6项相关测试PASS（6.793s），compile PASS；一次独立P0/P1审查及原问题定向复核 PASS。D0 显式关闭 labeled generic augmentor，回归实际 native post-concat 路径，原 U 强视图保持。该计数是实现阶段覆盖，不是训练完成或用户批准。最高风险仍是相对身份目标能否改善固定默认头的clean与low，不能从正负关系正确或局部梯度非0推导双增益。

历史对照核查：local-order是full-L clean的t1/t2额外cohort，soft-relation是EMA软图且源梯度条件FAIL。本轮是相同内部折的真实标签clean/practical双向joint160排序，不重复其已完成训练，也不声称身份排序是新提出的通用方法。

实际r01 e8f51edab：archive/compile/真实CP smoke PASS。D3/D4首步CUDA标签与CPU元数据mask技术失败；本地修复前复现、后7测试PASS6.435s，设备问题定向P1复核PASS。D0仍健康，不热改；修复尚未发布，新run仅恢复失败两行，旧D0和D1/D2复用。不得把r01标为完整训练或正式性能完成。

恢复版 r02：新scratch匹配D0/D3/D4，D1/D2复用。原r01 D0关闭通用增强与concat控制不匹配，保留健康诊断但排除主比较，额外E50成本另列。新版保留native generic前后位置，仅捕获实际L输入并在U/EMA前移除hook。8检查PASS6.717s；源calibration从原r01原样复制，新前向0。r02尚未发布。

恢复差异独立P0/P1审查PASS；D0新增匹配问题的定向复核PASS。无额外全量或报告审查。当前最高风险仍是原头的实际双增益，须等待新E50冻结证据。

r02实际发布63f6bd461：archive/compile/真实CP smoke/源校准原样复用通过；三row独立活跃、D0/D3/D4 epoch4/2/2，日志增长。actual use_aug=True匹配确认。原r01 D0仍独立诊断。当前verified12/pending4/deferred2，缺项为完整预算、七视图、双增益及机制/跨seed；保持目标active。

S15冻结源机制实现计划：`experiments/cvs_urban_semantic_mechanism/`，严格加载五条件final50/默认头，固定D1/A0 clean正确集合和一份共同source practical视图，记录错误修复/破坏、原头间隔、状态/RX/质量、实际分支/门控及同RX/day参照相对身份差。mid/low的零更新配对梯度使用同一次clean+urban实际joint图，D4 clean anchor不被无意detach；全部eval、autograd.grad不累积.grad。预算36722前向、L_dev720、U不开/V前向0；新增机制诊断须登记后在模型冻结时独立CPU发布，不热改训练或读取target。当前pending，不能由实现替代实际结果。

S11/S15完整日志与出图：新增 `experiments/cvs_urban_semantic50/_audit_worker.py`、`full_log_audit.py` 和 `experiments/cvs_urban_semantic_mechanism/figures.py`。新三行逐条读取全11100步及50轮native/更新记录；旧D1/D2复用已验证八位置完整审计，不重复分析未变化日志。图只从完整真实artifact生成，当前无数据图；五条件全列，不按目标排序或挑最佳。

完整日志reader的3项聚焦检查PASS0.066s（全步平均/clip计数、缺步/非有限/隐藏标签负例、矩阵未完成不读行或目标分数）；两CLI --help与compile通过。本地测试文件初次写偏目录已移回本工作树，未影响实验。图函数尚无真实结果可渲染，不声称图验证完成。实际正确D0已自然E50/11100并完成七视图评分，2620.404159s；D3/D4最新34/33，S11/S13/S15仍等待完整矩阵证据。
