# 30份源抽样：本行完整结果，独立配对对照待完成

本行已完成从零单阶段30轮和七景正式测试，七景全部低于默认 DURA；clean较raw clean少360条正确、低0.214286个百分点，六星地0/6通过3pp。默认不变，完整目标未达成。独立纯G repeatViews对照仍在健康训练；本报告先闭合本行，不提前判定额外10份抽样的独立收益。

固定原F50/defaulthead、同一168000物理样本×7景，全部目标预测固定后独立truth-last。126个总体/TX/RX/day记录、MacroF1和净纠错已独立复算一致。完整分组、混淆矩阵、配对IQ误差见results.json。单seed、暴露基准，不称独立盲测。

|场景|准确率%|相对默认（百分点）|距raw clean（百分点）|
|---|---:|---:|---:|
|clean|84.461905|-0.216071|0.214286|
|practical_high|80.363095|-2.461905|4.313095|
|practical_mid|79.179762|-2.380952|5.496429|
|practical_low_suburban|77.792262|-2.266071|6.883929|
|practical_high_urban|79.452976|-2.292857|5.223214|
|practical_mid_urban|73.383333|-2.102976|11.292857|
|practical_low_urban|66.973810|-1.916071|17.702381|

low urban相对默认修复12746条、损伤15965条，净变化-3219条，其平均IQ NMSE为6.982867%。原R17逐景最多下降1pp容忍=True；这与3pp优化目标分别判断，不能平均掩盖失守。

## 输入变化与完整源训练

紧凑G266951从零、全模块第一步联合训练、单AdamW日程E30/7410，B256/L189000U1701000，.001warm247后cos至.00002、WD1e-4、clip1。IQ1/prior.1/STFT.05/action.05，variance仅日志；没有神经教师、F/TX/V/target优化、另训头、学生预训练或优化器重置。方法研究用过源F，实际优化无F不等于整个研发未见F。

原R07缓存01..20只读链接，源clean/physicalIDs/roles/receiver-day/V面板保持。21..30沿用原native生成器、augmentation2026101912/receiver2026101911和原namespace规则生成，不把后10轮映射回01..10。原R07生成器与固定47运行基底相关代码在Git差异检查中未改变，新增变量是抽样namespace。30份仍是63000物理源样本的30次腐蚀抽样，clean分支可重复，不称189万个独立物理样本。

全部30条日志的source_cache_view恰为1..30、更新/曝光/五项加权loss已独立复算。最终IQ NMSE0.022375600、总loss0.027121996、clip0.000%。源V仅固定1800条面板×7景，不是完整V27000，未用于训练或选epoch。在线损失下降不能替代身份收益。

![完整训练、制备成本与七景结果](E:/type10-7/automation_reports/CV-SincNet/20261031-phase1-DURA-freshViews-E30-s2026092701-r01/training_and_formal.png)

## 实际成本与比较边界

新增10份共630000次源模拟访问、4195800000数值字节（约4.20GB，另有NPY头和groups），CPU8制备530.740140秒（8.846分钟）。每份原nativeactionreplay/shape/FP32/finite/固定scene顺序由生成路径完整检查后才准入G训练，旧20份没有重新生成或复制。模型NN教师输入0；native物理信道及clean-independent action探针不是神经教师，二者成本不混记。源缓存后续可复用，但不把本次成本隐藏或当免费知识。

G训练及保存976.312565秒（16.272分钟），控制器起点至全量分析1691.498856秒（28.192分钟），其中七景目标预测159.619987秒。源制备已包含在起点至worker的534.387954秒中，不再次与全流程相加；控制器末尾清理另有5.184999秒。

部署266951参数/1067804参数字节/1305444checkpoint字节，25068072 ConvLinearMAC/2prior4CG27FFT-IFFT，训练峰值allocated1084664832字节。GPU6同卡交替G+F B1：默认5.249518ms、本版5.191909ms；全部B1/B128/P95/吞吐在JSON，共同驻留profile显存不当独立部署显存。没有达到同等质量，不宣称训练效率优化已成功。

旧IQenhance physical对照的G643个初state与纯G一致，但head初始化额外把CUDA探针seed设为2026103107，先前只恢复CPU RNG的检查不完整。旧分数真实、预算匹配；历史vs它的涨跌不是严格同探针的单机制因果比较。当前代码已修复CPU head初始化，N607纯CPU trace复测所有工厂CUDA调用只余2026092701、初state不变，0CUDAinit/G-Fforward/checkpoint/更新/TX-V-target，旧archive不改。

本版主配对对照为20261031-phase1-DURA-repeatViews-E30-s2026092701-r01，独立stock纯G路径/同初始化、CUDAprobe2026092701、npshuffle2026103004、网络/损失/LR/预算，仅21..30输入日程不同；在本版访问目标前已补齐登记。它尚未完成，matched_control_delta暂为null，不比较不完整目标矩阵，不停止或修改其健康训练。跨GPU浮点轨迹不承诺逐位相同。

执行32a32998d3c76105b017a0905300e8edbce64e16；4聚焦检查/唯一P0P1/真实2Ls梯度-严格重载-F noquery、实际PID/CWD/argv/CUDA/GPU/log和自然结束已核实。本结果不回流选epoch、参数或场景，后续对照完成后按预定完整矩阵补齐分析。默认保持、目标ACTIVE。
