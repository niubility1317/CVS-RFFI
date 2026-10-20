# R15：保持先验通道的低秩压缩

状态：LOCAL_PREPARED。唯一父模型仍为原R38 dense_half、794529参数、source E20/4940、method e0868f0a0c3cf056a1753a5b4e5ce4f0bff84381。0继承R13/R14已测权重；原7/63/30数据与合法祖先复用，非新独立盲测。

R14全部30轮source日志表明，冻结context使裁剪几乎归零，但窄版末轮teacher-stage IQ NMSE仍为原宽度约17/29倍。该source证据支持进一步检验通道裁剪造成的函数缺口，不证明GN为唯一起因。R15只在prior内对实际节省参数的长卷积做固定截断SVD，保持所有外部通道、GN分组/参数、skip与输出维度。left/right权重均跨三stage共享、各stage bias独立；0IQ校准。原548795参数context逐位冻结。结构与rank不依据F/V分类分数或target结果选择。

|行|完整参数|可训练prior|rank比例|GPU|
|---|---:|---:|---:|---:|
|factor_three_quarter|779745|230950|0.75|2|
|factor_five_eighth|750035|201240|0.625|3|

两行同final-clean IQ NMSE+1×三stage原R38 IQ锚；reset prior-only AdamW1e-4/warm247/cos1e-5/WD1e-4/clip1、B256/micro64真实尾24，received01..10、MODELseed2026092701、DATAseed2026102001。固定E10/2470，L63000/U567000曝光、teacher/studentprior630000、共享context630000/studentcontext0；0TX标签/F权重/F输出/V/target训练。context累计40、prior累计50，component-epoch130、祖先串行pass90，原F50E50另计。

source1800×7仅诊断，每版无论源结果均自动target168000×clean/三suburban/三urban。每版全部预测固定后独立truth-last，原F50/defaulthead不重训。原R17逐场景1pp容差、原R38保持/实际收益、lowurban70/其他五场景距rawclean5pp、净纠错、TX/RX/day、NMSE/FFT/配对clean特征距离分别报告。同预算control复用R14 prior_keep全部既有结果，不重跑。0目标反馈调参/候选筛选/选择性重跑。

部署只当前原生float32[B,2,256] IQ，完整自身context/prior/3×2CG，无teacher、clean、标签、场景或状态更新。完整参数、Conv/Linear MAC（FFT/CG另列）、文件字节、16sourceV/batch1/2warmup实测完整链延迟/显存均报告，不凭因子参数减少断言加速。

GPU2/3各一本任务/max2，R14对应owned句柄终态且实际退出才交接，free>=18GB、全owner<2，其他owner保持；原R06共享锁max1own scorer/CPU2。新不可覆盖run/release；普通预检、一次归档两端SHA、远端compile。launcher首步真实父/F50/2Ls no-query smoke通过立即训练；prior梯度、精确冻结context、teacher/F/head零更新、严格重载输出一致。技术故障保留该行、健康行继续，0自动重试/回退。

REJECTED_EXTRA_GATE：项目八项最小流程覆盖通用设计/计划审批。一次独立P0/P1限于下一真实运行；具体命令、环境、CWD、路径、停止规则及产物见experiment.json。
