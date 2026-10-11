# 可辨识特征几何版多解耦：逐项实施追踪

源报告：用户 2026-10-11 附件 `26b79d33-3ac9-44c9-a123-2c1b376115e3/已粘贴的文本.txt`，保存于同目录 `CVS_STAGE2_GEOMETRY_USER_REPORT_20261011.md`。本版按报告先完成源域审计，再执行 B0—B3 三 seed E40。尚未启动新训练，不把计划写成结果。

用户最新要求“按照报告，开始追求目标”包含报告第九节的训练顺序。沿用对应纯 CE E200 父权重、40 epoch/8,880 更新、三 seed、每 GPU 至多 4 总进程、自动 clean-only 独立评分；不是为全部未来实验取消 practical。

|ID|报告章节|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|G01|一、九|真实分类路径的归一化 joint160，含 PA/reference，不取349或部分 feat_cls|features.py|pending|实际父模型 logits 复算、梯度检查|不可更换表示接口|
|G02|二|核验共轭—延迟歧义、重复不增秩、L 有界作用与 CFO 混杂|physics_audit.py|implemented; local verified; source pending|模板代数及有界算子检查|不宣称纯硬件参数|
|G03|三|RX/day 等权 TX 散度及至多5秩边界|geometry_audit.py|implemented; local verified; source pending|合成均衡样例、全L统计|不硬压身份表示|
|G04|三|单位明确的L/T局部响应、Omega、C_L/C_T|geometry_audit.py|implemented; local verified; source pending|有限差分尺度与收敛、全L响应|记录近似而非假称解析Jacobian|
|G05|三、六|只用留出可信R构成C_R，加C_res/ridge，求广义谱|geometry_audit.py|implemented; local verified; source pending|数值正定、广义残差、源复现|C_res不是热噪声|
|G06|三|各RX/day复现、身份/扰动重合、分类头利用方向|geometry_audit.py|implemented; local verified; source pending|固定源留出分析|源V不拟合|
|G07|四|同包射影时间响应，分段证据与边界失效|physics_audit.py|implemented; local verified; source pending|共同复增益不变、全L/RX/day|扩展段不能默认STF|
|G08|四|非线性先高采样/连续展开再限带采样，残差有效秩|physics_audit.py|implemented; local verified; source pending|共线比17/8、秩与条件数|不能输出不可辨识PA系数|
|G09|四|可重复能量/创新、同步泄漏、相对joint额外信息|physics_audit.py,geometry_audit.py|implemented; local verified; source pending|源条件留出读出/残差统计|负估计不截零|
|G10|五|精确单侧有限margin公式，m_req固定|objectives.py|implemented; local verified; source pending|零动作零梯度、任务完成停止|m_req=log(95)，epsilon=.05；不表示校准保证|
|G11|五|有限父keep、主CE与确定辅助前向分离|objectives.py,source.py|pending|初始keep零、真实动作身份梯度|无额外无截断aug CE|
|G12|五、八|固定覆盖与条件动作，同边界同视图预算|actions.py|implemented; local verified; source pending|实际参数分布、边界/风险、随机流隔离|动作准备维持原margin启发式，冻结|
|G13|五|辅助冻结，不默认在线刷新或min-max|source.py|pending|resolved、参数不变核验|每4步32样本|
|G14|六|三折rank≤2岭差分，仅fit TX标准化和回归|receiver.py|implemented; local verified; source pending|heldout不参与fit、低秩恢复|ridge=.1固定，不搜索|
|G15|六|完整向量可靠度，原始/子空间误差谱|receiver.py|implemented; local verified; source pending|旋转不变、零预测边界|无坐标c、无正下限|
|G16|六|完整160维组差目标、完整父特征cache|receiver.py,source.py|pending|旋转逃逸反例、父初始化|alpha≤.25按确定可行性规则|
|G17|六|控制变量及无放回方差迹修正，保留反向与负值|receiver.py|implemented; local verified; source pending|穷举损失与梯度期望|不detach、不clamp|
|G18|六|块内统一缩放组势、环路一致和单位球可行性|receiver.py|implemented; local verified; source pending|环路反例、目标组范数|不逐边裁剪|
|G19|七|同动作参数/相噪区分次序差、边界和混合曲率|geometry_audit.py|implemented; local verified; source pending|固定参数的IQ与表示比较|不重生成state策略|
|G20|八|切向特征增强先测绝对/相对线性误差|geometry_audit.py|implemented; local verified; source pending|真实动作响应核验|本轮主矩阵不加虚拟特征训练|
|G21|八|共同主CE/关系流，袋内物理ID无放回|source.py,receiver.py|pending|跨arm索引一致|增强视图不算独立样本|
|G22|八|固定有限权重，不按饱和CE比值自动定权|design.py,source.py|pending|配置/实际raw weighted梯度|keep=.1,LT=1,R=1；444步warmup|
|G23|九|先全三seed源审计，随后B0/B1/B2/B3×3，E40|design.py,dispatch.py|pending|预算、路径与条件执行顺序|主矩阵12行|
|G24|九|对应合规父权重/无U/EMA/satellite，保留身份结构|source.py|pending|来源、实际分支、匹配初始输出|源准备成本另计|
|G25|九|每行冻结E40后立即自动clean预测/独立truth-last|evaluate.py,dispatch.py|pending|完整prediction、独立评分|无目标回流|
|G26|九|日志、成本、每seed与配对结果、负结果留存|source.py,report.py|pending|全日志汇总/测试完整矩阵|不把源V当测试|
|G27|九、十|有额外源证据才进入新物理特征身份路径|后续独立变体|deferred|先完成G07—G09|报告明确条件后续；零初始化残差与容量CE对照不可漏|
|G28|七、八、九|交互网络/刷新/二阶结构/Fishr按具体失败证据分开研究|后续独立变体|deferred|先完成本轮主线|不默认叠加|
|G29|十|未来激励/新观测不在现有数据契约中伪造|报告|verified|当前未更改数据|未来方向，不是本轮采集授权|

本表中的工程数值是在目标访问前依据报告建议和现有训练底座一次性确定，并非已有最优结论。新证据若要求调整，先修订尚未启动的设计及记录；健康训练不热改。
