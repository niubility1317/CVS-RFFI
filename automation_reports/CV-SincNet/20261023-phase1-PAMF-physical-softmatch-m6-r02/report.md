# PAMF 半监督方法：首轮预登记

状态：RUNNING（启动已独立核实）。目标性能尚未完成。

## 问题与验收

完整原生半监督相对匹配监督版未显示明确正收益。新方法须在同一模型上同时满足：三seed raw clean均值相对F20监督版及F21普通PL分别至少+1pp，六practical种子内平均再跨seed的均值相对两对照分别严格+1.5pp。主口径为raw F，不以DURA读出替代。按未四舍五入计数计算，复用对照如下。

```json
{
  "F20": {
    "clean": 84.78531746031746,
    "six_mean": 70.59619708994708,
    "per_seed_clean": [
      84.86309523809524,
      83.92559523809524,
      85.5672619047619
    ],
    "per_seed_six": [
      71.63412698412698,
      69.76388888888889,
      70.3905753968254
    ]
  },
  "F21": {
    "clean": 84.98174603174603,
    "six_mean": 70.35935846560847,
    "per_seed_clean": [
      84.8625,
      84.8827380952381,
      85.2
    ],
    "per_seed_six": [
      71.3561507936508,
      69.44642857142857,
      70.27549603174603
    ]
  }
}
```

## 方法和匹配

PSM：保留T1监督骨架，own encoder EMA0.996；不使用原RX/day分位数、批内相邻观测及强视图argmax筛选，不施加全U熵最小化。teacher clean softmax置信经单边Gaussian连续权重（momentEMA0.999、方差下限1e-4、置信floor0.7），soft target以T0.5 sharpen。U256的一半为ordinary噪声0.015，另一半用与原As相同的source-only Practical残差信道（low suburban/mid urban/low urban随机）；同一物理IQ的teacher类别分布监督两类student视图。U损失以全U均值归一，权重0.5，E14起5epoch ramp。不使用类别配额、uniform balancing、U真值或目标统计。

RXPSM：在PSM上，仅由当步可见L_s的EMA teacher特征更新每RX/类原型，momentum0.9。对每个U排除其自身RX，只在至少两个其他RX覆盖全部六类时计算各RX cosine/T0.1分布并等权平均；否则退回teacher分布。proto分布混合20%，teacher/proto不一致时权重缩到25%而非归零；再加0.05的归一特征原型距离。所有类采用相同公式。原型不接收U伪标签或V更新。

两方法分别按3seed从零20→自身10→自身20，累计E50/11100，E1–13 L-only、E14–50 LU；所有阶段保留原监督成对CE、ordinary/raw KL、As CE/KL0、CRIC、SIG_TIME_X3、LS0.01、CENTER_UNITS和默认头联合训练。优化器/cosine按原三段重置，teacher、Gaussian moments、L-only原型跨自身段继续。旧F20/F21同数据契约、输入绑定已修复的完整来源由load_F核验，仅复用为冻结对照；不作为真实训练初始化。本研究是已暴露基准上的新源域研发，不声称新的独立盲测。

## 执行、验证与产物

唯一矩阵、六类seed、commit、精确命令/CWD/环境、输入输出路径、GPU映射、技术停止规则和artifact计划见同目录experiment.json。GPU0/1/2各固定seed2701/2702/2703，每卡最多2计算worker、全run最多6，另一个独立CPU scorer。所有七景预测冻结后CPU scorer才连接真值。训练进程不接收truth内容；目标分数不回流调参、选模、排序或重跑。

本地9项聚焦检查已通过（含修复前失败的CUDA混合设备回归）：U梯度和own状态序列化、权重边界、隐藏标签拒绝、非sourceRX拒绝、自身RX排除、原型不全退回、类置换等变性、target输入拒绝和固定预算、三seed分卡。发布第一步为实际固定F21 checkpoint的source-only前向，以及两条实际L/U/default-head反向路径；为避免完整13epoch warmup，只有smoke teacher使用该合规固定权重，真实实验仍全部scratch。PASS立即继续；不创建smoke授权artifact。一次独立P0/P1审查无发现；真实GPU smoke暴露的设备放置问题已修复，唯一一次定点复审PASS。

每row训练冻结后先做720个固定V物理样本的clean/Practical混合诊断（不更新任何模型/EMA/原型），再自动预测raw clean+六景；PSM worker补做同seed已有F20/F21的相同V诊断。源域证据用于后续科学诊断；目标读出仅核验预声明收益，不参与下一候选决策。完整step/segment日志与实际U有效曝光、类别权重、物理视图数、训练秒数保留；目标未达到则如实保持未完成。

来源：[SoftMatch](https://arxiv.org/abs/2301.10921)、[SimMatch](https://openaccess.thecvf.com/content/CVPR2022/html/Zheng_SimMatch_Semi-Supervised_Learning_With_Similarity_Matching_CVPR_2022_paper.html)。仅为机制依据，不构成PAMF效果证据。独立默认PAMF与DURA不因本次研究自动替换。


## 已验证启动

r01在正式派发前的设备放置错误已保留，r02为同科学矩阵的技术替代。执行commit为`4633ff7dddb9abe0afce4c0e76b0c0424db38c9c`，本地与upstream OID一致；单归档传输比较、远端编译均通过。启动器真实F21/source L前向和PSM/RXPSM完整L/U/default-head反向检查PASS，随后立即派发6个worker。

独立读回确认GPU0/1/2分别固定种子2026092701/02/03，各2个worker；6个PID/CWD/argv与配置一致、实际GPU分配一致、日志均增长，已完成E1/222更新。首轮时长约83.5至86.2秒，属于共享负载下L-only阶段的观察值，不推算为完整LU或测试耗时。此时U曝光为0符合E14启用设计；后续需继续核实实际权重和曝光，不能用启动smoke代替完整训练的证据。

控制器自动接续每row冻结后的源V诊断、全部七视图raw预测及独立CPU评分。验收精确门槛为clean≥85.98174603174603%、六景均值>72.09619708994708%；须由同一方法相对F20/F21两个对照同时达成。当前没有新目标分数，goal仍未完成。


## 完整源域日志诊断（不改变当前训练）

已解析原生完整T1、F20监督、F21普通PL各3seed的全部99,900条step记录与450条epoch记录；另完整解析既有F12（仅去批内邻近门控）3seed的33,300条step记录。合计133,200条step。原stdout未在source/row/run邻近目录定位，本结论限定于完整结构化指标。未读取F12目标分数。

|方法|seed2701接受率|seed2702接受率|seed2703接受率|
|---|---:|---:|---:|
|完整原生半监督|1.5696%|1.5565%|1.5772%|
|F12：仅去批内邻近门控|90.8409%|90.9362%|91.0409%|
|普通PL|94.7583%|94.9671%|94.8193%|

各半监督模型都是37LU×222＝8214个U步骤、2,097,900次U曝光。原生每seed仅32,654至33,088次曝光进入伪标签CE；F12约190.6至191.0万，普通PL约198.8至199.2万。曝光可重复，不能写成独立包数。原生约11.55%至12.25%的U步骤没有任何选中样本。F12的变化支持“批内邻近门控主要压低CE接受量”，但不能单凭接受量推出目标效果或标签质量；普通PL的现有收益很小也说明单纯放宽筛选不足。

代码中原生伪标签CE按被选样本均值归一，并未额外乘接受比例；不能把1.56%误解成损失又自动缩到1.56%。全部U仍参与0.01熵项，不能说未选中的U完全没有训练作用。U标签一直隐藏，日志pseudo_correct=0不是伪标签精度0%。没有熵梯度和逐门控存活日志，因果解释保留这些边界。当前新方案因此检验“连续权重＋物理视图学习”，以及新增跨RX标签原型参考，而不将接受率本身当成功标准。全部六个健康任务保持原矩阵和参数。
