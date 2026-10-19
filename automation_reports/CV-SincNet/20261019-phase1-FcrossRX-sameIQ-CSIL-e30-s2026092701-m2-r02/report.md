r02 技术恢复：仅补历史factory的完整仓库依赖路径；隔离回归与唯一问题定点复审PASS。原r01保留，无训练结果。原科学配方如下。

# 同受损观测CSIL类别迁移：默认头CE匹配E30

- run_id：`20261019-phase1-FcrossRX-sameIQ-CSIL-e30-s2026092701-m2-r02`
- group_id：`phase1-sameIQ-CSIL-transfer`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

同As20父模型独立CE/KD两条E10；全标签CE不变，双方去除AsKL，冻结CSIL同IQ类别监督，控制做等成本teacher shadow；自动all7 truth-last。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。
# 最小预登记：同受损 IQ 的 CSIL 类别迁移

这条路线区别于另一对话的 U practical 路由和 HQ-labelCE。先前同源、同1800物理包、同两draw完整实用信道的冻结比较已证实：CSIL mid/low为86.6667%/78.1944%，当前PAMF为72.8056%/59.0833%。CSIL固定为附件预先首选教师；MoPC-HR仅提供互补证据，不按分数换教师。历史教师为six-block IQ扩展，不冒充原论文数据复现。

本轮2行从同一个原P43 As20 student＋ownEMA独立追加E10，分别累计E30/6660更新；重置AdamW、cosine10和Ucache，沿用原Sinc物理坐标修正。原As30/NoAs30仅复用冻结产物。矩阵/seed/原配方/准确命令与路径以experiment.json、configs/*.json和运行resolved args为准。

C01、D01都保留原6×128学生前向、.75ordinary/.25raw/.25完整As标签CE、原ordinary/raw RDrop、ORD128跨RX同TX项和原U/自身EMA规则；每步所有64个As继续原温度1/smoothing.01 CE。仅从原总目标减去 .25×KL_As64；其余KL保留。

D01额外加入 .25×6×mean64[teacherCorrectL×mean两学生pass KL(softmax(教师原logits) || softmax(学生logits/6))]。教师原尺度5与学生部署尺度30对应；这是蒸馏分布尺度，不改变As CE温度或部署头。教师正确性只按合法L标签判断，mask后仍除以64，不按接受数归一化；没有学生高置信条件。两次学生实际mixed IQ必须相同；教师读取其中同一64个完整As，不另生成信道、clean或HQ输入，不对隐藏特征对齐。

教师固定seed2026092701、scratch final200/9800更新，L/U/V所有实际role_ids与当前契约精确一致，无target训练/选模。运行检查checkpoint完整来源/类别映射，始终eval且无梯度，保持BN buffer；不注册到学生Module/state_dict/optimizer，不复制进部署checkpoint。U不接入该教师，TX真值继续隐藏。C01也做相同64教师前向，监督系数为0，匹配计算；教师初始化保存/恢复训练RNG。

新增每row：2220更新、1704960学生L前向＋142080冻结CSIL前向＋567000原U学生＋567000原U EMA，共2981040样本前向；共享As20父训练和历史CSIL200训练分别披露，不计为免费。无额外学生前向、部署参数或另训头。真实CP首/末步合成smoke2502NN单列，0更新/query。

源判据：D相对C的mid/low均值至少+.5个百分点；D low不低于原As30；D完整V clean相对NoAs30不低于−.1个百分点。任一失败关闭固定教师/系数/温度，不扩E50/种子或重调救援。两行固定末轮后都自动执行合法源读出、clean＋全部六种practical预测、全部prediction固定后独立exec truth-last评分、同row配对报告，无论源条件成功与否。正式IQ/信道/SNR/样本不变，正式成绩不回流选模。

N607普通账号；GPU按全owner≤2且free≥12GB分配，CPU2，FP32/TF32off。唯一owner=codex/root/sameIQ-CSIL；不可覆盖run/output root、命令和停止规则见登记。仅契约/来源/非有限/无法执行/预测闭合/错误truth连接/输出冲突等系统技术失败中止所属run，保留产物；低性能不停止，健康旧任务不干预。

本地实际Torch检查PASS：12协议负例、KD完整批次公式和两pass梯度与直接计算一致；真实checkpoint/full-path smoke作为launcher首步，PASS后直接训练。一次P0/P1审查，单归档一次SHA对比、一次远端编译。REJECTED_EXTRA_GATE：额外设计审批、报告/成员hash、重复审查和不变数据重验。
