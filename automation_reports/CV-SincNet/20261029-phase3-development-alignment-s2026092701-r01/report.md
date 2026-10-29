# PAMF第三阶段：可靠旧类锚点的无标签正交对齐

- run_id：`20261029-phase3-development-alignment-s2026092701-r01`
- group_id：`pamf-phase3-unlabeled-source-anchored-alignment`；类别：`diagnostic`；阶段：`Phase3-unlabeled-development-adaptation`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Fit8source-regularized orthogonal branch maps from initial reliable unlabeled U cores, preserve source/head/readout/threshold, frozen D1V and reused D0 control.

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

## 本次机制与对照

本候选使用原六类PAMF E51 uniformOE、冻结DURA v1/R69（仅P1）、原MLP与source-known阈值，新增外部正交特征adapter。F当前只有GroupNorm/LayerNorm，没有BatchNorm运行统计；本方法不伪装AdaBN、不改默认身份头、不回读source IQ/逐样本feature bank。

初始U核心只选同时通过原source MSP/cosine阈值、原头与fused source锚点最近类一致、time/frequency/PA至少2票一致的记录。ties不计可靠票，不按未知比例或top50%截断，不要求现MLP已经接受。合格旧伪类每类至少8个不同ID、至少3个合格类；类内用MSP×一致票比例加权，各合格类等质量，不强迫填满旧6类、更不预设新类槽位。

8分支分别在原source标准化坐标中求目标核心类方向M与原source方向P，以`R=polar(I+0.5*M.T@P/K)`固定一次拟合。R右乘标准化raw特征再反标准化；fused160交回原cosine六类头重算logits，再以所有新branch和logits重算descriptor/MLP。reference29第27维固定为0。它保持标准化空间的样本间欧氏距离与cosine，不保持原whitened Mahalanobis距离；这是latent特征坐标对齐，不把变换后坐标当真实PA参数或IQ恢复。无可靠core/fused支持则精确identity，不能叫适应成功。

C0 clean与P1星地+DURA各自在D1日期U384拟合独立R，完成adapter checkpoint/marker后才开放D2的V864，V只逐样本预测、不更新状态。C0/P1不共享目标状态或逐条clean参考。D0复用前两个已完成run的同输入预测，在R冻结后才加载，绝不参与core/拟合；原F同输入closed参照也复用其中记录。Utruth不传服务器，新release只含两套U公共三文件；Qpublic与truth分别复用既有不可变根，真实TX/unknown数/配额都不进入predictor。

主方法仍MLP，另外6分数仅诊断；C0未知≥95%、P1总体及每景≥90%，known FRR≤5%、原F同输入open损伤≤2pp，并最大化known接收。适应有效还须实际applied、D1旧类open较D0严格正收益；不足/identity/负结果不扩seed、不回调阈值或重做。无新类注册，本候选不能代表Phase3完成。具体全矩阵、预算、命令/CWD、路径和stop rule见experiment.json及configs/pamf_phase3_development_alignment_20261029.json。

## 来源与本项目推断

SFOW-RFFI的可靠旧类保护和局部几何思想提供参考，但其论文明确假设新类数已知，并采用均衡slot分配、transductive目标训练/评测；公开Stage2代码还按每轮有标签test准确率保存best。这些条件不用于本项目。[PMLR论文](https://proceedings.mlr.press/v306/li26bl.html)、[作者代码](https://github.com/likunling-jingjing/SFOW_RFFI/blob/main/SFOW_RFFI_stage_2.py)。UPUK可动态split/merge、不需真实unknown类数，但固定低熵50%核心及人工置信门槛仍施加容量假设；本候选不用这种固定比例。[CVPR原文](https://openaccess.thecvf.com/content/CVPR2024/papers/Wan_Unveiling_the_Unknown_Unleashing_the_Power_of_Unknown_to_Known_CVPR_2024_paper.pdf)。

本项目的正交adapter、源阈值/多分支票选、identity正则和U→futureV冻结是新的待证假设，不是上述论文复现或已证明收益。此前弱四类CE代理共同平移失败保留，不能用本设计宣称已推翻。当前V已作开发评价，后续新机制可参考开发结论但不回收V训练；正式RX24-18仍隔离。

## 执行边界

一个新run/一个owner，先实际code/config commit、聚焦负测和一次P0/P1，再单归档/一次SHA比较/remote compile及N607 preflight。launcher首步真实checkpoint合成2IQ no-query smoke，PASS继续U；detached driver自动逐condition预测后独立score，保存实际PID/CWD/argv/GPU/log，科学失败保留报告。每次仅新增一GPUworker，实际显存不足等待，不停止/修改其他owner。无标签更新是U拟合R状态，神经optimizer步数为0，不把“无神经训练”写成“没有目标适应”。

## 实际发布前固定

实际code/config commit `cdaeccaf201f8c242ac00400b5c499f4dc1e63a8`。pure+outer原联合22tests/39subtests PASS（包括core11，不重复相加）；launcher独立3tests PASS工具stdout。唯一完整P0/P1发现U/Q存储同根假设与N607 ndarray ABI两问题，已分别修复为不同角色句柄且允许跨release、适配后head走原byte-buffer tensor桥。两新回归先RED，修复后仅两回归＋1邻近守卫3tests/14subtests PASS，10项未变未重跑；一次原问题定向复核PASS，无额外全审。完整core/pipeline/P1日志和launcher工具返回摘要在verification_summary.json引用，未把重复执行次数相加为覆盖数量。

一次N607 preflight PASS：两旧D0与Q public句柄、源聚合/阈值及P1 G存在，新run/release/archive路径未占；GPU空闲19726至22694MB（启动逐row再查实际memory），只新增1worker同时运行，不停止外人。服务器将只接收两个U的公共三文件，无Utruth；Q和D0复用旧不可变根。模型真实checkpoint synthetic2 smoke是每condition首步，PASS立即继续，不设smoke token。适应目标另外保护U未见later unknown总体及每有样本主景95%/90%，相关私有分组只在独立scorer。

新增R状态每condition234825个FP64系数、1878600字节（约1.79MiB），不是新增神经可训练权重；原F232993和MLP1633参数不变。小型合成时长不外推部署吞吐，实际本run计时/状态按后态证据报告。当前真实U尚未读取、尚无新性能结果，总目标仍ACTIVE。

## 实际U适应与独立评分完成（2026-10-29）

已VERIFIED/ANALYZED_DEVELOPMENT_ONLY，C0/P1各adapter→预测→独立score全COMPLETE，无技术失败。两行均实际applied，不是identity；C0合格core157，P1初始150/合格147，均支持5个旧伪类0/2/3/4/5，不能把伪类覆盖写成真实TX覆盖或纯度。每条件只U384拟合R，再V864逐样本预测，D0精确旧结果复用，全部新预测固定后独立truth-last。source/F/G/head/读出/阈值更新0，原模型参数和R在query保持不变，formalRX24-18读取0。

MLP clean旧open62.6736→63.1944（+0.5208pp）、known接收80.7292→81.25、unknown99.3056→98.9583；星地旧open53.2986→55.0347（+1.7361pp）、known接收72.9167→74.8264、unknown57.2917不变。两行虽有正收益，known保护仍失败，P1unknown/later保护也失败，均NO_PROMOTION，不扩seed、不回调η或阈值。真实无标签适应已执行，但尚无通过目标的适应方法，匿名注册0，总目标ACTIVE。

实际driver2575506和C0worker2575527的live /proc PID/CWD/cmdline已捕获，C0日志541→960B；完成后owned均退出。两个模型工作期间未捕获owned CUDA compute快照，GPU3来自driver分配记录，不将其他用户GPU表或单次采样称峰值；不为补观察重跑。完整17个元数据/日志文件及adapter/predictions NPZ实际存在/大小已读回。真实checkpoint合成2IQ smoke均在U前PASS，适配后head在已确认N607 ABI组合上实际成功。一次archive比较和compile为VERIFIED。阶段实际计时见post_state，不当作逐包部署延迟。

同row完整报告见`docs/PAMF_PHASE3_DEVELOPMENT_ALIGNMENT_RESULTS_20261029.md`，评分/核心诊断/后态在本local_artifacts根。clean与star仍1248个原始物理记录的两条件、U日D1→V日D2，开发TX已source评过，不是正式新TX盲测。保留全部负结果与其他owner工作，foreignstop0。
