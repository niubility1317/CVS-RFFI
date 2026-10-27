# RXTrust-AL：给现有 L 接收响应视图加入真实身份锚

状态：QUEUED_VERIFIED_NEURAL_NOT_STARTED。独立CPU等待控制器已核实PID/CWD/argv/UID及日志/状态增长，12项job排队，0新GPU；实际AL checkpoint/联合梯度校准和训练尚未开始。固定3seed2026092701/02/03，在任何A目标分数读取前提交并发布；整个A完成后自动执行，不按其分数决定运行。已有PL/Sup/A控制复用，当前A健康配置保持。

这是对用户“L与U是否应该同样优化”的具体检验。当前A的额外64条L只计算停止梯度的软KL；真实TX没有进入该新增分支。AL保留原fullT1 L和A全部U目标，只将这64条现有RX扰动L视图的新增目标换成真实TX CE，沿用原LS0.01并启用其梯度。原L监督不替换或削弱，U仍无TX真值。

必须保留输入区别：L侧原本取raw或native普通As增强再作同一RX均值差变换，不是U侧的星地practical H。AL保持这些既有输入和全部前向预算，只检验真实标签在现有L接收扰动上的作用，不把L/U称为相同物理视图。L教师前向仅作停止梯度监测和预算匹配，不提供AL标签。额外L标签严格来自原有L批次的相同parity mask子集。

联合新增损失为(nU·A_U+nL·CE_L)/(nU+nL)，nL=64，nU=256（尾批124）；与A相比U的归一化因子完全保持，新增L梯度。A_U仍是raw阶段全U同base软一致性、H阶段raw被原PL.95接受则用rawargmax硬CE，否则同H软KL。原ordinaryPL.95/weight1不变，legacy P_H_pair始终0，knownRX/Hcache/四RX最大同baseKL选择/物理强度/ramp不变。未新增DURA、恢复IQ、EMA教师、分类头、推理参数或推理视图。

原完整PAMF、单SIG_TIME、原默认头联合训练、各seed原合法scratch E20独立+10+20至E50/11100、13L37LU保持。每seed自身真实E20上测12个训练L/U批（6raw6H），0optimizer/V/target，参数/buffer保持；联合辅助梯度按与A相同max_mode_median规则定alpha，再444步ramp。目标增加后须新测梯度，不能复用A旧系数或旧梯度。实际alpha可能改变，不能宣称严格等更新剂量的因果实验。

源校准另外保存U和L两个梯度分量，检查两条实际logit路径非零与L标签来源；两次额外反向摘要增加计算但不增加F前向，不把负余弦当技术错误或停止标准。首真实E20/full863730/SIG/defaulthead/jointLU检查PASS后立即继续校准和训练，不复制旧smoke。配置和记录见experiment.json。

每模型新增6660更新、U56700各30遍、raw/H各850500，H硬锚/fallback合计850500、每RX170100。额外L真实CE总426240，raw/nativeAs各213120，六个真实TX类别只做计数、不拟合新头或配额。训练仍18724860个F包/model，与A匹配；校准每model33792个计账F包及24次额外反向摘要单列，实际时间保存。原sourceV健康/最终诊断及target测试另计，不从源微小分数选seed或checkpoint。

全部GPU任务等待整个A complete.json，纯资源依赖不读其指标。随后own≤4、每卡own≤1、总CUDA+pending≤2/free10000MiB，总CPUscorer≤1（计入其他owner）。首次本run技术/非有限/输入或独占输出错误停止新派发，健康子进程结束、输出保留；科学负结果不取消任何seed。正在运行的A不热改、不停止。

逐行finalE50后自动原sourceV raw+3H诊断及rawclean+六practical，cross_rx126000+cross_both42000；每行全部14组预测固定后独立truth-last，3×168000×7=3528000新正式判决。报告全三seed均值±样本SD/同seed PL-Sup-A配对/逐景/TX/RX/day/macroF1/净纠错与成本。原rawclean≥85.98174603174603%、六景>72.0961970899471%及全部声明范围严格优于PL/Sup保持，F0探索或源梯度不替代原完整目标，整体goal仍active。

本地17项检查PASS（AL核心/队列7、完整结果3、A默认行为回归7），6模块编译PASS；本候选唯一一次独立P0/P1审查PASS。实际AL E20联合梯度和native checkpoint smoke仍PENDING_FIRST_GPU，不能用本地测试或A旧PASS替代。共享Max两个hook的默认softKL/停止L梯度行为逐值与梯度回归相同，当前A在旧不可变release运行，不受本地修改影响。见local_verification.json。

计算边界：AL每个新增更新都给原64条L分支增加反向传播，共6660个L反向批/model；前向包数匹配不代表总FLOPs或时延相同。最终同时报告真实训练秒数和源校准秒数，不能将“0额外F前向”写成训练免费。

执行commit `e52deb10c235667db76a03af3444d5415e209264` 已提交推送并独立核对上游OID；单archive传输校验VERIFIED，remote compile PASS，普通账号一次preflight核实独占root/三个原scratchE20同契约/已有fullU bank。实际CPU控制器与等待状态增长已核实，当前0新GPU/校准F/optimizer/target。AL真实jointL/U梯度和native checkpoint smoke仍在首GPU阶段等待，未将本地检查或旧A PASS当作本版实际检查。见launch_verification.json。
