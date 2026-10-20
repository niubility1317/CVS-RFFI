# R17 先验共享与低秩蒸馏精简

状态：RUNNING。用户最新要求本对话专门优化 R17 精简版，尽量精简且不能降低性能。固定原 R17 unroll_unshared/source-selected finalE20/4940/method cd977448；同原 F50 seed2026092701/defaulthead、168000 目标物理样本、clean＋六 practical 的准确率逐场景不得下降，容许下降为 0 个百分点。原 low urban67.2166666667%，70% 长期要求未达到。基准已有暴露，不称新盲测。

|固定源候选|完整恢复链参数|新先验参数|相对原4665073|
|---|---:|---:|---:|
|shared_mean|2993389|835842|减少35.83%|
|factor_half|3775105|1617558|减少19.08%|

shared_mean 把原三份训练后 Core9 状态逐元素平均，三步共享一份。factor_half 对原三份先验分别作卷积低秩分解：kernel>1、rank=floor(min(out,in×kernel)/2)，且确实减少参数时，使用均衡平方根奇异值的 CPU SVD 初始化两层卷积；保留1×1、GroupNorm和激活。三步展开、每步2次warm-start CG、原R02 coarse/R07 action/nativevariance/SCALE²保持。首轮只研究先验压缩，不同时减少步数。

各新训练20轮/4940更新，完整合法L6300/U56700，原R07 cachedreceived epochs1..20、15%clean/85%residual、batch256、原data seed2026102001，不做额外clean拼接。原R17教师完全冻结；一次原coarse/action上下文复用于教师与学生独立三步轨迹，教师阶段输出等价于原R17。损失为最终配对IQNMSE＋10×三步学生/教师IQNMSE均值，分母均为配对clean I/Q均方（下界1e-4）。AdamW2e-4/warm247/cos2e-5、WD1e-4/clip1、FP32/noTF32。0TX标签/F权重或输出/V/target训练；不拟合任何分类头。新prior累计40轮，coarse20/action20冻结，完整累计80串行训练pass；蒸馏教师计算与原训练成本单列，不能当作从零20轮或相同总训练成本。

每候选完成固定finalE20后，源V1800七视图全部预测固定，再独立CPU评分。每个场景accuracy必须≥原R17；另保留mean6NMSE≤1.05×原R17、cleanNMSE≤.02波形护栏。所有合格源候选按完整参数量升序、名称排序，固定选一个；只有它自动进行完整168000×七视图目标确认，无合格则0target。目标下降后不补测落选者、不回流调参或选择性重跑。源V标签仅资格/固定选择，不反传或更新状态。

目标预测读取同一当前IQ，经完整恢复图输出单一IQ，供原F50所有分支及默认头；不读教师、clean、源样本缓存、TX/RX/场景/其他帧或更新状态。原R17同输入source/target既有完整预测复用，不重复生成。每行七预测固定后独立CPU truth-last；报告accuracy/MacroF1、逐TX/RX/day混淆矩阵/修复/损伤、全帧/主体232/尾24的配对clean NMSE与FFT/特征差距。正式逐场景accuracy>=原R17是零退化验收，平均分不替代逐场景，数学全局最简与多seed稳定性未证。

相同GPU、源V固定16个IQ、batch1、各2次warmup实测原R17与学生完整图延迟/峰值显存；GPU/样本/运行顺序保留，噪声与缓存效应限制如实报告。参数减少不预先保证延迟收益，包含coarse/action与CG，不只报prior。FP32模型状态文件不含优化器；训练权重/原始日志/IQ/逐样本预测与私有机器细节不提交。

N607普通账户，一次preflight/归档SHA比较/远端compile。GPU4/5等待旧scratch根终态且实际ownedsource/target进程退出，才创建新模型；保留spectral6/7及其他owner，最多4ownGPU、每卡一个、source≥18GB/target≥12GB、allowner<2。CPU2线程/最多一ownscorer，等待spectral根终态后共用score.lock，无循环。launcher首步实际R17/2Ls/原F默认头无querysmoke，教师三步函数完全一致、各学生先验梯度、教师/context冻结、checkpoint往返，0TX/V/target/optimizer，PASS直接继续。聚焦测试与唯一一次独立P0/P1结果在启动前追加。技术失败保留产物，健康行继续，pendingtarget暂停，无自动重试；源/目标科学失败不作技术重启理由。精确配置、六类seed、命令/CWD/路径/资源/停止规则见experiment.json。


发布前验证：28项聚焦检查通过。唯一一次独立P0/P1审查独立复跑19项新检查，发现控制器状态文件重复独占创建的P1；已改为仅对运行状态使用临时文件替换，新增两次轮询与source→selectedtarget自动派发回归。该问题唯一一次定点复审及回归PASS，无重复全审。一次N607普通账户预检VERIFIED：原R17源选择先于目标、完整训练预算与0TX/F/target、同原F七视图基线句柄、原R07缓存和源角色均匹配；GPU仍有旧scratch实际handle，launcher将等待它自然退出，不停止健康进程。


实际发布状态：方法commit `606b5b20943e9839fe97cbe15a6c19baf6ef0189`，push后本地HEAD/远端OID独立比较一致；一次归档SHA比较、远端compile和入口help均通过。launcher实际原R17/原F50/2Ls无querysmoke PASS：教师stage函数完全等价、所有学生先验梯度有限非零、context/教师无梯度、严格checkpoint输出逐位相同、0TX/V/target/optimizer。旧scratch根终态且所有owned GPU handles实际退出后，新GPU4/5两臂才创建；独立/proc与GPU证据确认controller/两source实际LIVE、CWD/完整命令/commit匹配、GPU PID与日志增长，均已完成第1轮/247updates，各曝光L6300/U56700。梯度连通与源训练loss不证明目标性能；当前无压缩模型目标结果，零退化未证实。
