# P35 有界时移 × 原 practical，E30

状态：RUNNING，尚无新模型正式成绩。已核实预算为从零 E30（20L+10U），每条6660次更新。新增 B10 有界时移、B11 有界时移后原 practical；复用原 P32 B00/B01 完整同预算控制。全部源5RX、L6300/U56700隐藏标签/V27000不变，不留出训练域。

代码提交 7e4af64bb887c08c6a225bcb6bee4856fdb7c588；起点 d4391760b787aca67bdd298e26f2939e577becf9。±16点整数reflect时移并恢复原始RMS；native As residual/post_sync/EQoffG1/ref30/25MHz/2.45GHz保留，不改正式IQ、SNR或信道。F+跨RX同TX对比/RDrop/旧门控ownEMA/defaultcos30正常联合训练。

验证：7项针对性测试与包编译PASS，独立P0/P1检查PASS（NaN占位比较已修复、一次定向复核）；单archive传输比较与N607编译VERIFIED；启动真实CP no-query兼容smoke2NN PASS。两条各848个原生参数与旧对应控制一致，排除仅运行ID/候选ID/输出目录。兼容权重不作训练初始化。

普通账户N607启动已独立读回PID/CWD/命令/GPU/日志增长：dispatcher663968，B10 PID664133/GPU2，B11 PID664210/GPU3。旧B00独立源时移参考3602NN完成、无状态改变/训练/目标访问；其时移源准确率两个draw均92.555556%，这不是正式测试结果。共享GPU总进程cap3含其他owner，保护健康实验。

源判据见 preregistration.md：两条全源clean保持、B10两draw时移至少+5pp、B11每个两draw×两原urban至少+1pp。collector在读正式分数之前固定源判定。源FAIL仍完成每条七环境truth-last，不救幅度/概率/窗口/seed或预算。单seed不能声称复现。

完整逐行七环境结果、训练曲线、分层和收益归因待运行完成后补入；不宣称低urban>60%或clean突破。原正式As控制clean83.057143%、lowurban53.394643%，相对其无As匹配控制clean−2.180357pp、lowurban+21.639286pp，属于历史P32单seed。

本地产物 E:/type10-7/local_artifacts/cvs_FcrossRX_timing_factorial_E30_20261017。远端根 /home/szu2070436088/2510044040/CV-SincNet/runs/20261017-phase1-FcrossRX-timing-factorial-e30-s2026092701-m2-r01。七环境固定预测后独立truth-last自动接续。GitHub推送FAILED（连接重置），API独立核对远端仍618b9b5912f28de5650049c493a8e8a38484bba4；本地提交保留，训练发布使用实际本地代码提交。

后续交付更新：GitHub推送已恢复，远端分支OID与本地分析提交2a1efbaeca711303083cb78c5a4645e089ed5dae独立核对VERIFIED，覆盖训练代码提交7e4af64bb。完成后分析已在独立CPU进程等待全行正式评价完成，分析代码提交2a1efbaeca；不读取IQ/checkpoint或训练参数，源判据独立复算后才读取正式标签。最近核实B10完成9/30、B11完成4/30，均健康、无跳过更新；尚无新正式成绩。

## 首条已完成，源域固定判据否定

B10 已完成 FINAL30 / 6660 更新。源训练前向样本5407922（student train3976920/eval864002、teacher train0/eval567000），有标签前向3409920；源训练墙钟1633.512086s，源读出16202NN/8.976482s。时移实际覆盖852480视图、全部6300个L物理样本、[-16,16]全部33偏移，最大相对功率误差1.307899e-7；平均相对L2变化1.390768，不将整数时移称为数值幅度很小的波形变化。

全源V clean：B10=97.896296%，匹配B00=98.303704%，差−0.407407pp，超过预登记−0.25pp下限。两个同观察timing V draw：B00均92.555556%，B10为97.444444%/97.333333%，提升4.888889/4.777778pp，均低于固定+5pp。Ag clean保持与双draw时移增益两项均FAIL，不放宽阈值、不缩小幅度或概率救跑、无扩seed依据。完整源判定文件仍由原dispatcher在两条训练完成后生成；此处是提前固定的源域否定事实，未读取新正式分数。

B10自动正式评价已核实：clean+全部六种practical，每视图168000样本，共1176000固定预测，actualNN1176002；参数全部冻结、truth_read=false/query_fit=false/head_fit_updates=0，之后独立scorer truth-last，126条分层指标完成。预测墙钟47.255914s。worker已自然退出，未停止或重启。新正式分数暂不读取，等待B11及完整源判定后统一独立分析。B11仍健康执行E30，不因当前源域FAIL中止。

原U阶段已读回：第21轮U56700、pseudo_truth_available=0、U_truth_read=false、head_fit_updates=0，教师评估56700NN/训练0NN；默认头随骨干有更新。门控选择数和参数位移不作为身份性能收益。
