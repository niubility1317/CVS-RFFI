# 源谱方向：当前E20默认头的有限IQ验证

状态：REGISTERED_NOT_RUN。当前唯一问题是：较丰富、TX平衡的源关联谱形方向能否在当前SIG_TIME＋RDrop的有限IQ上保持身份；不是训练收益实验。前一固定理想代理得到当前三抽头族9.3023%方向覆盖，只提供形状不足线索。

源银行只用旧L6300的18折预测，按excluded day/RX平均6个excluded TX，固定15方向。DCT1至8在256频点重建形状，IFFT后取居中33tap并乘Hann、去零延迟标量。固定正交旋转为控制。每对核匹配L2，以共同缩放保证delta核L1<=.06，即有效内部H位于[.94,1.06]；不增加噪声、相位、PA、剪裁或深衰落。非平坦方向保留，包RMS恢复。

按epoch/RX/day/eq/view生成虚拟session，所有TX共享；固定源E1和E20、EMP/ROT及reflect/edge。未经缩放的内部[16,240)不接触人工边界，原参考[80,160)在其中；RMS标量可能传播端部影响。实际波形扰动能量未保证相等，必须报告。

固定原E20_RDROP seed2026092701 final默认头，完整V27000。严格核查原scratch来源与fullL6300/U56700/V27000契约，不使用任何新训练继承。首步真实checkpoint无query smoke2通过后立即继续7条件；预计189002个前向，CPU2、CUDA隐藏，0优化/EMA/头拟合、0新模型。角色和具体路径/命令/owner/输出见experiment.json。禁止覆盖，技术异常保留产物且不自动重试，不改其他任务。

预先固定：EMP E1/E20 reflect各自overall损失<=1pp且任何源RX损失<=2pp才安全PASS。ROT/EDGE仅作控制，不事后替换候选或调幅。安全PASS仍不证明泛化、物理方向优越或训练受益；FAIL结束本固定候选。旧33tapPHYS/ROT无明确性能优势的负结果保留。

尚无本run实际源IQ性能、训练或目标结果。后续训练若由源证据支持，仍限每模型scratchE20=13L+7U，固定CVS/defaulthead/原U规则；正式clean及六practical不变。不能把本probe当作已达clean/urban双突破。

本地算子与协议检查、编译通过；本候选唯一一次限定独立P0/P1审查PASS，无直接P0/P1。真实checkpoint smoke和源IQ验证由已提交launcher首步执行。
