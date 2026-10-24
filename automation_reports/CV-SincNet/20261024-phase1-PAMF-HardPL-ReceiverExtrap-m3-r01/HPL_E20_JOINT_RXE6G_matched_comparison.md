# HPL_E20_JOINT_RXE6G 对 HPL_E20_NAMS_JOINT6G 的固定匹配比较

比较因素：Only fixed alpha.25 fulljoint160 samepseudo crossRX extrapolation and ordinaryCE full.25+extrap.25 at total1.25;reuse sharedstyle partner,invalid zero delta,all accepted retained;2existinghead calls,no backbone/teacher/newhead;originalE20+30/fullLguard/6students unchanged。两边均完成原登记三个模型seed及raw／同一DURA七景。下表为候选减参照，SD为三个配对F seed的样本标准差；不以单因素解释包含两项改变的权重／保护组合，不声称统计显著。相对F20/F21的主目标仍单独判定。

| 读出 | 场景 | 候选（%） | 参照（%） | 配对差 ± SD（pp） | 正收益seed |
|---|---|---:|---:|---:|---:|
| raw | clean | 85.726786 | 85.641667 | +0.085119 ± 0.123322 | 2/3 |
| raw | practical_high | 79.579167 | 79.691270 | -0.112103 ± 0.177071 | 1/3 |
| raw | practical_mid | 77.036310 | 77.192857 | -0.156548 ± 0.142376 | 1/3 |
| raw | practical_low_suburban | 74.658730 | 74.751984 | -0.093254 ± 0.096242 | 1/3 |
| raw | practical_high_urban | 76.156151 | 76.249206 | -0.093056 ± 0.153568 | 1/3 |
| raw | practical_mid_urban | 62.025000 | 62.084325 | -0.059325 ± 0.030248 | 0/3 |
| raw | practical_low_urban | 51.057738 | 51.057738 | -0.000000 ± 0.028253 | 2/3 |
| raw | six_mean | 70.085516 | 70.171230 | -0.085714 ± 0.093285 | 1/3 |
| DURA-LU | clean | 85.790675 | 85.685516 | +0.105159 ± 0.139908 | 2/3 |
| DURA-LU | practical_high | 82.760714 | 82.910714 | -0.150000 ± 0.111042 | 0/3 |
| DURA-LU | practical_mid | 81.490476 | 81.667063 | -0.176587 ± 0.115498 | 0/3 |
| DURA-LU | practical_low_suburban | 79.704960 | 79.922222 | -0.217262 ± 0.164768 | 0/3 |
| DURA-LU | practical_high_urban | 81.498611 | 81.657540 | -0.158929 ± 0.118228 | 0/3 |
| DURA-LU | practical_mid_urban | 74.383532 | 74.582937 | -0.199405 ± 0.138440 | 0/3 |
| DURA-LU | practical_low_urban | 67.548016 | 67.744048 | -0.196032 ± 0.188939 | 0/3 |
| DURA-LU | six_mean | 77.897718 | 78.080754 | -0.183036 ± 0.138775 | 0/3 |

固定参照run：20261024-phase1-PAMF-HardPL-Joint160-m3-r01。全部seed、六景总和及逐seed整数正确数在配对JSON中，完整分层／修复损伤／成本仍见各自目标报告；未完成方法不参与本次读取。
