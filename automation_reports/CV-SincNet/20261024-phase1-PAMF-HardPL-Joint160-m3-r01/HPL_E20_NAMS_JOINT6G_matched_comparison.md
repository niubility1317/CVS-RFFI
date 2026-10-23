# HPL_E20_NAMS_JOINT6G 对 HPL_E20_NAMS6G 的固定匹配比较

比较因素：Only add native crossRX on deployedjoint160 from two existing ordinaryU forwards,weight.05/1.25;keep shallow/middle losses,originalE20+30,weight6/fullLguard/six-U/sharedstyle。两边均完成原登记三个模型seed及raw／同一DURA七景。下表为候选减参照，SD为三个配对F seed的样本标准差；不以单因素解释包含两项改变的权重／保护组合，不声称统计显著。相对F20/F21的主目标仍单独判定。

| 读出 | 场景 | 候选（%） | 参照（%） | 配对差 ± SD（pp） | 正收益seed |
|---|---|---:|---:|---:|---:|
| raw | clean | 85.641667 | 85.500794 | +0.140873 ± 0.373388 | 2/3 |
| raw | practical_high | 79.691270 | 79.755556 | -0.064286 ± 0.037134 | 0/3 |
| raw | practical_mid | 77.192857 | 77.277183 | -0.084325 ± 0.067265 | 0/3 |
| raw | practical_low_suburban | 74.751984 | 74.862103 | -0.110119 ± 0.044492 | 0/3 |
| raw | practical_high_urban | 76.249206 | 76.323214 | -0.074008 ± 0.084884 | 1/3 |
| raw | practical_mid_urban | 62.084325 | 62.257540 | -0.173214 ± 0.094504 | 0/3 |
| raw | practical_low_urban | 51.057738 | 51.225000 | -0.167262 ± 0.073665 | 0/3 |
| raw | six_mean | 70.171230 | 70.283433 | -0.112202 ± 0.064195 | 0/3 |
| DURA-LU | clean | 85.685516 | 85.543452 | +0.142063 ± 0.382252 | 2/3 |
| DURA-LU | practical_high | 82.910714 | 82.764683 | +0.146032 ± 0.146659 | 3/3 |
| DURA-LU | practical_mid | 81.667063 | 81.460913 | +0.206151 ± 0.069389 | 3/3 |
| DURA-LU | practical_low_suburban | 79.922222 | 79.743254 | +0.178968 ± 0.110801 | 3/3 |
| DURA-LU | practical_high_urban | 81.657540 | 81.511706 | +0.145833 ± 0.149247 | 3/3 |
| DURA-LU | practical_mid_urban | 74.582937 | 74.431944 | +0.150992 ± 0.111140 | 3/3 |
| DURA-LU | practical_low_urban | 67.744048 | 67.575397 | +0.168651 ± 0.126387 | 3/3 |
| DURA-LU | six_mean | 78.080754 | 77.914649 | +0.166104 ± 0.112841 | 3/3 |

固定参照run：20261024-phase1-PAMF-HardPL-GradientGuard-m6-r01。全部seed、六景总和及逐seed整数正确数在配对JSON中，完整分层／修复损伤／成本仍见各自目标报告；未完成方法不参与本次读取。
