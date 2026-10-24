# HPL_E20_JOINT_AS6G 对 HPL_E20_JOINT_RAS6G 的固定匹配比较

比较因素：Use fullnativeAs half instead of Asconfidence/sameargmax gate with exactraw fallback;rawacceptedmask/losses/6students/fullLguard/originalE20+30 unchanged.Also removes extra As evalteacher;not sameFLOP or isolated gate-only effect。两边均完成原登记三个模型seed及raw／同一DURA七景。下表为候选减参照，SD为三个配对F seed的样本标准差；不以单因素解释包含两项改变的权重／保护组合，不声称统计显著。相对F20/F21的主目标仍单独判定。

| 读出 | 场景 | 候选（%） | 参照（%） | 配对差 ± SD（pp） | 正收益seed |
|---|---|---:|---:|---:|---:|
| raw | clean | 84.831944 | 85.448810 | -0.616865 ± 0.554545 | 0/3 |
| raw | practical_high | 81.259921 | 80.243254 | +1.016667 ± 0.233032 | 3/3 |
| raw | practical_mid | 79.017659 | 77.780556 | +1.237103 ± 0.176224 | 3/3 |
| raw | practical_low_suburban | 76.743452 | 75.380754 | +1.362698 ± 0.264550 | 3/3 |
| raw | practical_high_urban | 77.997024 | 76.793254 | +1.203770 ± 0.125046 | 3/3 |
| raw | practical_mid_urban | 64.627778 | 62.523214 | +2.104563 ± 0.095038 | 3/3 |
| raw | practical_low_urban | 53.940476 | 51.487897 | +2.452579 ± 0.079585 | 3/3 |
| raw | six_mean | 72.264385 | 70.701488 | +1.562897 ± 0.141356 | 3/3 |
| DURA-LU | clean | 84.944444 | 85.518651 | -0.574206 ± 0.568178 | 1/3 |
| DURA-LU | practical_high | 82.159722 | 82.694048 | -0.534325 ± 0.781435 | 1/3 |
| DURA-LU | practical_mid | 80.938889 | 81.449008 | -0.510119 ± 0.673074 | 1/3 |
| DURA-LU | practical_low_suburban | 79.260913 | 79.722222 | -0.461310 ± 0.628290 | 1/3 |
| DURA-LU | practical_high_urban | 80.948413 | 81.417262 | -0.468849 ± 0.655619 | 1/3 |
| DURA-LU | practical_mid_urban | 74.015079 | 74.366667 | -0.351587 ± 0.596593 | 1/3 |
| DURA-LU | practical_low_urban | 67.247619 | 67.536111 | -0.288492 ± 0.488598 | 1/3 |
| DURA-LU | six_mean | 77.428439 | 77.864220 | -0.435780 ± 0.635772 | 1/3 |

固定参照run：20261024-phase1-PAMF-HardPL-ReliableAs-m3-r01。全部seed、六景总和及逐seed整数正确数在配对JSON中，完整分层／修复损伤／成本仍见各自目标报告；未完成方法不参与本次读取。
