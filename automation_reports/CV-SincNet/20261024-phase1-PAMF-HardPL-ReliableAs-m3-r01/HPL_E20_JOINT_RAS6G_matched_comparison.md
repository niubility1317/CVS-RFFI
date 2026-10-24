# HPL_E20_JOINT_RAS6G 对 HPL_E20_NAMS_JOINT6G 的固定匹配比较

比较因素：Replace only mixed-half exactraw policy with native source As gated by rawaccepted/sameargmax/Asconfidence.95 and exactraw fallback;all accepted raw samples/loss coefficients/student views unchanged. Includes extra source As generation and eval cost;not an isolated gate-only effect。两边均完成原登记三个模型seed及raw／同一DURA七景。下表为候选减参照，SD为三个配对F seed的样本标准差；不以单因素解释包含两项改变的权重／保护组合，不声称统计显著。相对F20/F21的主目标仍单独判定。

| 读出 | 场景 | 候选（%） | 参照（%） | 配对差 ± SD（pp） | 正收益seed |
|---|---|---:|---:|---:|---:|
| raw | clean | 85.448810 | 85.641667 | -0.192857 ± 0.199460 | 1/3 |
| raw | practical_high | 80.243254 | 79.691270 | +0.551984 ± 0.149494 | 3/3 |
| raw | practical_mid | 77.780556 | 77.192857 | +0.587698 ± 0.228405 | 3/3 |
| raw | practical_low_suburban | 75.380754 | 74.751984 | +0.628770 ± 0.242885 | 3/3 |
| raw | practical_high_urban | 76.793254 | 76.249206 | +0.544048 ± 0.199271 | 3/3 |
| raw | practical_mid_urban | 62.523214 | 62.084325 | +0.438889 ± 0.190002 | 3/3 |
| raw | practical_low_urban | 51.487897 | 51.057738 | +0.430159 ± 0.157531 | 3/3 |
| raw | six_mean | 70.701488 | 70.171230 | +0.530258 ± 0.193313 | 3/3 |
| DURA-LU | clean | 85.518651 | 85.685516 | -0.166865 ± 0.198008 | 1/3 |
| DURA-LU | practical_high | 82.694048 | 82.910714 | -0.216667 ± 0.159174 | 0/3 |
| DURA-LU | practical_mid | 81.449008 | 81.667063 | -0.218056 ± 0.151899 | 0/3 |
| DURA-LU | practical_low_suburban | 79.722222 | 79.922222 | -0.200000 ± 0.169299 | 0/3 |
| DURA-LU | practical_high_urban | 81.417262 | 81.657540 | -0.240278 ± 0.132141 | 0/3 |
| DURA-LU | practical_mid_urban | 74.366667 | 74.582937 | -0.216270 ± 0.168754 | 0/3 |
| DURA-LU | practical_low_urban | 67.536111 | 67.744048 | -0.207937 ± 0.102312 | 0/3 |
| DURA-LU | six_mean | 77.864220 | 78.080754 | -0.216534 ± 0.145595 | 0/3 |

固定参照run：20261024-phase1-PAMF-HardPL-Joint160-m3-r01。全部seed、六景总和及逐seed整数正确数在配对JSON中，完整分层／修复损伤／成本仍见各自目标报告；未完成方法不参与本次读取。
