# HPL_E20_JOINT_PATH6G 对 HPL_E20_JOINT_COMP6G 的固定匹配比较

比较因素：Only split no-signature CE.25 into no-signature.125+core-only.125;retain full.25,total1.25,joint160/fullLguard/style/weight6/originalE20+30;four rather than two existing-head views,no extra backbone/new head。两边均完成原登记三个模型seed及raw／同一DURA七景。下表为候选减参照，SD为三个配对F seed的样本标准差；不以单因素解释包含两项改变的权重／保护组合，不声称统计显著。相对F20/F21的主目标仍单独判定。

| 读出 | 场景 | 候选（%） | 参照（%） | 配对差 ± SD（pp） | 正收益seed |
|---|---|---:|---:|---:|---:|
| raw | clean | 84.818651 | 84.958929 | -0.140278 ± 0.197139 | 1/3 |
| raw | practical_high | 79.803968 | 79.830159 | -0.026190 ± 0.431321 | 2/3 |
| raw | practical_mid | 77.283532 | 77.276786 | +0.006746 ± 0.384827 | 2/3 |
| raw | practical_low_suburban | 74.942262 | 74.878175 | +0.064087 ± 0.284399 | 2/3 |
| raw | practical_high_urban | 76.312103 | 76.287500 | +0.024603 ± 0.438865 | 2/3 |
| raw | practical_mid_urban | 62.191468 | 62.112698 | +0.078770 ± 0.267112 | 2/3 |
| raw | practical_low_urban | 51.278571 | 51.131349 | +0.147222 ± 0.197054 | 2/3 |
| raw | six_mean | 70.301984 | 70.252778 | +0.049206 ± 0.332566 | 2/3 |
| DURA-LU | clean | 84.902778 | 85.040079 | -0.137302 ± 0.172619 | 1/3 |
| DURA-LU | practical_high | 81.250595 | 81.342659 | -0.092063 ± 0.149648 | 1/3 |
| DURA-LU | practical_mid | 79.849008 | 79.973413 | -0.124405 ± 0.216753 | 1/3 |
| DURA-LU | practical_low_suburban | 78.031349 | 78.158532 | -0.127183 ± 0.226311 | 1/3 |
| DURA-LU | practical_high_urban | 79.881349 | 79.977778 | -0.096429 ± 0.166167 | 1/3 |
| DURA-LU | practical_mid_urban | 72.751984 | 72.985317 | -0.233333 ± 0.137423 | 0/3 |
| DURA-LU | practical_low_urban | 66.017659 | 66.240476 | -0.222817 ± 0.141163 | 0/3 |
| DURA-LU | six_mean | 76.296991 | 76.446362 | -0.149372 ± 0.172182 | 0/3 |

固定参照run：20261024-phase1-PAMF-HardPL-Complement-m3-r01。全部seed、六景总和及逐seed整数正确数在配对JSON中，完整分层／修复损伤／成本仍见各自目标报告；未完成方法不参与本次读取。
