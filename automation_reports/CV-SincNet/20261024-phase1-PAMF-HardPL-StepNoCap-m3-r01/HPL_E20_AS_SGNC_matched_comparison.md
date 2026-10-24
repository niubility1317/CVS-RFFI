# HPL_E20_AS_SGNC 对 HPL_E20_AS_SG2 的固定匹配比较

比较因素：Only remove raw U/L norm cap2;retain three raw direction constraints/raw-norm nonexpansion and actual current-state FP32 parameter step guard,all loss/acceptance/defaulthead/forwards/derivative/reference-step/originalE20+30 budgets unchanged。两边均完成原登记三个模型seed及raw／同一DURA七景。下表为候选减参照，SD为三个配对F seed的样本标准差；不以单因素解释包含两项改变的权重／保护组合，不声称统计显著。相对F20/F21的主目标仍单独判定。

| 读出 | 场景 | 候选（%） | 参照（%） | 配对差 ± SD（pp） | 正收益seed |
|---|---|---:|---:|---:|---:|
| raw | clean | 84.347817 | 84.485714 | -0.137897 ± 0.075314 | 0/3 |
| raw | practical_high | 81.345437 | 81.270040 | +0.075397 ± 0.192819 | 2/3 |
| raw | practical_mid | 79.190873 | 79.106548 | +0.084325 ± 0.117629 | 2/3 |
| raw | practical_low_suburban | 76.957143 | 76.862500 | +0.094643 ± 0.089713 | 2/3 |
| raw | practical_high_urban | 78.141865 | 78.085516 | +0.056349 ± 0.176366 | 2/3 |
| raw | practical_mid_urban | 64.927381 | 64.880357 | +0.047024 ± 0.105937 | 2/3 |
| raw | practical_low_urban | 54.369048 | 54.285317 | +0.083730 ± 0.029704 | 3/3 |
| raw | six_mean | 72.488624 | 72.415046 | +0.073578 ± 0.115687 | 2/3 |
| DURA-LU | clean | 84.451587 | 84.564484 | -0.112897 ± 0.073830 | 0/3 |
| DURA-LU | practical_high | 81.545833 | 81.700000 | -0.154167 ± 0.076142 | 0/3 |
| DURA-LU | practical_mid | 80.359127 | 80.520040 | -0.160913 ± 0.055979 | 0/3 |
| DURA-LU | practical_low_suburban | 78.727381 | 78.865873 | -0.138492 ± 0.045239 | 0/3 |
| DURA-LU | practical_high_urban | 80.357738 | 80.496627 | -0.138889 ± 0.094443 | 0/3 |
| DURA-LU | practical_mid_urban | 73.493056 | 73.602579 | -0.109524 ± 0.077516 | 0/3 |
| DURA-LU | practical_low_urban | 66.794841 | 66.898016 | -0.103175 ± 0.041757 | 0/3 |
| DURA-LU | six_mean | 76.879663 | 77.013856 | -0.134193 ± 0.064024 | 0/3 |

固定参照run：20261024-phase1-PAMF-HardPL-StepGuard-m3-r01。全部seed、六景总和及逐seed整数正确数在配对JSON中，完整分层／修复损伤／成本仍见各自目标报告；未完成方法不参与本次读取。
