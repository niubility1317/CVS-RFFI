# HPL_E20_AS_SG2 对 HPL_E20_AS_BG2 的固定匹配比较

比较因素：Only add actual FP32 extra-parameter-step projection relative to the same current-state L-only AdamW reference;all branch gradient guards/rawacceptance/losses/defaulthead/forwards/originalE20+30 retained.Extra6660parameter-only reference optimizer steps permodel,no extra autograd/model/headforward;not equal compute。两边均完成原登记三个模型seed及raw／同一DURA七景。下表为候选减参照，SD为三个配对F seed的样本标准差；不以单因素解释包含两项改变的权重／保护组合，不声称统计显著。相对F20/F21的主目标仍单独判定。

| 读出 | 场景 | 候选（%） | 参照（%） | 配对差 ± SD（pp） | 正收益seed |
|---|---|---:|---:|---:|---:|
| raw | clean | 84.485714 | 84.526984 | -0.041270 ± 0.081139 | 1/3 |
| raw | practical_high | 81.270040 | 81.357738 | -0.087698 ± 0.058170 | 0/3 |
| raw | practical_mid | 79.106548 | 79.179960 | -0.073413 ± 0.050803 | 0/3 |
| raw | practical_low_suburban | 76.862500 | 76.927579 | -0.065079 ± 0.016207 | 0/3 |
| raw | practical_high_urban | 78.085516 | 78.157937 | -0.072421 ± 0.036452 | 0/3 |
| raw | practical_mid_urban | 64.880357 | 64.931349 | -0.050992 ± 0.012391 | 0/3 |
| raw | practical_low_urban | 54.285317 | 54.335913 | -0.050595 ± 0.032543 | 0/3 |
| raw | six_mean | 72.415046 | 72.481746 | -0.066700 ± 0.020633 | 0/3 |
| DURA-LU | clean | 84.564484 | 84.629762 | -0.065278 ± 0.075260 | 1/3 |
| DURA-LU | practical_high | 81.700000 | 81.725992 | -0.025992 ± 0.073661 | 1/3 |
| DURA-LU | practical_mid | 80.520040 | 80.543254 | -0.023214 ± 0.087666 | 1/3 |
| DURA-LU | practical_low_suburban | 78.865873 | 78.887103 | -0.021230 ± 0.094285 | 1/3 |
| DURA-LU | practical_high_urban | 80.496627 | 80.520437 | -0.023810 ± 0.072061 | 1/3 |
| DURA-LU | practical_mid_urban | 73.602579 | 73.613690 | -0.011111 ± 0.073794 | 1/3 |
| DURA-LU | practical_low_urban | 66.898016 | 66.907143 | -0.009127 ± 0.057993 | 1/3 |
| DURA-LU | six_mean | 77.013856 | 77.032937 | -0.019081 ± 0.074982 | 1/3 |

固定参照run：20261024-phase1-PAMF-HardPL-BranchGuard-m3-r01。全部seed、六景总和及逐seed整数正确数在配对JSON中，完整分层／修复损伤／成本仍见各自目标报告；未完成方法不参与本次读取。
