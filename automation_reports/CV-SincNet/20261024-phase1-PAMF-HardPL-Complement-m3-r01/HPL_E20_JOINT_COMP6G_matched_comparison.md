# HPL_E20_JOINT_COMP6G 对 HPL_E20_NAMS_JOINT6G 的固定匹配比较

比较因素：Only reallocate ordinary U CE0.5 to full-joint0.25 plus same-pass no-signature0.25 at fixed CE dose1.25;retain joint160/fullLguard/style/weight6/originalE20+30;two existing-head calls,no backbone/new head。两边均完成原登记三个模型seed及raw／同一DURA七景。下表为候选减参照，SD为三个配对F seed的样本标准差；不以单因素解释包含两项改变的权重／保护组合，不声称统计显著。相对F20/F21的主目标仍单独判定。

| 读出 | 场景 | 候选（%） | 参照（%） | 配对差 ± SD（pp） | 正收益seed |
|---|---|---:|---:|---:|---:|
| raw | clean | 84.958929 | 85.641667 | -0.682738 ± 0.697244 | 0/3 |
| raw | practical_high | 79.830159 | 79.691270 | +0.138889 ± 0.584078 | 2/3 |
| raw | practical_mid | 77.276786 | 77.192857 | +0.083929 ± 0.592452 | 2/3 |
| raw | practical_low_suburban | 74.878175 | 74.751984 | +0.126190 ± 0.531365 | 2/3 |
| raw | practical_high_urban | 76.287500 | 76.249206 | +0.038294 ± 0.494736 | 2/3 |
| raw | practical_mid_urban | 62.112698 | 62.084325 | +0.028373 ± 0.291441 | 2/3 |
| raw | practical_low_urban | 51.131349 | 51.057738 | +0.073611 ± 0.258815 | 2/3 |
| raw | six_mean | 70.252778 | 70.171230 | +0.081548 ± 0.447603 | 2/3 |
| DURA-LU | clean | 85.040079 | 85.685516 | -0.645437 ± 0.714815 | 1/3 |
| DURA-LU | practical_high | 81.342659 | 82.910714 | -1.568056 ± 0.748127 | 0/3 |
| DURA-LU | practical_mid | 79.973413 | 81.667063 | -1.693651 ± 0.667426 | 0/3 |
| DURA-LU | practical_low_suburban | 78.158532 | 79.922222 | -1.763690 ± 0.575942 | 0/3 |
| DURA-LU | practical_high_urban | 79.977778 | 81.657540 | -1.679762 ± 0.673163 | 0/3 |
| DURA-LU | practical_mid_urban | 72.985317 | 74.582937 | -1.597619 ± 0.478150 | 0/3 |
| DURA-LU | practical_low_urban | 66.240476 | 67.744048 | -1.503571 ± 0.333424 | 0/3 |
| DURA-LU | six_mean | 76.446362 | 78.080754 | -1.634392 ± 0.577675 | 0/3 |

固定参照run：20261024-phase1-PAMF-HardPL-Joint160-m3-r01。全部seed、六景总和及逐seed整数正确数在配对JSON中，完整分层／修复损伤／成本仍见各自目标报告；未完成方法不参与本次读取。
