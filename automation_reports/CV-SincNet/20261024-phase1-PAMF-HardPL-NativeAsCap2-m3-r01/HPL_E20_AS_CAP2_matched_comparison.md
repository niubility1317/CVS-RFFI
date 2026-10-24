# HPL_E20_AS_CAP2 对 HPL_E20_JOINT_AS6G 的固定匹配比较

比较因素：Only change per-group applied U/L normcap1to2 on the complete fixed NativeAs objective;fullLanchor/negative-dot projection/tinyL/raw.95/weight6/6students/nativeAs/losses/defaulthead/originalE20+30 unchanged.No extra autograd or model/headforwards。两边均完成原登记三个模型seed及raw／同一DURA七景。下表为候选减参照，SD为三个配对F seed的样本标准差；不以单因素解释包含两项改变的权重／保护组合，不声称统计显著。相对F20/F21的主目标仍单独判定。

| 读出 | 场景 | 候选（%） | 参照（%） | 配对差 ± SD（pp） | 正收益seed |
|---|---|---:|---:|---:|---:|
| raw | clean | 84.461111 | 84.831944 | -0.370833 ± 0.125412 | 0/3 |
| raw | practical_high | 81.258929 | 81.259921 | -0.000992 ± 0.117831 | 1/3 |
| raw | practical_mid | 79.108730 | 79.017659 | +0.091071 ± 0.106486 | 3/3 |
| raw | practical_low_suburban | 76.875992 | 76.743452 | +0.132540 ± 0.132761 | 3/3 |
| raw | practical_high_urban | 78.084524 | 77.997024 | +0.087500 ± 0.115124 | 2/3 |
| raw | practical_mid_urban | 64.876190 | 64.627778 | +0.248413 ± 0.044791 | 3/3 |
| raw | practical_low_urban | 54.293452 | 53.940476 | +0.352976 ± 0.045472 | 3/3 |
| raw | six_mean | 72.416303 | 72.264385 | +0.151918 ± 0.079463 | 3/3 |
| DURA-LU | clean | 84.551190 | 84.944444 | -0.393254 ± 0.109532 | 0/3 |
| DURA-LU | practical_high | 81.739683 | 82.159722 | -0.420040 ± 0.224276 | 0/3 |
| DURA-LU | practical_mid | 80.570040 | 80.938889 | -0.368849 ± 0.202326 | 0/3 |
| DURA-LU | practical_low_suburban | 78.910913 | 79.260913 | -0.350000 ± 0.252594 | 0/3 |
| DURA-LU | practical_high_urban | 80.547421 | 80.948413 | -0.400992 ± 0.189099 | 0/3 |
| DURA-LU | practical_mid_urban | 73.662698 | 74.015079 | -0.352381 ± 0.160318 | 0/3 |
| DURA-LU | practical_low_urban | 66.955159 | 67.247619 | -0.292460 ± 0.118161 | 0/3 |
| DURA-LU | six_mean | 77.064319 | 77.428439 | -0.364120 ± 0.189632 | 0/3 |

固定参照run：20261024-phase1-PAMF-HardPL-NativeAs-m3-r01。全部seed、六景总和及逐seed整数正确数在配对JSON中，完整分层／修复损伤／成本仍见各自目标报告；未完成方法不参与本次读取。
