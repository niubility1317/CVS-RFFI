# HPL_E20_AS_SGNC 对 HPL_E20_AS_BG2 的固定匹配比较

比较因素：Both remove raw U/L magnitude cap2 and add actual current-state FP32 parameter step guard;nativeAs/rawacceptance/defaulthead/originalE20+30 retained,one extra parameter-only L optimizer reference perstep;not a singlefactor comparison。两边均完成原登记三个模型seed及raw／同一DURA七景。下表为候选减参照，SD为三个配对F seed的样本标准差；不以单因素解释包含两项改变的权重／保护组合，不声称统计显著。相对F20/F21的主目标仍单独判定。

| 读出 | 场景 | 候选（%） | 参照（%） | 配对差 ± SD（pp） | 正收益seed |
|---|---|---:|---:|---:|---:|
| raw | clean | 84.347817 | 84.526984 | -0.179167 ± 0.103845 | 0/3 |
| raw | practical_high | 81.345437 | 81.357738 | -0.012302 ± 0.135051 | 2/3 |
| raw | practical_mid | 79.190873 | 79.179960 | +0.010913 ± 0.076465 | 2/3 |
| raw | practical_low_suburban | 76.957143 | 76.927579 | +0.029563 ± 0.079346 | 2/3 |
| raw | practical_high_urban | 78.141865 | 78.157937 | -0.016071 ± 0.139976 | 2/3 |
| raw | practical_mid_urban | 64.927381 | 64.931349 | -0.003968 ± 0.098454 | 2/3 |
| raw | practical_low_urban | 54.369048 | 54.335913 | +0.033135 ± 0.046874 | 2/3 |
| raw | six_mean | 72.488624 | 72.481746 | +0.006878 ± 0.095481 | 2/3 |
| DURA-LU | clean | 84.451587 | 84.629762 | -0.178175 ± 0.100548 | 0/3 |
| DURA-LU | practical_high | 81.545833 | 81.725992 | -0.180159 ± 0.149357 | 0/3 |
| DURA-LU | practical_mid | 80.359127 | 80.543254 | -0.184127 ± 0.142229 | 0/3 |
| DURA-LU | practical_low_suburban | 78.727381 | 78.887103 | -0.159722 ± 0.138531 | 0/3 |
| DURA-LU | practical_high_urban | 80.357738 | 80.520437 | -0.162698 ± 0.166128 | 1/3 |
| DURA-LU | practical_mid_urban | 73.493056 | 73.613690 | -0.120635 ± 0.151184 | 1/3 |
| DURA-LU | practical_low_urban | 66.794841 | 66.907143 | -0.112302 ± 0.097354 | 0/3 |
| DURA-LU | six_mean | 76.879663 | 77.032937 | -0.153274 ± 0.138880 | 1/3 |

固定参照run：20261024-phase1-PAMF-HardPL-BranchGuard-m3-r01。全部seed、六景总和及逐seed整数正确数在配对JSON中，完整分层／修复损伤／成本仍见各自目标报告；未完成方法不参与本次读取。
