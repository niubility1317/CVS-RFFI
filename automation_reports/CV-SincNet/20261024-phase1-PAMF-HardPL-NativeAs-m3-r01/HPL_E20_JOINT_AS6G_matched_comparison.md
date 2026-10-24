# HPL_E20_JOINT_AS6G 对 HPL_E20_NAMS_JOINT6G 的固定匹配比较

比较因素：Only restore original mixedhalf native sourceAs,no augmentedviewgate/rawfallback/extra teacher;rawpseudo/acceptedmask/losses/6students/fullLguard/originalE20+30 unchanged.Includes physicalgenerator cost。两边均完成原登记三个模型seed及raw／同一DURA七景。下表为候选减参照，SD为三个配对F seed的样本标准差；不以单因素解释包含两项改变的权重／保护组合，不声称统计显著。相对F20/F21的主目标仍单独判定。

| 读出 | 场景 | 候选（%） | 参照（%） | 配对差 ± SD（pp） | 正收益seed |
|---|---|---:|---:|---:|---:|
| raw | clean | 84.831944 | 85.641667 | -0.809722 ± 0.628186 | 0/3 |
| raw | practical_high | 81.259921 | 79.691270 | +1.568651 ± 0.270936 | 3/3 |
| raw | practical_mid | 79.017659 | 77.192857 | +1.824802 ± 0.336828 | 3/3 |
| raw | practical_low_suburban | 76.743452 | 74.751984 | +1.991468 ± 0.422167 | 3/3 |
| raw | practical_high_urban | 77.997024 | 76.249206 | +1.747817 ± 0.263296 | 3/3 |
| raw | practical_mid_urban | 64.627778 | 62.084325 | +2.543452 ± 0.282566 | 3/3 |
| raw | practical_low_urban | 53.940476 | 51.057738 | +2.882738 ± 0.231467 | 3/3 |
| raw | six_mean | 72.264385 | 70.171230 | +2.093155 ± 0.287087 | 3/3 |
| DURA-LU | clean | 84.944444 | 85.685516 | -0.741071 ± 0.644892 | 0/3 |
| DURA-LU | practical_high | 82.159722 | 82.910714 | -0.750992 ± 0.626566 | 0/3 |
| DURA-LU | practical_mid | 80.938889 | 81.667063 | -0.728175 ± 0.522828 | 0/3 |
| DURA-LU | practical_low_suburban | 79.260913 | 79.922222 | -0.661310 ± 0.475926 | 0/3 |
| DURA-LU | practical_high_urban | 80.948413 | 81.657540 | -0.709127 ± 0.525653 | 0/3 |
| DURA-LU | practical_mid_urban | 74.015079 | 74.582937 | -0.567857 ± 0.427912 | 0/3 |
| DURA-LU | practical_low_urban | 67.247619 | 67.744048 | -0.496429 ± 0.397225 | 0/3 |
| DURA-LU | six_mean | 77.428439 | 78.080754 | -0.652315 ± 0.494291 | 0/3 |

固定参照run：20261024-phase1-PAMF-HardPL-Joint160-m3-r01。全部seed、六景总和及逐seed整数正确数在配对JSON中，完整分层／修复损伤／成本仍见各自目标报告；未完成方法不参与本次读取。
