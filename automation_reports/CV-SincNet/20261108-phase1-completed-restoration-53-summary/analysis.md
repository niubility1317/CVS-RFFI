# 53 项恢复候选的逐景配对汇总

范围：N607 中 2026-11-06 至 2026-11-08 命名为 Phase1 ground-test 且有完整评分和 DURA 配对的 53 项。包含一次固定双路概率平均读出，因此 53 项为候选或推理方式的测试数，不全部是不同恢复网络。每项均核对 Tri-SIG-TIME SSL Main split 单 seed 2026092701、默认头、每景 168000 条、七景预测先于 truth-last、无目标反馈，以及 DURA 正确数和净纠错算术。范围不代表全部历史研究。

| 受损场景 | 高于 DURA 的测试数 |
| --- | ---: |
| 郊区高 | 1/53 |
| 郊区中 | 0/53 |
| 郊区低 | 1/53 |
| 城市高 | 0/53 |
| 城市中 | 1/53 |
| 城市低 | 1/53 |

合成响应约束在低郊区、中城区、低城区分别净多对 89、179、266 条，其余三景净少对 468、214、403 条。大上下文版在郊区高净多对 12 条，其余五景均下降。两项都不是六景全优，也没有全部达到 3% 目标。

以 raw clean 86.538690% 为基准，相对下降不超过 3% 要求每景至少 141024 条正确；DURA 六景分别正确 141697、139467、136621、139523、127186、115586 条，低城区达到该门槛须净多对 25438 条。以绝对 3 个百分点计算，阈值为 140345 条，低城区须净多对 24759 条。

## 完整清单

每项依次给出 clean 和六个受损场景相对 DURA 的净正确数（纠错减损伤）；负数表示候选正确数更少。

| 固定测试 | clean | 郊区高 | 郊区中 | 郊区低 | 城市高 | 城市中 | 城市低 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [20261106-phase1-adaptive-inverse-ground-test-s2026092701-r01](../20261106-phase1-adaptive-inverse-ground-test-s2026092701-r01/analysis.md) | -45991 | -49099 | -49290 | -48828 | -50425 | -53130 | -52271 |
| [20261106-phase1-direct-dualdomain-ground-test-s2026092701-r01](../20261106-phase1-direct-dualdomain-ground-test-s2026092701-r01/analysis.md) | -38302 | -40397 | -42156 | -43659 | -42285 | -49499 | -51719 |
| [20261106-phase1-direct-spectral-distill-ground-test-s2026092701-r01](../20261106-phase1-direct-spectral-distill-ground-test-s2026092701-r01/analysis.md) | -9686 | -41365 | -43415 | -44577 | -44318 | -51803 | -53931 |
| [20261106-phase1-direct-spectral-wide-distill-ground-test-s2026092701-r01](../20261106-phase1-direct-spectral-wide-distill-ground-test-s2026092701-r01/analysis.md) | 437 | -26710 | -29035 | -30380 | -30875 | -42657 | -47103 |
| [20261106-phase1-dura-band-guard-ground-test-s2026092701-r01](../20261106-phase1-dura-band-guard-ground-test-s2026092701-r01/analysis.md) | -258 | -495 | -522 | -549 | -563 | -690 | -710 |
| [20261106-phase1-dura-crosshalf-iq-ssl-ground-test-s2026092701-r01](../20261106-phase1-dura-crosshalf-iq-ssl-ground-test-s2026092701-r01/analysis.md) | 3 | -931 | -709 | -736 | -918 | -585 | -652 |
| [20261106-phase1-dura-crossview-guard-ground-test-s2026092701-r01](../20261106-phase1-dura-crossview-guard-ground-test-s2026092701-r01/analysis.md) | -245 | -971 | -798 | -782 | -943 | -815 | -740 |
| [20261106-phase1-dura-expert-mixture-ground-test-s2026092701-r01](../20261106-phase1-dura-expert-mixture-ground-test-s2026092701-r01/analysis.md) | 1 | -2850 | -2978 | -2991 | -2943 | -2788 | -2649 |
| [20261106-phase1-dura-fixed-iq-ssl-ground-test-s2026092701-r01](../20261106-phase1-dura-fixed-iq-ssl-ground-test-s2026092701-r01/analysis.md) | 1 | -1291 | -1121 | -1090 | -1311 | -969 | -878 |
| [20261106-phase1-dura-fixed-prior-action-ground-test-s2026092701-r01](../20261106-phase1-dura-fixed-prior-action-ground-test-s2026092701-r01/analysis.md) | -45 | -585 | -641 | -613 | -588 | -850 | -973 |
| [20261106-phase1-dura-internal-guard-ground-test-s2026092701-r02](../20261106-phase1-dura-internal-guard-ground-test-s2026092701-r02/analysis.md) | -80 | -938 | -1188 | -1158 | -963 | -1269 | -1568 |
| [20261106-phase1-dura-iq-geometry-ground-test-s2026092701-r01](../20261106-phase1-dura-iq-geometry-ground-test-s2026092701-r01/analysis.md) | 1 | -923 | -820 | -790 | -971 | -826 | -769 |
| [20261106-phase1-dura-joint-iq-ssl-ground-test-s2026092701-r01](../20261106-phase1-dura-joint-iq-ssl-ground-test-s2026092701-r01/analysis.md) | 1 | -1101 | -873 | -898 | -1127 | -928 | -966 |
| [20261106-phase1-dura-moment-relation-ground-test-s2026092701-r01](../20261106-phase1-dura-moment-relation-ground-test-s2026092701-r01/analysis.md) | 1 | -802 | -705 | -659 | -860 | -757 | -773 |
| [20261106-phase1-dura-multiscale-spectral-ground-test-s2026092701-r01](../20261106-phase1-dura-multiscale-spectral-ground-test-s2026092701-r01/analysis.md) | 1 | -971 | -796 | -797 | -973 | -796 | -830 |
| [20261106-phase1-dura-noise-gated-ground-test-s2026092701-r01](../20261106-phase1-dura-noise-gated-ground-test-s2026092701-r01/analysis.md) | 1 | -826 | -743 | -708 | -863 | -748 | -735 |
| [20261106-phase1-dura-nonlinear-guard-ground-test-s2026092701-r01](../20261106-phase1-dura-nonlinear-guard-ground-test-s2026092701-r01/analysis.md) | -240 | -592 | -548 | -597 | -620 | -739 | -764 |
| [20261106-phase1-dura-orthogonal-ground-test-s2026092701-r01](../20261106-phase1-dura-orthogonal-ground-test-s2026092701-r01/analysis.md) | -357 | -1118 | -889 | -822 | -1043 | -877 | -774 |
| [20261106-phase1-dura-orthogonal-guard-ground-test-s2026092701-r01](../20261106-phase1-dura-orthogonal-guard-ground-test-s2026092701-r01/analysis.md) | -252 | -662 | -601 | -638 | -683 | -778 | -760 |
| [20261106-phase1-dura-posterior-teacher-ground-test-s2026092701-r01](../20261106-phase1-dura-posterior-teacher-ground-test-s2026092701-r01/analysis.md) | 2 | -1010 | -865 | -780 | -962 | -751 | -673 |
| [20261106-phase1-dura-sim-action-ground-test-s2026092701-r01](../20261106-phase1-dura-sim-action-ground-test-s2026092701-r01/analysis.md) | -60 | -510 | -779 | -771 | -529 | -937 | -1381 |
| [20261106-phase1-dura-tail-excess-ground-test-s2026092701-r01](../20261106-phase1-dura-tail-excess-ground-test-s2026092701-r01/analysis.md) | 2 | -650 | -530 | -560 | -728 | -535 | -609 |
| [20261106-phase1-dura-timevarying-ground-test-s2026092701-r01](../20261106-phase1-dura-timevarying-ground-test-s2026092701-r01/analysis.md) | -26 | -620 | -533 | -500 | -603 | -505 | -603 |
| [20261106-phase1-dura-wide-source-ground-test-s2026092701-r01](../20261106-phase1-dura-wide-source-ground-test-s2026092701-r01/analysis.md) | 0 | -2436 | -2511 | -2493 | -2490 | -2561 | -2438 |
| [20261106-phase1-timevarying-inverse-ground-test-s2026092701-r01](../20261106-phase1-timevarying-inverse-ground-test-s2026092701-r01/analysis.md) | -63100 | -31141 | -33882 | -35089 | -34266 | -43511 | -46876 |
| [20261107-phase1-direct-circular-ground-test-s2026092701-r01](../20261107-phase1-direct-circular-ground-test-s2026092701-r01/analysis.md) | -37776 | -28550 | -29837 | -30120 | -31627 | -40225 | -43483 |
| [20261107-phase1-direct-iqflow-ground-test-s2026092701-r01](../20261107-phase1-direct-iqflow-ground-test-s2026092701-r01/analysis.md) | -81128 | -81538 | -79086 | -76302 | -80431 | -71887 | -64031 |
| [20261107-phase1-direct-multiview-distill-ground-test-s2026092701-r01](../20261107-phase1-direct-multiview-distill-ground-test-s2026092701-r01/analysis.md) | -3662 | -32685 | -34554 | -35264 | -36242 | -45930 | -49208 |
| [20261107-phase1-direct-phase-ground-test-s2026092701-r01](../20261107-phase1-direct-phase-ground-test-s2026092701-r01/analysis.md) | -29156 | -26162 | -27946 | -28813 | -29533 | -39037 | -43106 |
| [20261107-phase1-direct-student-fir-ground-test-s2026092701-r01](../20261107-phase1-direct-student-fir-ground-test-s2026092701-r01/analysis.md) | -733 | -27124 | -29198 | -30268 | -31212 | -42432 | -46593 |
| [20261107-phase1-dura-antithetic-flow-ground-test-s2026092701-r01](../20261107-phase1-dura-antithetic-flow-ground-test-s2026092701-r01/analysis.md) | -1 | -603 | -550 | -550 | -548 | -349 | -251 |
| [20261107-phase1-dura-antithetic-readout-ground-test-s2026092701-r01](../20261107-phase1-dura-antithetic-readout-ground-test-s2026092701-r01/report.md) | 0 | -1826 | -2160 | -2719 | -2225 | -3178 | -3875 |
| [20261107-phase1-dura-balanced-spectral-ground-test-s2026092701-r01](../20261107-phase1-dura-balanced-spectral-ground-test-s2026092701-r01/analysis.md) | 2 | -2091 | -2231 | -2460 | -2261 | -2453 | -2470 |
| [20261107-phase1-dura-circular-ground-test-s2026092701-r01](../20261107-phase1-dura-circular-ground-test-s2026092701-r01/analysis.md) | -1 | -451 | -341 | -329 | -400 | -163 | -140 |
| [20261107-phase1-dura-conditional-adversary-ground-test-s2026092701-r02](../20261107-phase1-dura-conditional-adversary-ground-test-s2026092701-r02/analysis.md) | 3 | -3179 | -3313 | -3461 | -3303 | -2945 | -2345 |
| [20261107-phase1-dura-conditional-phase-ground-test-s2026092701-r01](../20261107-phase1-dura-conditional-phase-ground-test-s2026092701-r01/analysis.md) | -1 | -580 | -501 | -394 | -520 | -249 | -125 |
| [20261107-phase1-dura-counterfactual-response-ground-test-s2026092701-r01](../20261107-phase1-dura-counterfactual-response-ground-test-s2026092701-r01/analysis.md) | 1 | -468 | -214 | 89 | -403 | 179 | 266 |
| [20261107-phase1-dura-expert-distill-ground-test-s2026092701-r01](../20261107-phase1-dura-expert-distill-ground-test-s2026092701-r01/analysis.md) | 0 | -2212 | -2269 | -2443 | -2490 | -2461 | -2184 |
| [20261107-phase1-dura-frequency-ground-test-s2026092701-r01](../20261107-phase1-dura-frequency-ground-test-s2026092701-r01/analysis.md) | -94 | -868 | -705 | -708 | -793 | -613 | -532 |
| [20261107-phase1-dura-instance-contrast-ground-test-s2026092701-r01](../20261107-phase1-dura-instance-contrast-ground-test-s2026092701-r01/analysis.md) | 0 | -1548 | -1474 | -1465 | -1427 | -1197 | -1129 |
| [20261107-phase1-dura-iqflow-ground-test-s2026092701-r01](../20261107-phase1-dura-iqflow-ground-test-s2026092701-r01/analysis.md) | -413 | -813 | -843 | -949 | -857 | -716 | -581 |
| [20261107-phase1-dura-iqflow-origin-ground-test-s2026092701-r01](../20261107-phase1-dura-iqflow-origin-ground-test-s2026092701-r01/analysis.md) | -1418 | -790 | -692 | -574 | -580 | -214 | -151 |
| [20261107-phase1-dura-lag-context-ground-test-s2026092701-r01](../20261107-phase1-dura-lag-context-ground-test-s2026092701-r01/analysis.md) | -1 | -140 | -103 | -146 | -87 | -53 | -23 |
| [20261107-phase1-dura-leaveout-multiview-ground-test-s2026092701-r01](../20261107-phase1-dura-leaveout-multiview-ground-test-s2026092701-r01/analysis.md) | 0 | -1813 | -2205 | -2450 | -2095 | -2496 | -2665 |
| [20261107-phase1-dura-masked-prior-ground-test-s2026092701-r01](../20261107-phase1-dura-masked-prior-ground-test-s2026092701-r01/analysis.md) | 0 | -2205 | -2273 | -2298 | -2310 | -2423 | -2345 |
| [20261107-phase1-dura-monotone-envelope-ground-test-s2026092701-r01](../20261107-phase1-dura-monotone-envelope-ground-test-s2026092701-r01/analysis.md) | -56 | -2509 | -2343 | -2183 | -2499 | -2187 | -1830 |
| [20261107-phase1-dura-phase-equivariant-ground-test-s2026092701-r01](../20261107-phase1-dura-phase-equivariant-ground-test-s2026092701-r01/analysis.md) | -1 | -517 | -403 | -339 | -447 | -202 | -103 |
| [20261107-phase1-dura-phase-moments-ground-test-s2026092701-r01](../20261107-phase1-dura-phase-moments-ground-test-s2026092701-r01/analysis.md) | -1 | -551 | -472 | -372 | -477 | -225 | -111 |
| [20261107-phase1-dura-quartic-ground-test-s2026092701-r02](../20261107-phase1-dura-quartic-ground-test-s2026092701-r02/analysis.md) | -1 | -2833 | -2852 | -2948 | -2965 | -2908 | -2663 |
| [20261107-phase1-dura-reversible-ground-test-s2026092701-r01](../20261107-phase1-dura-reversible-ground-test-s2026092701-r01/analysis.md) | -375 | -312 | -262 | -295 | -312 | -280 | -241 |
| [20261107-phase1-dura-supervised-phase-ground-test-s2026092701-r01](../20261107-phase1-dura-supervised-phase-ground-test-s2026092701-r01/analysis.md) | 0 | -177 | -168 | -191 | -142 | -107 | -87 |
| [20261108-phase1-dura-large-context-ground-test-s2026092701-r01](../20261108-phase1-dura-large-context-ground-test-s2026092701-r01/analysis.md) | 0 | 12 | -33 | -34 | -19 | -47 | -26 |
| [20261108-phase1-dura-large-context-tail-ground-test-s2026092701-r01](../20261108-phase1-dura-large-context-tail-ground-test-s2026092701-r01/analysis.md) | 0 | -1 | -31 | -34 | -23 | -77 | -29 |

以上是单 seed、反复观察过的地面模拟测试描述，不是全新独立盲测或真实在轨结果。局部正收益不证明因果机制或跨 seed 稳定性，也不用于改变已登记候选、训练参数或测试。
