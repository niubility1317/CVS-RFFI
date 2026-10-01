# 原地面A与联合LocalRidge方法：完整三阶段配对

全部4行已独立读回完成，共160个parent（每种方法160个）。旧类数6，新增类数0、2、5、10、20；K为1、5、10、20。
A使用原地面6类float32分类头；A/B/C旧类准确率来自同一source row、split与物理旧held support。B/C为原方法已固定预测，没有重拟合。
本报告是合法support持出诊断，不是query准确率或新增独立验证。A预测先固定，随后连接truth；结果不回流选模、参数或重跑。
H、注册下降、绝对新旧差先逐parent计算，再等parent平均；不以汇总准确率重新计算H或绝对差。true K1无独立held，记N/A。
理想目标：B−A≥10个百分点、B−C旧≤1个百分点、平均绝对新旧差≤3个百分点；当前为逐步改善方向，未增加硬门槛。

整体（OOF，K5/10/20、新增2/5/10/20，每方法96个可测parent）

| method | measured_parents | 旧类数 | 总注册类数 | A旧 | B旧 | C旧 | C新 | H | B−A | 注册下降 | 绝对差 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D92-MarginJointLocalRidge-v1 | 96 | 6 | 6+new | 62.6389 | 70.2083 | 66.5538 | 52.2734 | 57.8275 | 7.5694 | 3.6545 | 15.3220 |

完整K×新增类数

| method | diagnostic | k | new_count | measured_parents | 旧类数 | 总注册类数 | A旧 | B旧 | C旧 | C新 | H | B−A | 注册下降 | 绝对差 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D92-MarginJointLocalRidge-v1 | oof | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | 10 | 0 | 8 | 6 | 6 | 64.3750 | 72.0833 | 72.0833 | N/A | N/A | 7.7083 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | 10 | 10 | 8 | 6 | 16 | 64.3750 | 72.0833 | 68.9583 | 55.6250 | 61.4768 | 7.7083 | 3.1250 | 13.3333 |
| D92-MarginJointLocalRidge-v1 | oof | 10 | 2 | 8 | 6 | 8 | 64.3750 | 72.0833 | 71.4583 | 50.0000 | 57.6631 | 7.7083 | 0.6250 | 23.1250 |
| D92-MarginJointLocalRidge-v1 | oof | 10 | 20 | 8 | 6 | 26 | 64.3750 | 72.0833 | 67.9167 | 54.4375 | 60.3799 | 7.7083 | 4.1667 | 13.4792 |
| D92-MarginJointLocalRidge-v1 | oof | 10 | 5 | 8 | 6 | 11 | 64.3750 | 72.0833 | 69.3750 | 54.2500 | 60.3711 | 7.7083 | 2.7083 | 15.1250 |
| D92-MarginJointLocalRidge-v1 | oof | 20 | 0 | 8 | 6 | 6 | 61.4583 | 76.4583 | 76.4583 | N/A | N/A | 15.0000 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | 20 | 10 | 8 | 6 | 16 | 61.4583 | 76.4583 | 71.9792 | 59.7500 | 65.0278 | 15.0000 | 4.4792 | 12.2292 |
| D92-MarginJointLocalRidge-v1 | oof | 20 | 2 | 8 | 6 | 8 | 61.4583 | 76.4583 | 75.2083 | 51.8750 | 60.5651 | 15.0000 | 1.2500 | 23.3333 |
| D92-MarginJointLocalRidge-v1 | oof | 20 | 20 | 8 | 6 | 26 | 61.4583 | 76.4583 | 70.5208 | 60.3437 | 64.9104 | 15.0000 | 5.9375 | 10.1771 |
| D92-MarginJointLocalRidge-v1 | oof | 20 | 5 | 8 | 6 | 11 | 61.4583 | 76.4583 | 73.6458 | 57.5000 | 63.9920 | 15.0000 | 2.8125 | 16.1458 |
| D92-MarginJointLocalRidge-v1 | oof | 5 | 0 | 8 | 6 | 6 | 62.0833 | 62.0833 | 62.0833 | N/A | N/A | 0.0000 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | 5 | 10 | 8 | 6 | 16 | 62.0833 | 62.0833 | 56.6667 | 45.5000 | 49.8690 | 0.0000 | 5.4167 | 12.0000 |
| D92-MarginJointLocalRidge-v1 | oof | 5 | 2 | 8 | 6 | 8 | 62.0833 | 62.0833 | 61.2500 | 46.2500 | 50.2835 | 0.0000 | 0.8333 | 20.8333 |
| D92-MarginJointLocalRidge-v1 | oof | 5 | 20 | 8 | 6 | 26 | 62.0833 | 62.0833 | 53.7500 | 44.7500 | 48.2214 | 0.0000 | 8.3333 | 11.8333 |
| D92-MarginJointLocalRidge-v1 | oof | 5 | 5 | 8 | 6 | 11 | 62.0833 | 62.0833 | 57.9167 | 47.0000 | 51.1703 | 0.0000 | 4.1667 | 12.2500 |
| D92-MarginJointLocalRidge-v1 | proxy | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | 10 | 0 | 8 | 6 | 6 | 64.3750 | 53.0324 | 53.0324 | N/A | N/A | -11.3426 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | 10 | 10 | 8 | 6 | 16 | 64.3750 | 53.0324 | 48.0787 | 32.7639 | 38.2434 | -11.3426 | 4.9537 | 16.6759 |
| D92-MarginJointLocalRidge-v1 | proxy | 10 | 2 | 8 | 6 | 8 | 64.3750 | 53.0324 | 52.1296 | 20.9722 | 26.1998 | -11.3426 | 0.9028 | 32.6389 |
| D92-MarginJointLocalRidge-v1 | proxy | 10 | 20 | 8 | 6 | 26 | 64.3750 | 53.0324 | 45.8333 | 29.9514 | 35.4963 | -11.3426 | 7.1991 | 17.8819 |
| D92-MarginJointLocalRidge-v1 | proxy | 10 | 5 | 8 | 6 | 11 | 64.3750 | 53.0324 | 50.6713 | 30.9444 | 37.2502 | -11.3426 | 2.3611 | 20.1898 |
| D92-MarginJointLocalRidge-v1 | proxy | 20 | 0 | 8 | 6 | 6 | 61.4583 | 52.4726 | 52.4726 | N/A | N/A | -8.9857 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | 20 | 10 | 8 | 6 | 16 | 61.4583 | 52.4726 | 46.0800 | 32.2796 | 36.8615 | -8.9857 | 6.3925 | 15.9649 |
| D92-MarginJointLocalRidge-v1 | proxy | 20 | 2 | 8 | 6 | 8 | 61.4583 | 52.4726 | 50.8882 | 23.3059 | 28.5282 | -8.9857 | 1.5844 | 29.5998 |
| D92-MarginJointLocalRidge-v1 | proxy | 20 | 20 | 8 | 6 | 26 | 61.4583 | 52.4726 | 43.7061 | 30.0329 | 34.8775 | -8.9857 | 8.7664 | 15.2675 |
| D92-MarginJointLocalRidge-v1 | proxy | 20 | 5 | 8 | 6 | 11 | 61.4583 | 52.4726 | 48.6129 | 31.1579 | 36.5426 | -8.9857 | 3.8596 | 18.7971 |
| D92-MarginJointLocalRidge-v1 | proxy | 5 | 0 | 8 | 6 | 6 | 62.0833 | 50.9375 | 50.9375 | N/A | N/A | -11.1458 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | 5 | 10 | 8 | 6 | 16 | 62.0833 | 50.9375 | 46.2500 | 33.8125 | 38.1120 | -11.1458 | 4.6875 | 14.4792 |
| D92-MarginJointLocalRidge-v1 | proxy | 5 | 2 | 8 | 6 | 8 | 62.0833 | 50.9375 | 50.5208 | 27.5000 | 28.7583 | -11.1458 | 0.4167 | 30.3125 |
| D92-MarginJointLocalRidge-v1 | proxy | 5 | 20 | 8 | 6 | 26 | 62.0833 | 50.9375 | 44.2708 | 30.9375 | 35.2556 | -11.1458 | 6.6667 | 15.2292 |
| D92-MarginJointLocalRidge-v1 | proxy | 5 | 5 | 8 | 6 | 11 | 62.0833 | 50.9375 | 48.0208 | 33.7500 | 37.8817 | -11.1458 | 2.9167 | 17.6875 |

by_receiver_scene

| method | diagnostic | path | receiver | scenario | k | new_count | measured_parents | 旧类数 | 总注册类数 | A旧 | B旧 | C旧 | C新 | H | B−A | 注册下降 | 绝对差 |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 10 | 0 | 2 | 6 | 6 | 73.3333 | 96.6667 | 96.6667 | N/A | N/A | 23.3333 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 10 | 10 | 2 | 6 | 16 | 73.3333 | 96.6667 | 91.6667 | 78.5000 | 84.5526 | 23.3333 | 5.0000 | 13.1667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 10 | 2 | 2 | 6 | 8 | 73.3333 | 96.6667 | 96.6667 | 77.5000 | 86.0239 | 23.3333 | 0.0000 | 19.1667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 10 | 20 | 2 | 6 | 26 | 73.3333 | 96.6667 | 91.6667 | 77.0000 | 83.6802 | 23.3333 | 5.0000 | 14.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 10 | 5 | 2 | 6 | 11 | 73.3333 | 96.6667 | 90.8333 | 87.0000 | 88.8659 | 23.3333 | 5.8333 | 3.8333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 20 | 0 | 2 | 6 | 6 | 64.1667 | 95.4167 | 95.4167 | N/A | N/A | 31.2500 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 20 | 10 | 2 | 6 | 16 | 64.1667 | 95.4167 | 92.0833 | 85.7500 | 88.8038 | 31.2500 | 3.3333 | 6.3333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 20 | 2 | 2 | 6 | 8 | 64.1667 | 95.4167 | 95.0000 | 81.2500 | 87.5760 | 31.2500 | 0.4167 | 13.7500 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 20 | 20 | 2 | 6 | 26 | 64.1667 | 95.4167 | 91.2500 | 84.6250 | 87.8125 | 31.2500 | 4.1667 | 6.6250 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 20 | 5 | 2 | 6 | 11 | 64.1667 | 95.4167 | 92.5000 | 87.0000 | 89.6539 | 31.2500 | 2.9167 | 5.5000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 5 | 0 | 2 | 6 | 6 | 70.0000 | 88.3333 | 88.3333 | N/A | N/A | 18.3333 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 5 | 10 | 2 | 6 | 16 | 70.0000 | 88.3333 | 81.6667 | 76.0000 | 78.6266 | 18.3333 | 6.6667 | 5.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 5 | 2 | 2 | 6 | 8 | 70.0000 | 88.3333 | 86.6667 | 75.0000 | 80.1913 | 18.3333 | 1.6667 | 11.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 5 | 20 | 2 | 6 | 26 | 70.0000 | 88.3333 | 80.0000 | 72.0000 | 75.6983 | 18.3333 | 8.3333 | 8.0000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_high | 5 | 5 | 2 | 6 | 11 | 70.0000 | 88.3333 | 83.3333 | 86.0000 | 84.6342 | 18.3333 | 5.0000 | 2.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 10 | 0 | 2 | 6 | 6 | 39.1667 | 47.5000 | 47.5000 | N/A | N/A | 8.3333 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 10 | 10 | 2 | 6 | 16 | 39.1667 | 47.5000 | 44.1667 | 33.0000 | 37.6639 | 8.3333 | 3.3333 | 11.1667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 10 | 2 | 2 | 6 | 8 | 39.1667 | 47.5000 | 45.0000 | 47.5000 | 46.1234 | 8.3333 | 2.5000 | 4.1667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 10 | 20 | 2 | 6 | 26 | 39.1667 | 47.5000 | 42.5000 | 31.0000 | 35.7885 | 8.3333 | 5.0000 | 11.5000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 10 | 5 | 2 | 6 | 11 | 39.1667 | 47.5000 | 44.1667 | 34.0000 | 38.4071 | 8.3333 | 3.3333 | 10.1667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 20 | 0 | 2 | 6 | 6 | 39.1667 | 55.8333 | 55.8333 | N/A | N/A | 16.6667 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 20 | 10 | 2 | 6 | 16 | 39.1667 | 55.8333 | 47.5000 | 34.0000 | 39.5757 | 16.6667 | 8.3333 | 13.5000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 20 | 2 | 2 | 6 | 8 | 39.1667 | 55.8333 | 53.7500 | 26.2500 | 35.1311 | 16.6667 | 2.0833 | 27.5000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 20 | 20 | 2 | 6 | 26 | 39.1667 | 55.8333 | 45.4167 | 34.5000 | 39.2035 | 16.6667 | 10.4167 | 10.9167 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 20 | 5 | 2 | 6 | 11 | 39.1667 | 55.8333 | 50.4167 | 31.5000 | 38.7694 | 16.6667 | 5.4167 | 18.9167 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 5 | 0 | 2 | 6 | 6 | 38.3333 | 33.3333 | 33.3333 | N/A | N/A | -5.0000 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 5 | 10 | 2 | 6 | 16 | 38.3333 | 33.3333 | 28.3333 | 30.0000 | 29.1176 | -5.0000 | 5.0000 | 1.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 5 | 2 | 2 | 6 | 8 | 38.3333 | 33.3333 | 31.6667 | 25.0000 | 27.5000 | -5.0000 | 1.6667 | 6.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 5 | 20 | 2 | 6 | 26 | 38.3333 | 33.3333 | 25.0000 | 23.0000 | 23.7259 | -5.0000 | 8.3333 | 4.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 19-1 | practical_low_urban | 5 | 5 | 2 | 6 | 11 | 38.3333 | 33.3333 | 28.3333 | 20.0000 | 23.3333 | -5.0000 | 5.0000 | 8.3333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 10 | 0 | 2 | 6 | 6 | 91.6667 | 96.6667 | 96.6667 | N/A | N/A | 5.0000 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 10 | 10 | 2 | 6 | 16 | 91.6667 | 96.6667 | 96.6667 | 80.0000 | 87.5472 | 5.0000 | 0.0000 | 16.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 10 | 2 | 2 | 6 | 8 | 91.6667 | 96.6667 | 96.6667 | 50.0000 | 65.9091 | 5.0000 | 0.0000 | 46.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 10 | 20 | 2 | 6 | 26 | 91.6667 | 96.6667 | 95.8333 | 75.5000 | 84.4552 | 5.0000 | 0.8333 | 20.3333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 10 | 5 | 2 | 6 | 11 | 91.6667 | 96.6667 | 96.6667 | 71.0000 | 81.8648 | 5.0000 | 0.0000 | 25.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 20 | 0 | 2 | 6 | 6 | 87.9167 | 97.0833 | 97.0833 | N/A | N/A | 9.1667 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 20 | 10 | 2 | 6 | 16 | 87.9167 | 97.0833 | 95.4167 | 83.5000 | 89.0602 | 9.1667 | 1.6667 | 11.9167 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 20 | 2 | 2 | 6 | 8 | 87.9167 | 97.0833 | 96.2500 | 72.5000 | 82.7033 | 9.1667 | 0.8333 | 23.7500 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 20 | 20 | 2 | 6 | 26 | 87.9167 | 97.0833 | 95.8333 | 84.7500 | 89.9508 | 9.1667 | 1.2500 | 11.0833 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 20 | 5 | 2 | 6 | 11 | 87.9167 | 97.0833 | 95.8333 | 79.5000 | 86.8641 | 9.1667 | 1.2500 | 16.3333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 5 | 0 | 2 | 6 | 6 | 91.6667 | 93.3333 | 93.3333 | N/A | N/A | 1.6667 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 5 | 10 | 2 | 6 | 16 | 91.6667 | 93.3333 | 90.0000 | 62.0000 | 73.4211 | 1.6667 | 3.3333 | 28.0000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 5 | 2 | 2 | 6 | 8 | 91.6667 | 93.3333 | 93.3333 | 40.0000 | 55.2608 | 1.6667 | 0.0000 | 53.3333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 5 | 20 | 2 | 6 | 26 | 91.6667 | 93.3333 | 90.0000 | 60.0000 | 71.9952 | 1.6667 | 3.3333 | 30.0000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_high | 5 | 5 | 2 | 6 | 11 | 91.6667 | 93.3333 | 88.3333 | 64.0000 | 74.1887 | 1.6667 | 5.0000 | 24.3333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 10 | 0 | 2 | 6 | 6 | 53.3333 | 47.5000 | 47.5000 | N/A | N/A | -5.8333 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 10 | 10 | 2 | 6 | 16 | 53.3333 | 47.5000 | 43.3333 | 31.0000 | 36.1435 | -5.8333 | 4.1667 | 12.3333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 10 | 2 | 2 | 6 | 8 | 53.3333 | 47.5000 | 47.5000 | 25.0000 | 32.5962 | -5.8333 | 0.0000 | 22.5000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 10 | 20 | 2 | 6 | 26 | 53.3333 | 47.5000 | 41.6667 | 34.2500 | 37.5956 | -5.8333 | 5.8333 | 7.4167 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 10 | 5 | 2 | 6 | 11 | 53.3333 | 47.5000 | 45.8333 | 25.0000 | 32.3466 | -5.8333 | 1.6667 | 20.8333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 20 | 0 | 2 | 6 | 6 | 54.5833 | 57.5000 | 57.5000 | N/A | N/A | 2.9167 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 20 | 10 | 2 | 6 | 16 | 54.5833 | 57.5000 | 52.9167 | 35.7500 | 42.6715 | 2.9167 | 4.5833 | 17.1667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 20 | 2 | 2 | 6 | 8 | 54.5833 | 57.5000 | 55.8333 | 27.5000 | 36.8500 | 2.9167 | 1.6667 | 28.3333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 20 | 20 | 2 | 6 | 26 | 54.5833 | 57.5000 | 49.5833 | 37.5000 | 42.6748 | 2.9167 | 7.9167 | 12.0833 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 20 | 5 | 2 | 6 | 11 | 54.5833 | 57.5000 | 55.8333 | 32.0000 | 40.6806 | 2.9167 | 1.6667 | 23.8333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 5 | 0 | 2 | 6 | 6 | 48.3333 | 33.3333 | 33.3333 | N/A | N/A | -15.0000 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 5 | 10 | 2 | 6 | 16 | 48.3333 | 33.3333 | 26.6667 | 14.0000 | 18.3108 | -15.0000 | 6.6667 | 12.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 5 | 2 | 2 | 6 | 8 | 48.3333 | 33.3333 | 33.3333 | 45.0000 | 38.1818 | -15.0000 | 0.0000 | 11.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 5 | 20 | 2 | 6 | 26 | 48.3333 | 33.3333 | 20.0000 | 24.0000 | 21.4664 | -15.0000 | 13.3333 | 4.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 20-19 | practical_low_urban | 5 | 5 | 2 | 6 | 11 | 48.3333 | 33.3333 | 31.6667 | 18.0000 | 22.5249 | -15.0000 | 1.6667 | 13.6667 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 10 | 0 | 2 | 6 | 6 | 73.3333 | 72.4074 | 72.4074 | N/A | N/A | -0.9259 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 10 | 10 | 2 | 6 | 16 | 73.3333 | 72.4074 | 69.0741 | 51.6667 | 58.6806 | -0.9259 | 3.3333 | 18.3704 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 10 | 2 | 2 | 6 | 8 | 73.3333 | 72.4074 | 73.1481 | 35.2778 | 44.0618 | -0.9259 | -0.7407 | 37.8704 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 10 | 20 | 2 | 6 | 26 | 73.3333 | 72.4074 | 65.9259 | 46.1944 | 53.6932 | -0.9259 | 6.4815 | 22.0648 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 10 | 5 | 2 | 6 | 11 | 73.3333 | 72.4074 | 70.3704 | 53.8889 | 60.4704 | -0.9259 | 2.0370 | 16.5926 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 20 | 0 | 2 | 6 | 6 | 64.1667 | 69.0132 | 69.0132 | N/A | N/A | 4.8465 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 20 | 10 | 2 | 6 | 16 | 64.1667 | 69.0132 | 64.7588 | 54.4737 | 58.7518 | 4.8465 | 4.2544 | 12.0658 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 20 | 2 | 2 | 6 | 8 | 64.1667 | 69.0132 | 68.6842 | 47.6316 | 54.6335 | 4.8465 | 0.3289 | 21.5789 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 20 | 20 | 2 | 6 | 26 | 64.1667 | 69.0132 | 61.3158 | 48.2895 | 53.6887 | 4.8465 | 7.6974 | 14.0000 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 20 | 5 | 2 | 6 | 11 | 64.1667 | 69.0132 | 66.3158 | 51.9474 | 57.7175 | 4.8465 | 2.6974 | 15.0263 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 5 | 0 | 2 | 6 | 6 | 70.0000 | 66.6667 | 66.6667 | N/A | N/A | -3.3333 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 5 | 10 | 2 | 6 | 16 | 70.0000 | 66.6667 | 64.5833 | 55.5000 | 59.2539 | -3.3333 | 2.0833 | 10.7500 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 5 | 2 | 2 | 6 | 8 | 70.0000 | 66.6667 | 67.9167 | 62.5000 | 63.7977 | -3.3333 | -1.2500 | 14.5833 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 5 | 20 | 2 | 6 | 26 | 70.0000 | 66.6667 | 60.0000 | 51.2500 | 55.1119 | -3.3333 | 6.6667 | 8.7500 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_high | 5 | 5 | 2 | 6 | 11 | 70.0000 | 66.6667 | 65.0000 | 66.5000 | 64.9121 | -3.3333 | 1.6667 | 11.8333 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 10 | 0 | 2 | 6 | 6 | 39.1667 | 25.2778 | 25.2778 | N/A | N/A | -13.8889 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 10 | 10 | 2 | 6 | 16 | 39.1667 | 25.2778 | 19.2593 | 15.3889 | 16.1812 | -13.8889 | 6.0185 | 7.2407 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 10 | 2 | 2 | 6 | 8 | 39.1667 | 25.2778 | 23.2407 | 16.6667 | 16.8793 | -13.8889 | 2.0370 | 11.2037 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 10 | 20 | 2 | 6 | 26 | 39.1667 | 25.2778 | 17.0370 | 14.2778 | 15.1175 | -13.8889 | 8.2407 | 5.2593 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 10 | 5 | 2 | 6 | 11 | 39.1667 | 25.2778 | 21.9444 | 13.3333 | 15.5954 | -13.8889 | 3.3333 | 9.7963 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 20 | 0 | 2 | 6 | 6 | 39.1667 | 26.0965 | 26.0965 | N/A | N/A | -13.0702 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 20 | 10 | 2 | 6 | 16 | 39.1667 | 26.0965 | 18.6184 | 14.8553 | 15.2719 | -13.0702 | 7.4781 | 8.1228 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 20 | 2 | 2 | 6 | 8 | 39.1667 | 26.0965 | 22.7193 | 14.5395 | 15.2297 | -13.0702 | 3.3772 | 12.4781 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 20 | 20 | 2 | 6 | 26 | 39.1667 | 26.0965 | 16.0965 | 13.0000 | 13.6097 | -13.0702 | 10.0000 | 6.5263 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 20 | 5 | 2 | 6 | 11 | 39.1667 | 26.0965 | 20.9211 | 14.2368 | 15.3873 | -13.0702 | 5.1754 | 9.1842 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 5 | 0 | 2 | 6 | 6 | 38.3333 | 25.8333 | 25.8333 | N/A | N/A | -12.5000 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 5 | 10 | 2 | 6 | 16 | 38.3333 | 25.8333 | 17.5000 | 14.5000 | 15.0470 | -12.5000 | 8.3333 | 6.8333 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 5 | 2 | 2 | 6 | 8 | 38.3333 | 25.8333 | 23.7500 | 7.5000 | 9.2292 | -12.5000 | 2.0833 | 16.2500 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 5 | 20 | 2 | 6 | 26 | 38.3333 | 25.8333 | 15.8333 | 13.6250 | 13.8731 | -12.5000 | 10.0000 | 5.4583 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 19-1 | practical_low_urban | 5 | 5 | 2 | 6 | 11 | 38.3333 | 25.8333 | 20.4167 | 8.5000 | 11.6990 | -12.5000 | 5.4167 | 11.9167 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 10 | 0 | 2 | 6 | 6 | 91.6667 | 86.6667 | 86.6667 | N/A | N/A | -5.0000 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 10 | 10 | 2 | 6 | 16 | 91.6667 | 86.6667 | 83.3333 | 50.3333 | 62.6100 | -5.0000 | 3.3333 | 33.0000 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 10 | 2 | 2 | 6 | 8 | 91.6667 | 86.6667 | 87.0370 | 17.7778 | 27.7050 | -5.0000 | -0.3704 | 69.2593 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 10 | 20 | 2 | 6 | 26 | 91.6667 | 86.6667 | 81.2963 | 44.3611 | 57.2565 | -5.0000 | 5.3704 | 36.9352 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 10 | 5 | 2 | 6 | 11 | 91.6667 | 86.6667 | 85.6481 | 44.7778 | 58.5026 | -5.0000 | 1.0185 | 40.8704 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 20 | 0 | 2 | 6 | 6 | 87.9167 | 84.9342 | 84.9342 | N/A | N/A | -2.9825 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 20 | 10 | 2 | 6 | 16 | 87.9167 | 84.9342 | 78.5965 | 45.2368 | 56.9723 | -2.9825 | 6.3377 | 33.3596 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 20 | 2 | 2 | 6 | 8 | 87.9167 | 84.9342 | 84.5395 | 18.5526 | 29.5798 | -2.9825 | 0.3947 | 65.9868 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 20 | 20 | 2 | 6 | 26 | 87.9167 | 84.9342 | 77.0833 | 44.3816 | 55.9805 | -2.9825 | 7.8509 | 32.7018 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 20 | 5 | 2 | 6 | 11 | 87.9167 | 84.9342 | 81.5570 | 44.5263 | 57.0240 | -2.9825 | 3.3772 | 37.0307 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 5 | 0 | 2 | 6 | 6 | 91.6667 | 90.8333 | 90.8333 | N/A | N/A | -0.8333 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 5 | 10 | 2 | 6 | 16 | 91.6667 | 90.8333 | 86.6667 | 52.7500 | 64.8308 | -0.8333 | 4.1667 | 34.0833 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 5 | 2 | 2 | 6 | 8 | 91.6667 | 90.8333 | 90.4167 | 13.7500 | 21.7112 | -0.8333 | 0.4167 | 76.6667 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 5 | 20 | 2 | 6 | 26 | 91.6667 | 90.8333 | 84.1667 | 42.3750 | 55.7662 | -0.8333 | 6.6667 | 41.7917 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_high | 5 | 5 | 2 | 6 | 11 | 91.6667 | 90.8333 | 86.6667 | 48.0000 | 61.1600 | -0.8333 | 4.1667 | 38.6667 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 10 | 0 | 2 | 6 | 6 | 53.3333 | 27.7778 | 27.7778 | N/A | N/A | -25.5556 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 10 | 10 | 2 | 6 | 16 | 53.3333 | 27.7778 | 20.6481 | 13.6667 | 15.5018 | -25.5556 | 7.1296 | 8.0926 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 10 | 2 | 2 | 6 | 8 | 53.3333 | 27.7778 | 25.0926 | 14.1667 | 16.1532 | -25.5556 | 2.6852 | 12.2222 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 10 | 20 | 2 | 6 | 26 | 53.3333 | 27.7778 | 19.0741 | 14.9722 | 15.9181 | -25.5556 | 8.7037 | 7.2685 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 10 | 5 | 2 | 6 | 11 | 53.3333 | 27.7778 | 24.7222 | 11.7778 | 14.4322 | -25.5556 | 3.0556 | 13.5000 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 20 | 0 | 2 | 6 | 6 | 54.5833 | 29.8465 | 29.8465 | N/A | N/A | -24.7368 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 20 | 10 | 2 | 6 | 16 | 54.5833 | 29.8465 | 22.3465 | 14.5526 | 16.4500 | -24.7368 | 7.5000 | 10.3114 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 20 | 2 | 2 | 6 | 8 | 54.5833 | 29.8465 | 27.6096 | 12.5000 | 14.6697 | -24.7368 | 2.2368 | 18.3553 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 20 | 20 | 2 | 6 | 26 | 54.5833 | 29.8465 | 20.3289 | 14.4605 | 16.2312 | -24.7368 | 9.5175 | 7.8421 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 20 | 5 | 2 | 6 | 11 | 54.5833 | 29.8465 | 25.6579 | 13.9211 | 16.0415 | -24.7368 | 4.1886 | 13.9474 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 5 | 0 | 2 | 6 | 6 | 48.3333 | 20.4167 | 20.4167 | N/A | N/A | -27.9167 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 5 | 10 | 2 | 6 | 16 | 48.3333 | 20.4167 | 16.2500 | 12.5000 | 13.3164 | -27.9167 | 4.1667 | 6.2500 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 5 | 2 | 2 | 6 | 8 | 48.3333 | 20.4167 | 20.0000 | 26.2500 | 20.2951 | -27.9167 | 0.4167 | 13.7500 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 5 | 20 | 2 | 6 | 26 | 48.3333 | 20.4167 | 17.0833 | 16.5000 | 16.2712 | -27.9167 | 3.3333 | 4.9167 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 20-19 | practical_low_urban | 5 | 5 | 2 | 6 | 11 | 48.3333 | 20.4167 | 20.0000 | 12.0000 | 13.7556 | -27.9167 | 0.4167 | 8.3333 |

by_model_row

| method | diagnostic | path | model_seed | row_id | k | new_count | measured_parents | 旧类数 | 总注册类数 | A旧 | B旧 | C旧 | C新 | H | B−A | 注册下降 | 绝对差 |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 0 | 2 | 6 | 6 | 70.8333 | 71.6667 | 71.6667 | N/A | N/A | 0.8333 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 10 | 2 | 6 | 16 | 70.8333 | 71.6667 | 70.0000 | 55.5000 | 61.8453 | 0.8333 | 1.6667 | 14.5000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 2 | 2 | 6 | 8 | 70.8333 | 71.6667 | 70.8333 | 35.0000 | 46.8007 | 0.8333 | 0.8333 | 35.8333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 20 | 2 | 6 | 26 | 70.8333 | 71.6667 | 69.1667 | 54.5000 | 60.9555 | 0.8333 | 2.5000 | 14.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 5 | 2 | 6 | 11 | 70.8333 | 71.6667 | 69.1667 | 47.0000 | 55.6627 | 0.8333 | 2.5000 | 22.1667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 0 | 2 | 6 | 6 | 70.0000 | 77.5000 | 77.5000 | N/A | N/A | 7.5000 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 10 | 2 | 6 | 16 | 70.0000 | 77.5000 | 74.1667 | 59.2500 | 65.6475 | 7.5000 | 3.3333 | 14.9167 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 2 | 2 | 6 | 8 | 70.0000 | 77.5000 | 75.8333 | 50.0000 | 59.6998 | 7.5000 | 1.6667 | 25.8333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 20 | 2 | 6 | 26 | 70.0000 | 77.5000 | 71.2500 | 61.2500 | 65.8312 | 7.5000 | 6.2500 | 10.0000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 5 | 2 | 6 | 11 | 70.0000 | 77.5000 | 75.4167 | 53.5000 | 62.2116 | 7.5000 | 2.0833 | 21.9167 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 0 | 2 | 6 | 6 | 68.3333 | 65.0000 | 65.0000 | N/A | N/A | -3.3333 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 10 | 2 | 6 | 16 | 68.3333 | 65.0000 | 61.6667 | 39.0000 | 47.5213 | -3.3333 | 3.3333 | 22.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 2 | 2 | 6 | 8 | 68.3333 | 65.0000 | 63.3333 | 50.0000 | 52.5581 | -3.3333 | 1.6667 | 30.0000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 20 | 2 | 6 | 26 | 68.3333 | 65.0000 | 58.3333 | 43.5000 | 49.5222 | -3.3333 | 6.6667 | 14.8333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 5 | 2 | 6 | 11 | 68.3333 | 65.0000 | 61.6667 | 46.0000 | 52.6877 | -3.3333 | 3.3333 | 15.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 0 | 2 | 6 | 6 | 54.1667 | 70.8333 | 70.8333 | N/A | N/A | 16.6667 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 10 | 2 | 6 | 16 | 54.1667 | 70.8333 | 68.3333 | 53.0000 | 59.5507 | 16.6667 | 2.5000 | 15.3333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 2 | 2 | 6 | 8 | 54.1667 | 70.8333 | 70.8333 | 60.0000 | 64.8209 | 16.6667 | 0.0000 | 10.8333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 20 | 2 | 6 | 26 | 54.1667 | 70.8333 | 66.6667 | 51.2500 | 57.8320 | 16.6667 | 4.1667 | 15.4167 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 5 | 2 | 6 | 11 | 54.1667 | 70.8333 | 66.6667 | 60.0000 | 62.9015 | 16.6667 | 4.1667 | 6.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 0 | 2 | 6 | 6 | 50.4167 | 75.8333 | 75.8333 | N/A | N/A | 25.4167 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 10 | 2 | 6 | 16 | 50.4167 | 75.8333 | 69.5833 | 61.0000 | 64.8884 | 25.4167 | 6.2500 | 8.5833 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 2 | 2 | 6 | 8 | 50.4167 | 75.8333 | 74.1667 | 56.2500 | 63.2810 | 25.4167 | 1.6667 | 17.9167 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 20 | 2 | 6 | 26 | 50.4167 | 75.8333 | 67.5000 | 59.5000 | 63.1477 | 25.4167 | 8.3333 | 8.0000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 5 | 2 | 6 | 11 | 50.4167 | 75.8333 | 72.0833 | 61.0000 | 65.4959 | 25.4167 | 3.7500 | 11.0833 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 0 | 2 | 6 | 6 | 53.3333 | 63.3333 | 63.3333 | N/A | N/A | 10.0000 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 10 | 2 | 6 | 16 | 53.3333 | 63.3333 | 56.6667 | 51.0000 | 53.6266 | 10.0000 | 6.6667 | 5.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 2 | 2 | 6 | 8 | 53.3333 | 63.3333 | 61.6667 | 45.0000 | 51.8750 | 10.0000 | 1.6667 | 16.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 20 | 2 | 6 | 26 | 53.3333 | 63.3333 | 53.3333 | 43.5000 | 47.8912 | 10.0000 | 10.0000 | 9.8333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 5 | 2 | 6 | 11 | 53.3333 | 63.3333 | 56.6667 | 54.0000 | 55.1660 | 10.0000 | 6.6667 | 3.3333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 0 | 2 | 6 | 6 | 74.1667 | 72.5000 | 72.5000 | N/A | N/A | -1.6667 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 10 | 2 | 6 | 16 | 74.1667 | 72.5000 | 70.0000 | 55.5000 | 61.8453 | -1.6667 | 2.5000 | 14.5000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 2 | 2 | 6 | 8 | 74.1667 | 72.5000 | 73.3333 | 40.0000 | 51.7045 | -1.6667 | -0.8333 | 33.3333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 20 | 2 | 6 | 26 | 74.1667 | 72.5000 | 68.3333 | 55.2500 | 61.0953 | -1.6667 | 4.1667 | 13.0833 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 5 | 2 | 6 | 11 | 74.1667 | 72.5000 | 73.3333 | 49.0000 | 58.5487 | -1.6667 | -0.8333 | 24.3333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 0 | 2 | 6 | 6 | 72.5000 | 77.0833 | 77.0833 | N/A | N/A | 4.5833 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 10 | 2 | 6 | 16 | 72.5000 | 77.0833 | 74.1667 | 60.0000 | 66.0842 | 4.5833 | 2.9167 | 14.1667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 2 | 2 | 6 | 8 | 72.5000 | 77.0833 | 76.2500 | 50.0000 | 59.8536 | 4.5833 | 0.8333 | 26.2500 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 20 | 2 | 6 | 26 | 72.5000 | 77.0833 | 74.1667 | 61.0000 | 66.7944 | 4.5833 | 2.9167 | 13.1667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 5 | 2 | 6 | 11 | 72.5000 | 77.0833 | 76.2500 | 58.0000 | 65.3331 | 4.5833 | 0.8333 | 18.2500 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 0 | 2 | 6 | 6 | 71.6667 | 61.6667 | 61.6667 | N/A | N/A | -10.0000 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 10 | 2 | 6 | 16 | 71.6667 | 61.6667 | 55.0000 | 37.0000 | 44.2105 | -10.0000 | 6.6667 | 18.0000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 2 | 2 | 6 | 8 | 71.6667 | 61.6667 | 63.3333 | 35.0000 | 40.8845 | -10.0000 | -1.6667 | 35.0000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 20 | 2 | 6 | 26 | 71.6667 | 61.6667 | 51.6667 | 40.5000 | 43.9395 | -10.0000 | 10.0000 | 19.8333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 5 | 2 | 6 | 11 | 71.6667 | 61.6667 | 58.3333 | 36.0000 | 44.0260 | -10.0000 | 3.3333 | 22.3333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 0 | 2 | 6 | 6 | 58.3333 | 73.3333 | 73.3333 | N/A | N/A | 15.0000 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 10 | 2 | 6 | 16 | 58.3333 | 73.3333 | 67.5000 | 58.5000 | 62.6658 | 15.0000 | 5.8333 | 9.0000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 2 | 2 | 6 | 8 | 58.3333 | 73.3333 | 70.8333 | 65.0000 | 67.3264 | 15.0000 | 2.5000 | 12.5000 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 20 | 2 | 6 | 26 | 58.3333 | 73.3333 | 67.5000 | 56.7500 | 61.6367 | 15.0000 | 5.8333 | 10.7500 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 5 | 2 | 6 | 11 | 58.3333 | 73.3333 | 68.3333 | 61.0000 | 64.3715 | 15.0000 | 5.0000 | 7.3333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 0 | 2 | 6 | 6 | 52.9167 | 75.4167 | 75.4167 | N/A | N/A | 22.5000 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 10 | 2 | 6 | 16 | 52.9167 | 75.4167 | 70.0000 | 58.7500 | 63.4911 | 22.5000 | 5.4167 | 11.2500 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 2 | 2 | 6 | 8 | 52.9167 | 75.4167 | 74.5833 | 51.2500 | 59.4261 | 22.5000 | 0.8333 | 23.3333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 20 | 2 | 6 | 26 | 52.9167 | 75.4167 | 69.1667 | 59.6250 | 63.8684 | 22.5000 | 6.2500 | 9.5417 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 5 | 2 | 6 | 11 | 52.9167 | 75.4167 | 70.8333 | 57.5000 | 62.9274 | 22.5000 | 4.5833 | 13.3333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 0 | 2 | 6 | 6 | 55.0000 | 58.3333 | 58.3333 | N/A | N/A | 3.3333 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 10 | 2 | 6 | 16 | 55.0000 | 58.3333 | 53.3333 | 55.0000 | 54.1176 | 3.3333 | 5.0000 | 1.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 2 | 2 | 6 | 8 | 55.0000 | 58.3333 | 56.6667 | 55.0000 | 55.8163 | 3.3333 | 1.6667 | 1.6667 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 20 | 2 | 6 | 26 | 55.0000 | 58.3333 | 51.6667 | 51.5000 | 51.5330 | 3.3333 | 6.6667 | 2.8333 |
| D92-MarginJointLocalRidge-v1 | oof | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 5 | 2 | 6 | 11 | 55.0000 | 58.3333 | 55.0000 | 52.0000 | 52.8016 | 3.3333 | 3.3333 | 7.6667 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 0 | 2 | 6 | 6 | 70.8333 | 58.1481 | 58.1481 | N/A | N/A | -12.6852 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 10 | 2 | 6 | 16 | 70.8333 | 58.1481 | 52.5000 | 31.0000 | 38.5275 | -12.6852 | 5.6481 | 21.7963 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 2 | 2 | 6 | 8 | 70.8333 | 58.1481 | 56.7593 | 14.7222 | 19.9145 | -12.6852 | 1.3889 | 43.1481 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 20 | 2 | 6 | 26 | 70.8333 | 58.1481 | 50.5556 | 29.1111 | 36.2431 | -12.6852 | 7.5926 | 22.9630 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 5 | 2 | 6 | 11 | 70.8333 | 58.1481 | 55.6481 | 26.6667 | 35.3025 | -12.6852 | 2.5000 | 29.1296 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 0 | 2 | 6 | 6 | 70.0000 | 57.3904 | 57.3904 | N/A | N/A | -12.6096 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 10 | 2 | 6 | 16 | 70.0000 | 57.3904 | 49.7368 | 29.0921 | 35.8703 | -12.6096 | 7.6535 | 21.8202 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 2 | 2 | 6 | 8 | 70.0000 | 57.3904 | 55.8333 | 15.1974 | 21.6159 | -12.6096 | 1.5570 | 42.2588 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 20 | 2 | 6 | 26 | 70.0000 | 57.3904 | 48.1579 | 29.3026 | 35.8723 | -12.6096 | 9.2325 | 19.7588 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 5 | 2 | 6 | 11 | 70.0000 | 57.3904 | 53.0921 | 27.5789 | 34.9806 | -12.6096 | 4.2982 | 26.4868 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 0 | 2 | 6 | 6 | 68.3333 | 54.5833 | 54.5833 | N/A | N/A | -13.7500 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 10 | 2 | 6 | 16 | 68.3333 | 54.5833 | 50.0000 | 32.7500 | 38.6764 | -13.7500 | 4.5833 | 18.4167 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 2 | 2 | 6 | 8 | 68.3333 | 54.5833 | 54.1667 | 18.7500 | 20.7090 | -13.7500 | 0.4167 | 44.5833 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 20 | 2 | 6 | 26 | 68.3333 | 54.5833 | 50.0000 | 29.5000 | 35.9519 | -13.7500 | 4.5833 | 22.3333 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 5 | 2 | 6 | 11 | 68.3333 | 54.5833 | 52.5000 | 30.0000 | 37.2540 | -13.7500 | 2.0833 | 22.8333 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 0 | 2 | 6 | 6 | 54.1667 | 49.5370 | 49.5370 | N/A | N/A | -4.6296 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 10 | 2 | 6 | 16 | 54.1667 | 49.5370 | 44.6296 | 33.5000 | 37.5679 | -4.6296 | 4.9074 | 13.4259 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 2 | 2 | 6 | 8 | 54.1667 | 49.5370 | 48.1481 | 26.9444 | 31.3964 | -4.6296 | 1.3889 | 24.1667 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 20 | 2 | 6 | 26 | 54.1667 | 49.5370 | 42.2222 | 29.9722 | 34.6036 | -4.6296 | 7.3148 | 14.0093 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 5 | 2 | 6 | 11 | 54.1667 | 49.5370 | 46.4815 | 35.0000 | 38.9933 | -4.6296 | 3.0556 | 12.4444 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 0 | 2 | 6 | 6 | 50.4167 | 47.3465 | 47.3465 | N/A | N/A | -3.0702 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 10 | 2 | 6 | 16 | 50.4167 | 47.3465 | 41.7325 | 34.6579 | 37.0097 | -3.0702 | 5.6140 | 10.2588 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 2 | 2 | 6 | 8 | 50.4167 | 47.3465 | 45.7018 | 31.1184 | 35.1818 | -3.0702 | 1.6447 | 16.4693 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 20 | 2 | 6 | 26 | 50.4167 | 47.3465 | 38.8377 | 30.6382 | 33.6526 | -3.0702 | 8.5088 | 10.5592 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 5 | 2 | 6 | 11 | 50.4167 | 47.3465 | 43.7719 | 32.5789 | 36.3369 | -3.0702 | 3.5746 | 12.4474 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 0 | 2 | 6 | 6 | 53.3333 | 45.8333 | 45.8333 | N/A | N/A | -7.5000 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 10 | 2 | 6 | 16 | 53.3333 | 45.8333 | 39.5833 | 34.5000 | 36.2728 | -7.5000 | 6.2500 | 7.7500 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 2 | 2 | 6 | 8 | 53.3333 | 45.8333 | 43.7500 | 30.0000 | 33.2127 | -7.5000 | 2.0833 | 13.7500 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 20 | 2 | 6 | 26 | 53.3333 | 45.8333 | 37.0833 | 31.2500 | 33.4403 | -7.5000 | 8.7500 | 7.4167 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 5 | 2 | 6 | 11 | 53.3333 | 45.8333 | 40.8333 | 36.0000 | 36.9348 | -7.5000 | 5.0000 | 11.3333 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 0 | 2 | 6 | 6 | 74.1667 | 56.2963 | 56.2963 | N/A | N/A | -17.8704 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 10 | 2 | 6 | 16 | 74.1667 | 56.2963 | 51.4815 | 33.0000 | 39.5843 | -17.8704 | 4.8148 | 19.2963 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 2 | 2 | 6 | 8 | 74.1667 | 56.2963 | 55.3704 | 17.2222 | 23.9437 | -17.8704 | 0.9259 | 38.3333 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 20 | 2 | 6 | 26 | 74.1667 | 56.2963 | 49.8148 | 30.2222 | 36.9316 | -17.8704 | 6.4815 | 21.2407 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 5 | 2 | 6 | 11 | 74.1667 | 56.2963 | 54.7222 | 29.8889 | 37.6324 | -17.8704 | 1.5741 | 25.2407 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 0 | 2 | 6 | 6 | 72.5000 | 57.3904 | 57.3904 | N/A | N/A | -15.1096 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 10 | 2 | 6 | 16 | 72.5000 | 57.3904 | 51.2061 | 30.6974 | 37.5520 | -15.1096 | 6.1842 | 21.8509 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 2 | 2 | 6 | 8 | 72.5000 | 57.3904 | 56.3158 | 15.8553 | 22.6336 | -15.1096 | 1.0746 | 42.0833 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 20 | 2 | 6 | 26 | 72.5000 | 57.3904 | 49.2544 | 29.5395 | 36.3394 | -15.1096 | 8.1360 | 20.7851 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 5 | 2 | 6 | 11 | 72.5000 | 57.3904 | 54.1228 | 30.8684 | 38.0849 | -15.1096 | 3.2675 | 24.4912 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 0 | 2 | 6 | 6 | 71.6667 | 56.6667 | 56.6667 | N/A | N/A | -15.0000 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 10 | 2 | 6 | 16 | 71.6667 | 56.6667 | 52.9167 | 32.5000 | 39.4708 | -15.0000 | 3.7500 | 21.9167 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 2 | 2 | 6 | 8 | 71.6667 | 56.6667 | 56.2500 | 21.2500 | 21.2974 | -15.0000 | 0.4167 | 45.8333 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 20 | 2 | 6 | 26 | 71.6667 | 56.6667 | 51.2500 | 29.3750 | 36.0855 | -15.0000 | 5.4167 | 24.3750 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 5 | 2 | 6 | 11 | 71.6667 | 56.6667 | 54.1667 | 30.0000 | 37.6615 | -15.0000 | 2.5000 | 24.1667 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 0 | 2 | 6 | 6 | 58.3333 | 48.1481 | 48.1481 | N/A | N/A | -10.1852 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 10 | 2 | 6 | 16 | 58.3333 | 48.1481 | 43.7037 | 33.5556 | 37.2939 | -10.1852 | 4.4444 | 12.1852 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 2 | 2 | 6 | 8 | 58.3333 | 48.1481 | 48.2407 | 25.0000 | 29.5447 | -10.1852 | -0.0926 | 24.9074 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 20 | 2 | 6 | 26 | 58.3333 | 48.1481 | 40.7407 | 30.5000 | 34.2072 | -10.1852 | 7.4074 | 13.3148 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 5 | 2 | 6 | 11 | 58.3333 | 48.1481 | 45.8333 | 32.2222 | 37.0726 | -10.1852 | 2.3148 | 13.9444 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 0 | 2 | 6 | 6 | 52.9167 | 47.7632 | 47.7632 | N/A | N/A | -5.1535 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 10 | 2 | 6 | 16 | 52.9167 | 47.7632 | 41.6447 | 34.6711 | 37.0141 | -5.1535 | 6.1184 | 9.9298 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 2 | 2 | 6 | 8 | 52.9167 | 47.7632 | 45.7018 | 31.0526 | 34.6814 | -5.1535 | 2.0614 | 17.5877 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 20 | 2 | 6 | 26 | 52.9167 | 47.7632 | 38.5746 | 30.6513 | 33.6458 | -5.1535 | 9.1886 | 9.9671 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 5 | 2 | 6 | 11 | 52.9167 | 47.7632 | 43.4649 | 33.6053 | 36.7679 | -5.1535 | 4.2982 | 11.7632 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 0 | 2 | 6 | 6 | 55.0000 | 46.6667 | 46.6667 | N/A | N/A | -8.3333 | 0.0000 | N/A |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 10 | 2 | 6 | 16 | 55.0000 | 46.6667 | 42.5000 | 35.5000 | 38.0281 | -8.3333 | 4.1667 | 9.8333 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 2 | 2 | 6 | 8 | 55.0000 | 46.6667 | 47.9167 | 40.0000 | 39.8141 | -8.3333 | -1.2500 | 17.0833 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 20 | 2 | 6 | 26 | 55.0000 | 46.6667 | 38.7500 | 33.6250 | 35.5446 | -8.3333 | 7.9167 | 6.7917 |
| D92-MarginJointLocalRidge-v1 | proxy | R_MARGIN_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 5 | 2 | 6 | 11 | 55.0000 | 46.6667 | 44.5833 | 39.0000 | 39.6762 | -8.3333 | 2.0833 | 12.4167 |

资源口径

唯一supervisor程序墙钟98.438323s，CPU进程峰值RSS为1325740032B；包含配对验证与日志。单条分类头评分计时不代表完整配对耗时。
原分类头及packet文件字节取下表当次pairing.resources的实测值；缺失值为N/A。文件/数组字节不等于传输字节或部署常驻内存；实际星地传输、GPU峰值、能耗未测，为N/A。
| 方法 | 行 | 原始实测资源 |
|---|---|---|
| D92-MarginJointLocalRidge-v1 | margin-rx3-cvs-daot-rc4-s2026092701 | {"packet_load_seconds": 0.0023573630023747683, "raw_cache_read_seconds": 0.12078597799700219, "float32_scalar_list_bridge_seconds": 0.026449442200828344, "native_single_record_score_seconds": 0.13662628672318533, "wall_seconds": 0.2991652069904376, "score_call_count": 1040, "scored_physical_count": 1040, "packet_total_local_file_bytes": 8737, "head_weight_file_bytes": 3840, "raw_z_id_numeric_bytes": 665600, "score_numeric_bytes": 24960, "incremental_transmission_bytes": null, "deployment_peak_memory_bytes": null, "gpu_peak_memory_bytes": null, "energy": null, "unmeasured_reason": "Local packet bytes and CPU timings measured; no network transfer, deployment peak or energy measured", "hardware": {"device": "cpu", "machine": "x86_64", "platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39", "torch_version": "2.1.0+cu121"}, "byte_scope": "Actual local packet/cache/prediction files and numeric arrays; not measured transmission or resident deployment memory"} |
| D92-MarginJointLocalRidge-v1 | margin-rx3-cvs-daot-rc4-s2026092702 | {"packet_load_seconds": 0.0008617660059826449, "raw_cache_read_seconds": 0.05058454000391066, "float32_scalar_list_bridge_seconds": 0.027187164305360056, "native_single_record_score_seconds": 0.1378936692053685, "wall_seconds": 0.2295179720094893, "score_call_count": 1040, "scored_physical_count": 1040, "packet_total_local_file_bytes": 8735, "head_weight_file_bytes": 3840, "raw_z_id_numeric_bytes": 665600, "score_numeric_bytes": 24960, "incremental_transmission_bytes": null, "deployment_peak_memory_bytes": null, "gpu_peak_memory_bytes": null, "energy": null, "unmeasured_reason": "Local packet bytes and CPU timings measured; no network transfer, deployment peak or energy measured", "hardware": {"device": "cpu", "machine": "x86_64", "platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39", "torch_version": "2.1.0+cu121"}, "byte_scope": "Actual local packet/cache/prediction files and numeric arrays; not measured transmission or resident deployment memory"} |
| D92-MarginJointLocalRidge-v1 | margin-rx1-cvs-daot-rc4-s2026092701 | {"packet_load_seconds": 0.0009174730075756088, "raw_cache_read_seconds": 0.017827410993049853, "float32_scalar_list_bridge_seconds": 0.027148160967044532, "native_single_record_score_seconds": 0.1382471171382349, "wall_seconds": 0.1971690740028862, "score_call_count": 1040, "scored_physical_count": 1040, "packet_total_local_file_bytes": 8737, "head_weight_file_bytes": 3840, "raw_z_id_numeric_bytes": 665600, "score_numeric_bytes": 24960, "incremental_transmission_bytes": null, "deployment_peak_memory_bytes": null, "gpu_peak_memory_bytes": null, "energy": null, "unmeasured_reason": "Local packet bytes and CPU timings measured; no network transfer, deployment peak or energy measured", "hardware": {"device": "cpu", "machine": "x86_64", "platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39", "torch_version": "2.1.0+cu121"}, "byte_scope": "Actual local packet/cache/prediction files and numeric arrays; not measured transmission or resident deployment memory"} |
| D92-MarginJointLocalRidge-v1 | margin-rx1-cvs-daot-rc4-s2026092702 | {"packet_load_seconds": 0.0008431629976257682, "raw_cache_read_seconds": 0.018296784997801296, "float32_scalar_list_bridge_seconds": 0.02677119674626738, "native_single_record_score_seconds": 0.13734812024631537, "wall_seconds": 0.19595418899552897, "score_call_count": 1040, "scored_physical_count": 1040, "packet_total_local_file_bytes": 8735, "head_weight_file_bytes": 3840, "raw_z_id_numeric_bytes": 665600, "score_numeric_bytes": 24960, "incremental_transmission_bytes": null, "deployment_peak_memory_bytes": null, "gpu_peak_memory_bytes": null, "energy": null, "unmeasured_reason": "Local packet bytes and CPU timings measured; no network transfer, deployment peak or energy measured", "hardware": {"device": "cpu", "machine": "x86_64", "platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39", "torch_version": "2.1.0+cu121"}, "byte_scope": "Actual local packet/cache/prediction files and numeric arrays; not measured transmission or resident deployment memory"} |

来源：`E:\type10-7\automation_reports\CV-SincNet\20261001-phase2-d92-margin-ground-a-support-m2-r01\results\raw_support_pairing`；实际runtime commit：`d86edc3235ec0ca0e1925223970e07373256fb9c`。
