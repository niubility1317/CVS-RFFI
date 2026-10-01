# 原地面A与联合LocalRidge方法：完整三阶段配对

全部8行已独立读回完成，共320个parent（每种方法160个）。旧类数6，新增类数0、2、5、10、20；K为1、5、10、20。
A使用原地面6类float32分类头；A/B/C旧类准确率来自同一source row、split与物理旧held support。B/C为原方法已固定预测，没有重拟合。
本报告是合法support持出诊断，不是query准确率或新增独立验证。A预测先固定，随后连接truth；结果不回流选模、参数或重跑。
H、注册下降、绝对新旧差先逐parent计算，再等parent平均；不以汇总准确率重新计算H或绝对差。true K1无独立held，记N/A。
理想目标：B−A≥10个百分点、B−C旧≤1个百分点、平均绝对新旧差≤3个百分点；当前为逐步改善方向，未增加硬门槛。

整体（OOF，K5/10/20、新增2/5/10/20，每方法96个可测parent）

| method | measured_parents | 旧类数 | 总注册类数 | A旧 | B旧 | C旧 | C新 | H | B−A | 注册下降 | 绝对差 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D92-AffineJointLocalRidge-v1 | 96 | 6 | 6+new | 62.6389 | 71.1458 | 66.6580 | 52.5260 | 57.9792 | 8.5069 | 4.4878 | 14.7917 |
| D92-ConditionalJointLocalRidge-v1 | 96 | 6 | 6+new | 62.6389 | 70.2083 | 61.3281 | 57.6094 | 58.1416 | 7.5694 | 8.8802 | 12.3247 |

完整K×新增类数

| method | diagnostic | k | new_count | measured_parents | 旧类数 | 总注册类数 | A旧 | B旧 | C旧 | C新 | H | B−A | 注册下降 | 绝对差 |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D92-AffineJointLocalRidge-v1 | oof | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | 10 | 0 | 8 | 6 | 6 | 64.3750 | 73.5417 | 73.5417 | N/A | N/A | 9.1667 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | 10 | 10 | 8 | 6 | 16 | 64.3750 | 73.5417 | 68.9583 | 56.3750 | 61.9638 | 9.1667 | 4.5833 | 12.5833 |
| D92-AffineJointLocalRidge-v1 | oof | 10 | 2 | 8 | 6 | 8 | 64.3750 | 73.5417 | 73.1250 | 50.0000 | 58.3349 | 9.1667 | 0.4167 | 23.1250 |
| D92-AffineJointLocalRidge-v1 | oof | 10 | 20 | 8 | 6 | 26 | 64.3750 | 73.5417 | 67.0833 | 55.1250 | 60.4783 | 9.1667 | 6.4583 | 11.9583 |
| D92-AffineJointLocalRidge-v1 | oof | 10 | 5 | 8 | 6 | 11 | 64.3750 | 73.5417 | 70.2083 | 54.5000 | 60.9349 | 9.1667 | 3.3333 | 15.7083 |
| D92-AffineJointLocalRidge-v1 | oof | 20 | 0 | 8 | 6 | 6 | 61.4583 | 76.5625 | 76.5625 | N/A | N/A | 15.1042 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | 20 | 10 | 8 | 6 | 16 | 61.4583 | 76.5625 | 71.2500 | 62.3750 | 66.3243 | 15.1042 | 5.3125 | 8.8750 |
| D92-AffineJointLocalRidge-v1 | oof | 20 | 2 | 8 | 6 | 8 | 61.4583 | 76.5625 | 75.9375 | 50.3125 | 59.4987 | 15.1042 | 0.6250 | 25.6250 |
| D92-AffineJointLocalRidge-v1 | oof | 20 | 20 | 8 | 6 | 26 | 61.4583 | 76.5625 | 69.7917 | 62.0000 | 65.5571 | 15.1042 | 6.7708 | 7.7917 |
| D92-AffineJointLocalRidge-v1 | oof | 20 | 5 | 8 | 6 | 11 | 61.4583 | 76.5625 | 72.2917 | 59.7500 | 64.8222 | 15.1042 | 4.2708 | 12.5417 |
| D92-AffineJointLocalRidge-v1 | oof | 5 | 0 | 8 | 6 | 6 | 62.0833 | 63.3333 | 63.3333 | N/A | N/A | 1.2500 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | 5 | 10 | 8 | 6 | 16 | 62.0833 | 63.3333 | 57.5000 | 46.7500 | 51.1009 | 1.2500 | 5.8333 | 10.7500 |
| D92-AffineJointLocalRidge-v1 | oof | 5 | 2 | 8 | 6 | 8 | 62.0833 | 63.3333 | 61.6667 | 43.7500 | 48.8603 | 1.2500 | 1.6667 | 21.2500 |
| D92-AffineJointLocalRidge-v1 | oof | 5 | 20 | 8 | 6 | 26 | 62.0833 | 63.3333 | 54.1667 | 44.8750 | 48.3407 | 1.2500 | 9.1667 | 13.3750 |
| D92-AffineJointLocalRidge-v1 | oof | 5 | 5 | 8 | 6 | 11 | 62.0833 | 63.3333 | 57.9167 | 44.5000 | 49.5340 | 1.2500 | 5.4167 | 13.9167 |
| D92-AffineJointLocalRidge-v1 | proxy | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | 10 | 0 | 8 | 6 | 6 | 64.3750 | 53.0324 | 53.0324 | N/A | N/A | -11.3426 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | 10 | 10 | 8 | 6 | 16 | 64.3750 | 53.0324 | 47.9167 | 33.0556 | 38.3731 | -11.3426 | 5.1157 | 16.3889 |
| D92-AffineJointLocalRidge-v1 | proxy | 10 | 2 | 8 | 6 | 8 | 64.3750 | 53.0324 | 52.1296 | 20.9722 | 26.1998 | -11.3426 | 0.9028 | 32.6389 |
| D92-AffineJointLocalRidge-v1 | proxy | 10 | 20 | 8 | 6 | 26 | 64.3750 | 53.0324 | 45.7176 | 30.0903 | 35.5429 | -11.3426 | 7.3148 | 17.7384 |
| D92-AffineJointLocalRidge-v1 | proxy | 10 | 5 | 8 | 6 | 11 | 64.3750 | 53.0324 | 50.5324 | 31.4444 | 37.5892 | -11.3426 | 2.5000 | 19.7546 |
| D92-AffineJointLocalRidge-v1 | proxy | 20 | 0 | 8 | 6 | 6 | 61.4583 | 52.4726 | 52.4726 | N/A | N/A | -8.9857 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | 20 | 10 | 8 | 6 | 16 | 61.4583 | 52.4726 | 45.9539 | 32.4770 | 36.9929 | -8.9857 | 6.5186 | 15.6502 |
| D92-AffineJointLocalRidge-v1 | proxy | 20 | 2 | 8 | 6 | 8 | 61.4583 | 52.4726 | 50.8882 | 23.3224 | 28.5396 | -8.9857 | 1.5844 | 29.5833 |
| D92-AffineJointLocalRidge-v1 | proxy | 20 | 20 | 8 | 6 | 26 | 61.4583 | 52.4726 | 43.4978 | 30.2039 | 34.9514 | -8.9857 | 8.9748 | 14.9200 |
| D92-AffineJointLocalRidge-v1 | proxy | 20 | 5 | 8 | 6 | 11 | 61.4583 | 52.4726 | 48.4814 | 31.4803 | 36.7860 | -8.9857 | 3.9912 | 18.3673 |
| D92-AffineJointLocalRidge-v1 | proxy | 5 | 0 | 8 | 6 | 6 | 62.0833 | 50.9375 | 50.9375 | N/A | N/A | -11.1458 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | 5 | 10 | 8 | 6 | 16 | 62.0833 | 50.9375 | 46.2500 | 33.8125 | 38.1120 | -11.1458 | 4.6875 | 14.4792 |
| D92-AffineJointLocalRidge-v1 | proxy | 5 | 2 | 8 | 6 | 8 | 62.0833 | 50.9375 | 50.5208 | 27.5000 | 28.7583 | -11.1458 | 0.4167 | 30.3125 |
| D92-AffineJointLocalRidge-v1 | proxy | 5 | 20 | 8 | 6 | 26 | 62.0833 | 50.9375 | 44.1667 | 30.9375 | 35.2303 | -11.1458 | 6.7708 | 15.1250 |
| D92-AffineJointLocalRidge-v1 | proxy | 5 | 5 | 8 | 6 | 11 | 62.0833 | 50.9375 | 48.0208 | 33.8750 | 37.9984 | -11.1458 | 2.9167 | 17.5625 |
| D92-ConditionalJointLocalRidge-v1 | oof | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | 10 | 0 | 8 | 6 | 6 | 64.3750 | 72.0833 | 72.0833 | N/A | N/A | 7.7083 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | 10 | 10 | 8 | 6 | 16 | 64.3750 | 72.0833 | 63.1250 | 58.7500 | 60.6423 | 7.7083 | 8.9583 | 7.7917 |
| D92-ConditionalJointLocalRidge-v1 | oof | 10 | 2 | 8 | 6 | 8 | 64.3750 | 72.0833 | 67.2917 | 60.0000 | 61.6891 | 7.7083 | 4.7917 | 20.6250 |
| D92-ConditionalJointLocalRidge-v1 | oof | 10 | 20 | 8 | 6 | 26 | 64.3750 | 72.0833 | 61.6667 | 56.0000 | 58.3895 | 7.7083 | 10.4167 | 9.2083 |
| D92-ConditionalJointLocalRidge-v1 | oof | 10 | 5 | 8 | 6 | 11 | 64.3750 | 72.0833 | 64.3750 | 58.2500 | 60.8186 | 7.7083 | 7.7083 | 8.5417 |
| D92-ConditionalJointLocalRidge-v1 | oof | 20 | 0 | 8 | 6 | 6 | 61.4583 | 76.4583 | 76.4583 | N/A | N/A | 15.0000 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | 20 | 10 | 8 | 6 | 16 | 61.4583 | 76.4583 | 67.8125 | 61.4375 | 64.4026 | 15.0000 | 8.6458 | 6.5417 |
| D92-ConditionalJointLocalRidge-v1 | oof | 20 | 2 | 8 | 6 | 8 | 61.4583 | 76.4583 | 71.7708 | 60.9375 | 65.7610 | 15.0000 | 4.6875 | 11.0417 |
| D92-ConditionalJointLocalRidge-v1 | oof | 20 | 20 | 8 | 6 | 26 | 61.4583 | 76.4583 | 66.4583 | 61.4375 | 63.8230 | 15.0000 | 10.0000 | 5.2083 |
| D92-ConditionalJointLocalRidge-v1 | oof | 20 | 5 | 8 | 6 | 11 | 61.4583 | 76.4583 | 68.8542 | 60.1250 | 64.0865 | 15.0000 | 7.6042 | 8.7292 |
| D92-ConditionalJointLocalRidge-v1 | oof | 5 | 0 | 8 | 6 | 6 | 62.0833 | 62.0833 | 62.0833 | N/A | N/A | 0.0000 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | 5 | 10 | 8 | 6 | 16 | 62.0833 | 62.0833 | 49.5833 | 50.2500 | 48.6904 | 0.0000 | 12.5000 | 11.1667 |
| D92-ConditionalJointLocalRidge-v1 | oof | 5 | 2 | 8 | 6 | 8 | 62.0833 | 62.0833 | 54.5833 | 65.0000 | 51.9057 | 0.0000 | 7.5000 | 35.4167 |
| D92-ConditionalJointLocalRidge-v1 | oof | 5 | 20 | 8 | 6 | 26 | 62.0833 | 62.0833 | 47.0833 | 47.6250 | 45.6812 | 0.0000 | 15.0000 | 13.6250 |
| D92-ConditionalJointLocalRidge-v1 | oof | 5 | 5 | 8 | 6 | 11 | 62.0833 | 62.0833 | 53.3333 | 51.5000 | 51.8093 | 0.0000 | 8.7500 | 10.0000 |
| D92-ConditionalJointLocalRidge-v1 | proxy | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | 10 | 0 | 8 | 6 | 6 | 64.3750 | 53.0324 | 53.0324 | N/A | N/A | -11.3426 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | 10 | 10 | 8 | 6 | 16 | 64.3750 | 53.0324 | 38.7269 | 38.1528 | 36.2993 | -11.3426 | 14.3056 | 12.5926 |
| D92-ConditionalJointLocalRidge-v1 | proxy | 10 | 2 | 8 | 6 | 8 | 64.3750 | 53.0324 | 44.0741 | 40.7639 | 33.7795 | -11.3426 | 8.9583 | 31.5972 |
| D92-ConditionalJointLocalRidge-v1 | proxy | 10 | 20 | 8 | 6 | 26 | 64.3750 | 53.0324 | 36.5509 | 33.0972 | 32.6556 | -11.3426 | 16.4815 | 13.5231 |
| D92-ConditionalJointLocalRidge-v1 | proxy | 10 | 5 | 8 | 6 | 11 | 64.3750 | 53.0324 | 41.1111 | 39.6111 | 37.5271 | -11.3426 | 11.9213 | 16.4259 |
| D92-ConditionalJointLocalRidge-v1 | proxy | 20 | 0 | 8 | 6 | 6 | 61.4583 | 52.4726 | 52.4726 | N/A | N/A | -8.9857 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | 20 | 10 | 8 | 6 | 16 | 61.4583 | 52.4726 | 37.0779 | 37.3717 | 35.2232 | -8.9857 | 15.3947 | 12.4803 |
| D92-ConditionalJointLocalRidge-v1 | proxy | 20 | 2 | 8 | 6 | 8 | 61.4583 | 52.4726 | 43.0976 | 44.5395 | 36.9171 | -8.9857 | 9.3750 | 28.1634 |
| D92-ConditionalJointLocalRidge-v1 | proxy | 20 | 20 | 8 | 6 | 26 | 61.4583 | 52.4726 | 35.1425 | 32.8026 | 32.4152 | -8.9857 | 17.3300 | 11.1820 |
| D92-ConditionalJointLocalRidge-v1 | proxy | 20 | 5 | 8 | 6 | 11 | 61.4583 | 52.4726 | 39.5943 | 39.3947 | 36.8457 | -8.9857 | 12.8783 | 15.0702 |
| D92-ConditionalJointLocalRidge-v1 | proxy | 5 | 0 | 8 | 6 | 6 | 62.0833 | 50.9375 | 50.9375 | N/A | N/A | -11.1458 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | 5 | 10 | 8 | 6 | 16 | 62.0833 | 50.9375 | 38.4375 | 38.8750 | 36.9418 | -11.1458 | 12.5000 | 11.8125 |
| D92-ConditionalJointLocalRidge-v1 | proxy | 5 | 2 | 8 | 6 | 8 | 62.0833 | 50.9375 | 43.6458 | 47.8125 | 36.1417 | -11.1458 | 7.2917 | 35.4167 |
| D92-ConditionalJointLocalRidge-v1 | proxy | 5 | 20 | 8 | 6 | 26 | 62.0833 | 50.9375 | 34.3750 | 33.5000 | 31.8192 | -11.1458 | 16.5625 | 13.8333 |
| D92-ConditionalJointLocalRidge-v1 | proxy | 5 | 5 | 8 | 6 | 11 | 62.0833 | 50.9375 | 40.0000 | 43.2500 | 38.6680 | -11.1458 | 10.9375 | 17.7500 |

by_receiver_scene

| method | diagnostic | path | receiver | scenario | k | new_count | measured_parents | 旧类数 | 总注册类数 | A旧 | B旧 | C旧 | C新 | H | B−A | 注册下降 | 绝对差 |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 10 | 0 | 2 | 6 | 6 | 73.3333 | 97.5000 | 97.5000 | N/A | N/A | 24.1667 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 10 | 10 | 2 | 6 | 16 | 73.3333 | 97.5000 | 90.8333 | 77.0000 | 83.3407 | 24.1667 | 6.6667 | 13.8333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 10 | 2 | 2 | 6 | 8 | 73.3333 | 97.5000 | 97.5000 | 77.5000 | 86.3452 | 24.1667 | 0.0000 | 20.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 10 | 20 | 2 | 6 | 26 | 73.3333 | 97.5000 | 89.1667 | 76.7500 | 82.4793 | 24.1667 | 8.3333 | 12.4167 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 10 | 5 | 2 | 6 | 11 | 73.3333 | 97.5000 | 91.6667 | 87.0000 | 89.2459 | 24.1667 | 5.8333 | 4.6667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 20 | 0 | 2 | 6 | 6 | 64.1667 | 95.4167 | 95.4167 | N/A | N/A | 31.2500 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 20 | 10 | 2 | 6 | 16 | 64.1667 | 95.4167 | 92.5000 | 87.0000 | 89.6657 | 31.2500 | 2.9167 | 5.5000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 20 | 2 | 2 | 6 | 8 | 64.1667 | 95.4167 | 95.4167 | 81.2500 | 87.7568 | 31.2500 | 0.0000 | 14.1667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 20 | 20 | 2 | 6 | 26 | 64.1667 | 95.4167 | 90.8333 | 86.0000 | 88.3471 | 31.2500 | 4.5833 | 4.8333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 20 | 5 | 2 | 6 | 11 | 64.1667 | 95.4167 | 92.5000 | 89.0000 | 90.7070 | 31.2500 | 2.9167 | 3.5000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 5 | 0 | 2 | 6 | 6 | 70.0000 | 91.6667 | 91.6667 | N/A | N/A | 21.6667 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 5 | 10 | 2 | 6 | 16 | 70.0000 | 91.6667 | 85.0000 | 81.0000 | 82.9455 | 21.6667 | 6.6667 | 4.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 5 | 2 | 2 | 6 | 8 | 70.0000 | 91.6667 | 90.0000 | 80.0000 | 84.3750 | 21.6667 | 1.6667 | 10.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 5 | 20 | 2 | 6 | 26 | 70.0000 | 91.6667 | 81.6667 | 73.0000 | 77.0460 | 21.6667 | 10.0000 | 8.6667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_high | 5 | 5 | 2 | 6 | 11 | 70.0000 | 91.6667 | 85.0000 | 86.0000 | 85.4968 | 21.6667 | 6.6667 | 1.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 10 | 0 | 2 | 6 | 6 | 39.1667 | 50.0000 | 50.0000 | N/A | N/A | 10.8333 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 10 | 10 | 2 | 6 | 16 | 39.1667 | 50.0000 | 41.6667 | 32.0000 | 36.1057 | 10.8333 | 8.3333 | 9.6667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 10 | 2 | 2 | 6 | 8 | 39.1667 | 50.0000 | 48.3333 | 47.5000 | 47.9098 | 10.8333 | 1.6667 | 0.8333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 10 | 20 | 2 | 6 | 26 | 39.1667 | 50.0000 | 40.8333 | 31.7500 | 35.6919 | 10.8333 | 9.1667 | 9.0833 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 10 | 5 | 2 | 6 | 11 | 39.1667 | 50.0000 | 45.8333 | 32.0000 | 37.6843 | 10.8333 | 4.1667 | 13.8333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 20 | 0 | 2 | 6 | 6 | 39.1667 | 56.6667 | 56.6667 | N/A | N/A | 17.5000 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 20 | 10 | 2 | 6 | 16 | 39.1667 | 56.6667 | 49.1667 | 35.7500 | 41.3742 | 17.5000 | 7.5000 | 13.4167 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 20 | 2 | 2 | 6 | 8 | 39.1667 | 56.6667 | 55.4167 | 25.0000 | 34.3451 | 17.5000 | 1.2500 | 30.4167 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 20 | 20 | 2 | 6 | 26 | 39.1667 | 56.6667 | 46.6667 | 34.7500 | 39.8351 | 17.5000 | 10.0000 | 11.9167 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 20 | 5 | 2 | 6 | 11 | 39.1667 | 56.6667 | 50.8333 | 30.0000 | 37.6438 | 17.5000 | 5.8333 | 20.8333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 5 | 0 | 2 | 6 | 6 | 38.3333 | 33.3333 | 33.3333 | N/A | N/A | -5.0000 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 5 | 10 | 2 | 6 | 16 | 38.3333 | 33.3333 | 30.0000 | 28.0000 | 28.9655 | -5.0000 | 3.3333 | 2.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 5 | 2 | 2 | 6 | 8 | 38.3333 | 33.3333 | 30.0000 | 20.0000 | 24.0000 | -5.0000 | 3.3333 | 10.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 5 | 20 | 2 | 6 | 26 | 38.3333 | 33.3333 | 28.3333 | 19.5000 | 22.9059 | -5.0000 | 5.0000 | 8.8333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 19-1 | practical_low_urban | 5 | 5 | 2 | 6 | 11 | 38.3333 | 33.3333 | 28.3333 | 16.0000 | 20.4348 | -5.0000 | 5.0000 | 12.3333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 10 | 0 | 2 | 6 | 6 | 91.6667 | 98.3333 | 98.3333 | N/A | N/A | 6.6667 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 10 | 10 | 2 | 6 | 16 | 91.6667 | 98.3333 | 97.5000 | 82.5000 | 89.3703 | 6.6667 | 0.8333 | 15.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 10 | 2 | 2 | 6 | 8 | 91.6667 | 98.3333 | 98.3333 | 52.5000 | 68.4178 | 6.6667 | 0.0000 | 45.8333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 10 | 20 | 2 | 6 | 26 | 91.6667 | 98.3333 | 95.0000 | 76.0000 | 84.4380 | 6.6667 | 3.3333 | 19.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 10 | 5 | 2 | 6 | 11 | 91.6667 | 98.3333 | 97.5000 | 72.0000 | 82.8304 | 6.6667 | 0.8333 | 25.5000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 20 | 0 | 2 | 6 | 6 | 87.9167 | 96.6667 | 96.6667 | N/A | N/A | 8.7500 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 20 | 10 | 2 | 6 | 16 | 87.9167 | 96.6667 | 94.1667 | 88.5000 | 91.2362 | 8.7500 | 2.5000 | 5.6667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 20 | 2 | 2 | 6 | 8 | 87.9167 | 96.6667 | 96.2500 | 68.7500 | 80.0313 | 8.7500 | 0.4167 | 27.5000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 20 | 20 | 2 | 6 | 26 | 87.9167 | 96.6667 | 95.0000 | 86.5000 | 90.5416 | 8.7500 | 1.6667 | 8.5000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 20 | 5 | 2 | 6 | 11 | 87.9167 | 96.6667 | 93.7500 | 85.5000 | 89.4328 | 8.7500 | 2.9167 | 8.2500 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 5 | 0 | 2 | 6 | 6 | 91.6667 | 93.3333 | 93.3333 | N/A | N/A | 1.6667 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 5 | 10 | 2 | 6 | 16 | 91.6667 | 93.3333 | 88.3333 | 64.0000 | 74.2168 | 1.6667 | 5.0000 | 24.3333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 5 | 2 | 2 | 6 | 8 | 91.6667 | 93.3333 | 93.3333 | 35.0000 | 50.7027 | 1.6667 | 0.0000 | 58.3333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 5 | 20 | 2 | 6 | 26 | 91.6667 | 93.3333 | 88.3333 | 60.5000 | 71.8014 | 1.6667 | 5.0000 | 27.8333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_high | 5 | 5 | 2 | 6 | 11 | 91.6667 | 93.3333 | 88.3333 | 60.0000 | 71.3347 | 1.6667 | 5.0000 | 28.3333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 10 | 0 | 2 | 6 | 6 | 53.3333 | 48.3333 | 48.3333 | N/A | N/A | -5.0000 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 10 | 10 | 2 | 6 | 16 | 53.3333 | 48.3333 | 45.8333 | 34.0000 | 39.0385 | -5.0000 | 2.5000 | 11.8333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 10 | 2 | 2 | 6 | 8 | 53.3333 | 48.3333 | 48.3333 | 22.5000 | 30.6667 | -5.0000 | 0.0000 | 25.8333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 10 | 20 | 2 | 6 | 26 | 53.3333 | 48.3333 | 43.3333 | 36.0000 | 39.3041 | -5.0000 | 5.0000 | 7.3333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 10 | 5 | 2 | 6 | 11 | 53.3333 | 48.3333 | 45.8333 | 27.0000 | 33.9789 | -5.0000 | 2.5000 | 18.8333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 20 | 0 | 2 | 6 | 6 | 54.5833 | 57.5000 | 57.5000 | N/A | N/A | 2.9167 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 20 | 10 | 2 | 6 | 16 | 54.5833 | 57.5000 | 49.1667 | 38.2500 | 43.0209 | 2.9167 | 8.3333 | 10.9167 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 20 | 2 | 2 | 6 | 8 | 54.5833 | 57.5000 | 56.6667 | 26.2500 | 35.8618 | 2.9167 | 0.8333 | 30.4167 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 20 | 20 | 2 | 6 | 26 | 54.5833 | 57.5000 | 46.6667 | 40.7500 | 43.5047 | 2.9167 | 10.8333 | 5.9167 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 20 | 5 | 2 | 6 | 11 | 54.5833 | 57.5000 | 52.0833 | 34.5000 | 41.5054 | 2.9167 | 5.4167 | 17.5833 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 5 | 0 | 2 | 6 | 6 | 48.3333 | 35.0000 | 35.0000 | N/A | N/A | -13.3333 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 5 | 10 | 2 | 6 | 16 | 48.3333 | 35.0000 | 26.6667 | 14.0000 | 18.2759 | -13.3333 | 8.3333 | 12.6667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 5 | 2 | 2 | 6 | 8 | 48.3333 | 35.0000 | 33.3333 | 40.0000 | 36.3636 | -13.3333 | 1.6667 | 6.6667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 5 | 20 | 2 | 6 | 26 | 48.3333 | 35.0000 | 18.3333 | 26.5000 | 21.6097 | -13.3333 | 16.6667 | 8.1667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 20-19 | practical_low_urban | 5 | 5 | 2 | 6 | 11 | 48.3333 | 35.0000 | 30.0000 | 16.0000 | 20.8696 | -13.3333 | 5.0000 | 14.0000 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 10 | 0 | 2 | 6 | 6 | 73.3333 | 72.4074 | 72.4074 | N/A | N/A | -0.9259 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 10 | 10 | 2 | 6 | 16 | 73.3333 | 72.4074 | 69.0741 | 51.7778 | 58.7612 | -0.9259 | 3.3333 | 18.2593 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 10 | 2 | 2 | 6 | 8 | 73.3333 | 72.4074 | 73.1481 | 35.2778 | 44.0618 | -0.9259 | -0.7407 | 37.8704 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 10 | 20 | 2 | 6 | 26 | 73.3333 | 72.4074 | 65.9259 | 46.2778 | 53.7552 | -0.9259 | 6.4815 | 21.9815 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 10 | 5 | 2 | 6 | 11 | 73.3333 | 72.4074 | 70.3704 | 53.8889 | 60.4704 | -0.9259 | 2.0370 | 16.5926 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 20 | 0 | 2 | 6 | 6 | 64.1667 | 69.0132 | 69.0132 | N/A | N/A | 4.8465 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 20 | 10 | 2 | 6 | 16 | 64.1667 | 69.0132 | 64.7588 | 54.4868 | 58.7615 | 4.8465 | 4.2544 | 12.0526 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 20 | 2 | 2 | 6 | 8 | 64.1667 | 69.0132 | 68.6842 | 47.6974 | 54.6792 | 4.8465 | 0.3289 | 21.5132 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 20 | 20 | 2 | 6 | 26 | 64.1667 | 69.0132 | 61.2061 | 48.3289 | 53.6791 | 4.8465 | 7.8070 | 13.8509 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 20 | 5 | 2 | 6 | 11 | 64.1667 | 69.0132 | 66.3158 | 52.0789 | 57.8205 | 4.8465 | 2.6974 | 14.8947 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 5 | 0 | 2 | 6 | 6 | 70.0000 | 66.6667 | 66.6667 | N/A | N/A | -3.3333 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 5 | 10 | 2 | 6 | 16 | 70.0000 | 66.6667 | 64.5833 | 55.5000 | 59.2539 | -3.3333 | 2.0833 | 10.7500 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 5 | 2 | 2 | 6 | 8 | 70.0000 | 66.6667 | 67.9167 | 62.5000 | 63.7977 | -3.3333 | -1.2500 | 14.5833 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 5 | 20 | 2 | 6 | 26 | 70.0000 | 66.6667 | 60.0000 | 51.2500 | 55.1119 | -3.3333 | 6.6667 | 8.7500 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_high | 5 | 5 | 2 | 6 | 11 | 70.0000 | 66.6667 | 65.0000 | 66.5000 | 64.9121 | -3.3333 | 1.6667 | 11.8333 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 10 | 0 | 2 | 6 | 6 | 39.1667 | 25.2778 | 25.2778 | N/A | N/A | -13.8889 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 10 | 10 | 2 | 6 | 16 | 39.1667 | 25.2778 | 19.2593 | 15.3889 | 16.1812 | -13.8889 | 6.0185 | 7.2407 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 10 | 2 | 2 | 6 | 8 | 39.1667 | 25.2778 | 23.2407 | 16.6667 | 16.8793 | -13.8889 | 2.0370 | 11.2037 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 10 | 20 | 2 | 6 | 26 | 39.1667 | 25.2778 | 17.0370 | 14.2778 | 15.1175 | -13.8889 | 8.2407 | 5.2593 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 10 | 5 | 2 | 6 | 11 | 39.1667 | 25.2778 | 21.9444 | 13.3333 | 15.5954 | -13.8889 | 3.3333 | 9.7963 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 20 | 0 | 2 | 6 | 6 | 39.1667 | 26.0965 | 26.0965 | N/A | N/A | -13.0702 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 20 | 10 | 2 | 6 | 16 | 39.1667 | 26.0965 | 18.6184 | 14.8947 | 15.3034 | -13.0702 | 7.4781 | 8.0833 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 20 | 2 | 2 | 6 | 8 | 39.1667 | 26.0965 | 22.7193 | 14.5395 | 15.2297 | -13.0702 | 3.3772 | 12.4781 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 20 | 20 | 2 | 6 | 26 | 39.1667 | 26.0965 | 16.0746 | 13.0329 | 13.6344 | -13.0702 | 10.0219 | 6.4715 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 20 | 5 | 2 | 6 | 11 | 39.1667 | 26.0965 | 20.9211 | 14.2368 | 15.3873 | -13.0702 | 5.1754 | 9.1842 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 5 | 0 | 2 | 6 | 6 | 38.3333 | 25.8333 | 25.8333 | N/A | N/A | -12.5000 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 5 | 10 | 2 | 6 | 16 | 38.3333 | 25.8333 | 17.5000 | 14.5000 | 15.0470 | -12.5000 | 8.3333 | 6.8333 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 5 | 2 | 2 | 6 | 8 | 38.3333 | 25.8333 | 23.7500 | 7.5000 | 9.2292 | -12.5000 | 2.0833 | 16.2500 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 5 | 20 | 2 | 6 | 26 | 38.3333 | 25.8333 | 15.8333 | 13.6250 | 13.8731 | -12.5000 | 10.0000 | 5.4583 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 19-1 | practical_low_urban | 5 | 5 | 2 | 6 | 11 | 38.3333 | 25.8333 | 20.4167 | 8.5000 | 11.6990 | -12.5000 | 5.4167 | 11.9167 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 10 | 0 | 2 | 6 | 6 | 91.6667 | 86.6667 | 86.6667 | N/A | N/A | -5.0000 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 10 | 10 | 2 | 6 | 16 | 91.6667 | 86.6667 | 83.0556 | 50.9444 | 62.9934 | -5.0000 | 3.6111 | 32.1111 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 10 | 2 | 2 | 6 | 8 | 91.6667 | 86.6667 | 87.0370 | 17.7778 | 27.7050 | -5.0000 | -0.3704 | 69.2593 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 10 | 20 | 2 | 6 | 26 | 91.6667 | 86.6667 | 81.2037 | 44.5278 | 57.3744 | -5.0000 | 5.4630 | 36.6759 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 10 | 5 | 2 | 6 | 11 | 91.6667 | 86.6667 | 85.4630 | 46.0000 | 59.5302 | -5.0000 | 1.2037 | 39.4630 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 20 | 0 | 2 | 6 | 6 | 87.9167 | 84.9342 | 84.9342 | N/A | N/A | -2.9825 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 20 | 10 | 2 | 6 | 16 | 87.9167 | 84.9342 | 78.1579 | 45.9211 | 57.4385 | -2.9825 | 6.7763 | 32.2368 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 20 | 2 | 2 | 6 | 8 | 87.9167 | 84.9342 | 84.5395 | 18.5526 | 29.5798 | -2.9825 | 0.3947 | 65.9868 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 20 | 20 | 2 | 6 | 26 | 87.9167 | 84.9342 | 76.5351 | 44.8553 | 56.2260 | -2.9825 | 8.3991 | 31.6798 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 20 | 5 | 2 | 6 | 11 | 87.9167 | 84.9342 | 81.0746 | 45.6316 | 57.8867 | -2.9825 | 3.8596 | 35.4430 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 5 | 0 | 2 | 6 | 6 | 91.6667 | 90.8333 | 90.8333 | N/A | N/A | -0.8333 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 5 | 10 | 2 | 6 | 16 | 91.6667 | 90.8333 | 86.6667 | 52.7500 | 64.8308 | -0.8333 | 4.1667 | 34.0833 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 5 | 2 | 2 | 6 | 8 | 91.6667 | 90.8333 | 90.4167 | 13.7500 | 21.7112 | -0.8333 | 0.4167 | 76.6667 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 5 | 20 | 2 | 6 | 26 | 91.6667 | 90.8333 | 83.7500 | 42.3750 | 55.6650 | -0.8333 | 7.0833 | 41.3750 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_high | 5 | 5 | 2 | 6 | 11 | 91.6667 | 90.8333 | 86.6667 | 48.5000 | 61.6269 | -0.8333 | 4.1667 | 38.1667 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 10 | 0 | 2 | 6 | 6 | 53.3333 | 27.7778 | 27.7778 | N/A | N/A | -25.5556 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 10 | 10 | 2 | 6 | 16 | 53.3333 | 27.7778 | 20.2778 | 14.1111 | 15.5564 | -25.5556 | 7.5000 | 7.9444 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 10 | 2 | 2 | 6 | 8 | 53.3333 | 27.7778 | 25.0926 | 14.1667 | 16.1532 | -25.5556 | 2.6852 | 12.2222 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 10 | 20 | 2 | 6 | 26 | 53.3333 | 27.7778 | 18.7037 | 15.2778 | 15.9245 | -25.5556 | 9.0741 | 7.0370 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 10 | 5 | 2 | 6 | 11 | 53.3333 | 27.7778 | 24.3519 | 12.5556 | 14.7606 | -25.5556 | 3.4259 | 13.1667 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 20 | 0 | 2 | 6 | 6 | 54.5833 | 29.8465 | 29.8465 | N/A | N/A | -24.7368 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 20 | 10 | 2 | 6 | 16 | 54.5833 | 29.8465 | 22.2807 | 14.6053 | 16.4681 | -24.7368 | 7.5658 | 10.2281 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 20 | 2 | 2 | 6 | 8 | 54.5833 | 29.8465 | 27.6096 | 12.5000 | 14.6697 | -24.7368 | 2.2368 | 18.3553 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 20 | 20 | 2 | 6 | 26 | 54.5833 | 29.8465 | 20.1754 | 14.5987 | 16.2660 | -24.7368 | 9.6711 | 7.6776 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 20 | 5 | 2 | 6 | 11 | 54.5833 | 29.8465 | 25.6140 | 13.9737 | 16.0493 | -24.7368 | 4.2325 | 13.9474 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 5 | 0 | 2 | 6 | 6 | 48.3333 | 20.4167 | 20.4167 | N/A | N/A | -27.9167 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 5 | 10 | 2 | 6 | 16 | 48.3333 | 20.4167 | 16.2500 | 12.5000 | 13.3164 | -27.9167 | 4.1667 | 6.2500 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 5 | 2 | 2 | 6 | 8 | 48.3333 | 20.4167 | 20.0000 | 26.2500 | 20.2951 | -27.9167 | 0.4167 | 13.7500 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 5 | 20 | 2 | 6 | 26 | 48.3333 | 20.4167 | 17.0833 | 16.5000 | 16.2712 | -27.9167 | 3.3333 | 4.9167 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 20-19 | practical_low_urban | 5 | 5 | 2 | 6 | 11 | 48.3333 | 20.4167 | 20.0000 | 12.0000 | 13.7556 | -27.9167 | 0.4167 | 8.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 10 | 0 | 2 | 6 | 6 | 73.3333 | 96.6667 | 96.6667 | N/A | N/A | 23.3333 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 10 | 10 | 2 | 6 | 16 | 73.3333 | 96.6667 | 87.5000 | 79.0000 | 83.0079 | 23.3333 | 9.1667 | 8.5000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 10 | 2 | 2 | 6 | 8 | 73.3333 | 96.6667 | 95.0000 | 80.0000 | 86.8505 | 23.3333 | 1.6667 | 15.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 10 | 20 | 2 | 6 | 26 | 73.3333 | 96.6667 | 86.6667 | 77.2500 | 81.6399 | 23.3333 | 10.0000 | 9.4167 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 10 | 5 | 2 | 6 | 11 | 73.3333 | 96.6667 | 89.1667 | 87.0000 | 88.0699 | 23.3333 | 7.5000 | 2.1667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 20 | 0 | 2 | 6 | 6 | 64.1667 | 95.4167 | 95.4167 | N/A | N/A | 31.2500 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 20 | 10 | 2 | 6 | 16 | 64.1667 | 95.4167 | 90.8333 | 85.5000 | 88.0810 | 31.2500 | 4.5833 | 5.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 20 | 2 | 2 | 6 | 8 | 64.1667 | 95.4167 | 94.5833 | 82.5000 | 88.1290 | 31.2500 | 0.8333 | 12.0833 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 20 | 20 | 2 | 6 | 26 | 64.1667 | 95.4167 | 90.4167 | 84.7500 | 87.4903 | 31.2500 | 5.0000 | 5.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 20 | 5 | 2 | 6 | 11 | 64.1667 | 95.4167 | 91.2500 | 87.0000 | 89.0444 | 31.2500 | 4.1667 | 4.2500 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 5 | 0 | 2 | 6 | 6 | 70.0000 | 88.3333 | 88.3333 | N/A | N/A | 18.3333 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 5 | 10 | 2 | 6 | 16 | 70.0000 | 88.3333 | 78.3333 | 80.0000 | 79.1155 | 18.3333 | 10.0000 | 3.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 5 | 2 | 2 | 6 | 8 | 70.0000 | 88.3333 | 83.3333 | 75.0000 | 78.7234 | 18.3333 | 5.0000 | 8.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 5 | 20 | 2 | 6 | 26 | 70.0000 | 88.3333 | 73.3333 | 74.0000 | 73.5810 | 18.3333 | 15.0000 | 5.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_high | 5 | 5 | 2 | 6 | 11 | 70.0000 | 88.3333 | 80.0000 | 84.0000 | 81.7880 | 18.3333 | 8.3333 | 7.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 10 | 0 | 2 | 6 | 6 | 39.1667 | 47.5000 | 47.5000 | N/A | N/A | 8.3333 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 10 | 10 | 2 | 6 | 16 | 39.1667 | 47.5000 | 31.6667 | 37.0000 | 34.1200 | 8.3333 | 15.8333 | 5.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 10 | 2 | 2 | 6 | 8 | 39.1667 | 47.5000 | 37.5000 | 57.5000 | 45.3479 | 8.3333 | 10.0000 | 20.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 10 | 20 | 2 | 6 | 26 | 39.1667 | 47.5000 | 29.1667 | 33.2500 | 31.0741 | 8.3333 | 18.3333 | 4.0833 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 10 | 5 | 2 | 6 | 11 | 39.1667 | 47.5000 | 34.1667 | 39.0000 | 36.4010 | 8.3333 | 13.3333 | 4.8333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 20 | 0 | 2 | 6 | 6 | 39.1667 | 55.8333 | 55.8333 | N/A | N/A | 16.6667 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 20 | 10 | 2 | 6 | 16 | 39.1667 | 55.8333 | 40.8333 | 37.2500 | 38.8325 | 16.6667 | 15.0000 | 4.2500 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 20 | 2 | 2 | 6 | 8 | 39.1667 | 55.8333 | 45.0000 | 38.7500 | 41.3057 | 16.6667 | 10.8333 | 7.0833 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 20 | 20 | 2 | 6 | 26 | 39.1667 | 55.8333 | 37.5000 | 35.5000 | 36.4659 | 16.6667 | 18.3333 | 2.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 20 | 5 | 2 | 6 | 11 | 39.1667 | 55.8333 | 40.4167 | 34.5000 | 37.1889 | 16.6667 | 15.4167 | 5.9167 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 5 | 0 | 2 | 6 | 6 | 38.3333 | 33.3333 | 33.3333 | N/A | N/A | -5.0000 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 5 | 10 | 2 | 6 | 16 | 38.3333 | 33.3333 | 16.6667 | 32.0000 | 21.5775 | -5.0000 | 16.6667 | 15.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 5 | 2 | 2 | 6 | 8 | 38.3333 | 33.3333 | 20.0000 | 55.0000 | 28.1984 | -5.0000 | 13.3333 | 35.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 5 | 20 | 2 | 6 | 26 | 38.3333 | 33.3333 | 15.0000 | 27.0000 | 19.1339 | -5.0000 | 18.3333 | 12.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 5 | 5 | 2 | 6 | 11 | 38.3333 | 33.3333 | 21.6667 | 26.0000 | 23.6364 | -5.0000 | 11.6667 | 4.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 10 | 0 | 2 | 6 | 6 | 91.6667 | 96.6667 | 96.6667 | N/A | N/A | 5.0000 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 10 | 10 | 2 | 6 | 16 | 91.6667 | 96.6667 | 96.6667 | 81.0000 | 88.1393 | 5.0000 | 0.0000 | 15.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 10 | 2 | 2 | 6 | 8 | 91.6667 | 96.6667 | 96.6667 | 60.0000 | 74.0426 | 5.0000 | 0.0000 | 36.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 10 | 20 | 2 | 6 | 26 | 91.6667 | 96.6667 | 95.8333 | 75.5000 | 84.4552 | 5.0000 | 0.8333 | 20.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 10 | 5 | 2 | 6 | 11 | 91.6667 | 96.6667 | 96.6667 | 73.0000 | 83.1789 | 5.0000 | 0.0000 | 23.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 20 | 0 | 2 | 6 | 6 | 87.9167 | 97.0833 | 97.0833 | N/A | N/A | 9.1667 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 20 | 10 | 2 | 6 | 16 | 87.9167 | 97.0833 | 95.0000 | 84.0000 | 89.1549 | 9.1667 | 2.0833 | 11.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 20 | 2 | 2 | 6 | 8 | 87.9167 | 97.0833 | 96.2500 | 75.0000 | 84.3062 | 9.1667 | 0.8333 | 21.2500 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 20 | 20 | 2 | 6 | 26 | 87.9167 | 97.0833 | 95.4167 | 85.0000 | 89.9037 | 9.1667 | 1.6667 | 10.4167 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 20 | 5 | 2 | 6 | 11 | 87.9167 | 97.0833 | 95.4167 | 81.5000 | 87.8842 | 9.1667 | 1.6667 | 13.9167 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 5 | 0 | 2 | 6 | 6 | 91.6667 | 93.3333 | 93.3333 | N/A | N/A | 1.6667 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 5 | 10 | 2 | 6 | 16 | 91.6667 | 93.3333 | 85.0000 | 65.0000 | 73.6663 | 1.6667 | 8.3333 | 20.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 5 | 2 | 2 | 6 | 8 | 91.6667 | 93.3333 | 91.6667 | 50.0000 | 64.7010 | 1.6667 | 1.6667 | 41.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 5 | 20 | 2 | 6 | 26 | 91.6667 | 93.3333 | 85.0000 | 61.0000 | 71.0208 | 1.6667 | 8.3333 | 24.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_high | 5 | 5 | 2 | 6 | 11 | 91.6667 | 93.3333 | 90.0000 | 68.0000 | 77.4026 | 1.6667 | 3.3333 | 22.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 10 | 0 | 2 | 6 | 6 | 53.3333 | 47.5000 | 47.5000 | N/A | N/A | -5.8333 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 10 | 10 | 2 | 6 | 16 | 53.3333 | 47.5000 | 36.6667 | 38.0000 | 37.3021 | -5.8333 | 10.8333 | 1.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 10 | 2 | 2 | 6 | 8 | 53.3333 | 47.5000 | 40.0000 | 42.5000 | 40.5155 | -5.8333 | 7.5000 | 10.8333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 10 | 20 | 2 | 6 | 26 | 53.3333 | 47.5000 | 35.0000 | 38.0000 | 36.3886 | -5.8333 | 12.5000 | 3.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 10 | 5 | 2 | 6 | 11 | 53.3333 | 47.5000 | 37.5000 | 34.0000 | 35.6248 | -5.8333 | 10.0000 | 3.5000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 20 | 0 | 2 | 6 | 6 | 54.5833 | 57.5000 | 57.5000 | N/A | N/A | 2.9167 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 20 | 10 | 2 | 6 | 16 | 54.5833 | 57.5000 | 44.5833 | 39.0000 | 41.5417 | 2.9167 | 12.9167 | 5.5833 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 20 | 2 | 2 | 6 | 8 | 54.5833 | 57.5000 | 51.2500 | 47.5000 | 49.3030 | 2.9167 | 6.2500 | 3.7500 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 20 | 20 | 2 | 6 | 26 | 54.5833 | 57.5000 | 42.5000 | 40.5000 | 41.4321 | 2.9167 | 15.0000 | 2.7500 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 20 | 5 | 2 | 6 | 11 | 54.5833 | 57.5000 | 48.3333 | 37.5000 | 42.2284 | 2.9167 | 9.1667 | 10.8333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 5 | 0 | 2 | 6 | 6 | 48.3333 | 33.3333 | 33.3333 | N/A | N/A | -15.0000 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 5 | 10 | 2 | 6 | 16 | 48.3333 | 33.3333 | 18.3333 | 24.0000 | 20.4024 | -15.0000 | 15.0000 | 5.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 5 | 2 | 2 | 6 | 8 | 48.3333 | 33.3333 | 23.3333 | 80.0000 | 36.0000 | -15.0000 | 10.0000 | 56.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 5 | 20 | 2 | 6 | 26 | 48.3333 | 33.3333 | 15.0000 | 28.5000 | 18.9894 | -15.0000 | 18.3333 | 13.5000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 5 | 5 | 2 | 6 | 11 | 48.3333 | 33.3333 | 21.6667 | 28.0000 | 24.4101 | -15.0000 | 11.6667 | 6.3333 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 10 | 0 | 2 | 6 | 6 | 73.3333 | 72.4074 | 72.4074 | N/A | N/A | -0.9259 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 10 | 10 | 2 | 6 | 16 | 73.3333 | 72.4074 | 58.7963 | 57.8333 | 57.6941 | -0.9259 | 13.6111 | 9.1852 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 10 | 2 | 2 | 6 | 8 | 73.3333 | 72.4074 | 67.1296 | 50.8333 | 55.9931 | -0.9259 | 5.2778 | 19.6296 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 10 | 20 | 2 | 6 | 26 | 73.3333 | 72.4074 | 53.5185 | 50.6111 | 51.3250 | -0.9259 | 18.8889 | 9.9444 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 10 | 5 | 2 | 6 | 11 | 73.3333 | 72.4074 | 60.6481 | 63.2222 | 61.4309 | -0.9259 | 11.7593 | 9.4259 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 20 | 0 | 2 | 6 | 6 | 64.1667 | 69.0132 | 69.0132 | N/A | N/A | 4.8465 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 20 | 10 | 2 | 6 | 16 | 64.1667 | 69.0132 | 54.6053 | 59.0132 | 56.2767 | 4.8465 | 14.4079 | 8.9254 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 20 | 2 | 2 | 6 | 8 | 64.1667 | 69.0132 | 63.7281 | 61.5132 | 61.8796 | 4.8465 | 5.2851 | 11.3816 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 20 | 20 | 2 | 6 | 26 | 64.1667 | 69.0132 | 49.5833 | 51.8355 | 50.2262 | 4.8465 | 19.4298 | 7.8838 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 20 | 5 | 2 | 6 | 11 | 64.1667 | 69.0132 | 57.8289 | 59.9474 | 58.4660 | 4.8465 | 11.1842 | 7.9781 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 5 | 0 | 2 | 6 | 6 | 70.0000 | 66.6667 | 66.6667 | N/A | N/A | -3.3333 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 5 | 10 | 2 | 6 | 16 | 70.0000 | 66.6667 | 56.2500 | 60.0000 | 57.5352 | -3.3333 | 10.4167 | 10.2500 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 5 | 2 | 2 | 6 | 8 | 70.0000 | 66.6667 | 62.0833 | 68.7500 | 64.0622 | -3.3333 | 4.5833 | 15.8333 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 5 | 20 | 2 | 6 | 26 | 70.0000 | 66.6667 | 45.0000 | 53.6250 | 48.5495 | -3.3333 | 21.6667 | 11.2083 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_high | 5 | 5 | 2 | 6 | 11 | 70.0000 | 66.6667 | 57.9167 | 72.5000 | 63.6055 | -3.3333 | 8.7500 | 17.9167 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 10 | 0 | 2 | 6 | 6 | 39.1667 | 25.2778 | 25.2778 | N/A | N/A | -13.8889 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 10 | 10 | 2 | 6 | 16 | 39.1667 | 25.2778 | 6.4815 | 18.9444 | 8.2432 | -13.8889 | 18.7963 | 12.6852 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 10 | 2 | 2 | 6 | 8 | 39.1667 | 25.2778 | 8.1481 | 43.0556 | 12.1921 | -13.8889 | 17.1296 | 34.9074 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 10 | 20 | 2 | 6 | 26 | 39.1667 | 25.2778 | 5.7407 | 16.3889 | 7.5808 | -13.8889 | 19.5370 | 10.6481 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 10 | 5 | 2 | 6 | 11 | 39.1667 | 25.2778 | 7.5000 | 22.8889 | 9.8963 | -13.8889 | 17.7778 | 15.5370 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 20 | 0 | 2 | 6 | 6 | 39.1667 | 26.0965 | 26.0965 | N/A | N/A | -13.0702 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 20 | 10 | 2 | 6 | 16 | 39.1667 | 26.0965 | 8.8596 | 19.4079 | 10.9850 | -13.0702 | 17.2368 | 11.0307 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 20 | 2 | 2 | 6 | 8 | 39.1667 | 26.0965 | 11.0088 | 41.4474 | 16.4884 | -13.0702 | 15.0877 | 30.4386 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 20 | 20 | 2 | 6 | 26 | 39.1667 | 26.0965 | 8.3772 | 14.7829 | 9.7020 | -13.0702 | 17.7193 | 7.5197 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 20 | 5 | 2 | 6 | 11 | 39.1667 | 26.0965 | 9.6930 | 23.3947 | 12.7679 | -13.0702 | 16.4035 | 13.9123 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 5 | 0 | 2 | 6 | 6 | 38.3333 | 25.8333 | 25.8333 | N/A | N/A | -12.5000 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 5 | 10 | 2 | 6 | 16 | 38.3333 | 25.8333 | 8.3333 | 18.0000 | 10.1489 | -12.5000 | 17.5000 | 11.0000 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 5 | 2 | 2 | 6 | 8 | 38.3333 | 25.8333 | 11.2500 | 28.7500 | 14.2434 | -12.5000 | 14.5833 | 19.1667 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 5 | 20 | 2 | 6 | 26 | 38.3333 | 25.8333 | 7.0833 | 14.7500 | 8.3481 | -12.5000 | 18.7500 | 7.6667 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 19-1 | practical_low_urban | 5 | 5 | 2 | 6 | 11 | 38.3333 | 25.8333 | 7.9167 | 21.0000 | 10.5385 | -12.5000 | 17.9167 | 13.0833 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 10 | 0 | 2 | 6 | 6 | 91.6667 | 86.6667 | 86.6667 | N/A | N/A | -5.0000 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 10 | 10 | 2 | 6 | 16 | 91.6667 | 86.6667 | 75.5556 | 56.7222 | 64.5986 | -5.0000 | 11.1111 | 18.8333 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 10 | 2 | 2 | 6 | 8 | 91.6667 | 86.6667 | 85.3704 | 33.6111 | 46.1399 | -5.0000 | 1.2963 | 51.7593 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 10 | 20 | 2 | 6 | 26 | 91.6667 | 86.6667 | 74.1667 | 48.0833 | 58.1921 | -5.0000 | 12.5000 | 26.0833 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 10 | 5 | 2 | 6 | 11 | 91.6667 | 86.6667 | 81.4815 | 52.0000 | 63.1567 | -5.0000 | 5.1852 | 29.4815 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 20 | 0 | 2 | 6 | 6 | 87.9167 | 84.9342 | 84.9342 | N/A | N/A | -2.9825 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 20 | 10 | 2 | 6 | 16 | 87.9167 | 84.9342 | 71.7105 | 51.5132 | 59.4705 | -2.9825 | 13.2237 | 20.2500 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 20 | 2 | 2 | 6 | 8 | 87.9167 | 84.9342 | 80.5482 | 34.2763 | 46.9099 | -2.9825 | 4.3860 | 46.2719 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 20 | 20 | 2 | 6 | 26 | 87.9167 | 84.9342 | 69.8684 | 48.1118 | 56.6114 | -2.9825 | 15.0658 | 21.9189 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 20 | 5 | 2 | 6 | 11 | 87.9167 | 84.9342 | 76.6009 | 50.7368 | 60.4402 | -2.9825 | 8.3333 | 25.8640 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 5 | 0 | 2 | 6 | 6 | 91.6667 | 90.8333 | 90.8333 | N/A | N/A | -0.8333 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 5 | 10 | 2 | 6 | 16 | 91.6667 | 90.8333 | 77.5000 | 60.2500 | 67.2146 | -0.8333 | 13.3333 | 17.9167 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 5 | 2 | 2 | 6 | 8 | 91.6667 | 90.8333 | 89.5833 | 32.5000 | 47.1783 | -0.8333 | 1.2500 | 57.0833 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 5 | 20 | 2 | 6 | 26 | 91.6667 | 90.8333 | 73.7500 | 45.6250 | 55.9226 | -0.8333 | 17.0833 | 28.1250 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_high | 5 | 5 | 2 | 6 | 11 | 91.6667 | 90.8333 | 82.5000 | 55.5000 | 65.7185 | -0.8333 | 8.3333 | 27.0000 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 10 | 0 | 2 | 6 | 6 | 53.3333 | 27.7778 | 27.7778 | N/A | N/A | -25.5556 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 10 | 10 | 2 | 6 | 16 | 53.3333 | 27.7778 | 14.0741 | 19.1111 | 14.6613 | -25.5556 | 13.7037 | 9.6667 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 10 | 2 | 2 | 6 | 8 | 53.3333 | 27.7778 | 15.6481 | 35.5556 | 20.7930 | -25.5556 | 12.1296 | 20.0926 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 10 | 20 | 2 | 6 | 26 | 53.3333 | 27.7778 | 12.7778 | 17.3056 | 13.5244 | -25.5556 | 15.0000 | 7.4167 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 10 | 5 | 2 | 6 | 11 | 53.3333 | 27.7778 | 14.8148 | 20.3333 | 15.6244 | -25.5556 | 12.9630 | 11.2593 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 20 | 0 | 2 | 6 | 6 | 54.5833 | 29.8465 | 29.8465 | N/A | N/A | -24.7368 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 20 | 10 | 2 | 6 | 16 | 54.5833 | 29.8465 | 13.1360 | 19.5526 | 14.1606 | -24.7368 | 16.7105 | 9.7149 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 20 | 2 | 2 | 6 | 8 | 54.5833 | 29.8465 | 17.1053 | 40.9211 | 22.3904 | -24.7368 | 12.7412 | 24.5614 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 20 | 20 | 2 | 6 | 26 | 54.5833 | 29.8465 | 12.7412 | 16.4803 | 13.1212 | -24.7368 | 17.1053 | 7.4057 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 20 | 5 | 2 | 6 | 11 | 54.5833 | 29.8465 | 14.2544 | 23.5000 | 15.7086 | -24.7368 | 15.5921 | 12.5263 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 5 | 0 | 2 | 6 | 6 | 48.3333 | 20.4167 | 20.4167 | N/A | N/A | -27.9167 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 5 | 10 | 2 | 6 | 16 | 48.3333 | 20.4167 | 11.6667 | 17.2500 | 12.8687 | -27.9167 | 8.7500 | 8.0833 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 5 | 2 | 2 | 6 | 8 | 48.3333 | 20.4167 | 11.6667 | 61.2500 | 19.0828 | -27.9167 | 8.7500 | 49.5833 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 5 | 20 | 2 | 6 | 26 | 48.3333 | 20.4167 | 11.6667 | 20.0000 | 14.4565 | -27.9167 | 8.7500 | 8.3333 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 20-19 | practical_low_urban | 5 | 5 | 2 | 6 | 11 | 48.3333 | 20.4167 | 11.6667 | 24.0000 | 14.8094 | -27.9167 | 8.7500 | 13.0000 |

by_model_row

| method | diagnostic | path | model_seed | row_id | k | new_count | measured_parents | 旧类数 | 总注册类数 | A旧 | B旧 | C旧 | C新 | H | B−A | 注册下降 | 绝对差 |
|---|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 0 | 2 | 6 | 6 | 70.8333 | 72.5000 | 72.5000 | N/A | N/A | 1.6667 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 10 | 2 | 6 | 16 | 70.8333 | 72.5000 | 71.6667 | 59.0000 | 64.6568 | 1.6667 | 0.8333 | 12.6667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 2 | 2 | 6 | 8 | 70.8333 | 72.5000 | 72.5000 | 37.5000 | 49.2717 | 1.6667 | 0.0000 | 35.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 20 | 2 | 6 | 26 | 70.8333 | 72.5000 | 71.6667 | 56.5000 | 63.1857 | 1.6667 | 0.8333 | 15.1667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 5 | 2 | 6 | 11 | 70.8333 | 72.5000 | 70.8333 | 49.0000 | 57.7437 | 1.6667 | 1.6667 | 21.8333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 0 | 2 | 6 | 6 | 70.0000 | 77.5000 | 77.5000 | N/A | N/A | 7.5000 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 10 | 2 | 6 | 16 | 70.0000 | 77.5000 | 70.8333 | 64.0000 | 67.1182 | 7.5000 | 6.6667 | 6.8333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 2 | 2 | 6 | 8 | 70.0000 | 77.5000 | 76.6667 | 43.7500 | 55.3051 | 7.5000 | 0.8333 | 32.9167 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 20 | 2 | 6 | 26 | 70.0000 | 77.5000 | 70.0000 | 64.1250 | 66.9267 | 7.5000 | 7.5000 | 5.8750 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 5 | 2 | 6 | 11 | 70.0000 | 77.5000 | 72.0833 | 60.0000 | 65.1316 | 7.5000 | 5.4167 | 12.0833 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 0 | 2 | 6 | 6 | 68.3333 | 65.0000 | 65.0000 | N/A | N/A | -3.3333 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 10 | 2 | 6 | 16 | 68.3333 | 65.0000 | 56.6667 | 40.0000 | 46.8142 | -3.3333 | 8.3333 | 16.6667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 2 | 2 | 6 | 8 | 68.3333 | 65.0000 | 63.3333 | 40.0000 | 46.1818 | -3.3333 | 1.6667 | 30.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 20 | 2 | 6 | 26 | 68.3333 | 65.0000 | 53.3333 | 43.5000 | 47.1057 | -3.3333 | 11.6667 | 15.8333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 5 | 2 | 6 | 11 | 68.3333 | 65.0000 | 58.3333 | 40.0000 | 47.2489 | -3.3333 | 6.6667 | 18.3333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 0 | 2 | 6 | 6 | 54.1667 | 74.1667 | 74.1667 | N/A | N/A | 20.0000 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 10 | 2 | 6 | 16 | 54.1667 | 74.1667 | 66.6667 | 52.5000 | 58.6364 | 20.0000 | 7.5000 | 14.1667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 2 | 2 | 6 | 8 | 54.1667 | 74.1667 | 74.1667 | 62.5000 | 67.6428 | 20.0000 | 0.0000 | 11.6667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 20 | 2 | 6 | 26 | 54.1667 | 74.1667 | 65.0000 | 52.2500 | 57.8766 | 20.0000 | 9.1667 | 12.7500 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 5 | 2 | 6 | 11 | 54.1667 | 74.1667 | 69.1667 | 61.0000 | 64.3959 | 20.0000 | 5.0000 | 8.1667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 0 | 2 | 6 | 6 | 50.4167 | 75.4167 | 75.4167 | N/A | N/A | 25.0000 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 10 | 2 | 6 | 16 | 50.4167 | 75.4167 | 70.8333 | 62.2500 | 66.1069 | 25.0000 | 4.5833 | 8.5833 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 2 | 2 | 6 | 8 | 50.4167 | 75.4167 | 74.5833 | 55.0000 | 62.3947 | 25.0000 | 0.8333 | 19.5833 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 20 | 2 | 6 | 26 | 50.4167 | 75.4167 | 68.3333 | 60.6250 | 64.0453 | 25.0000 | 7.0833 | 7.7083 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 5 | 2 | 6 | 11 | 50.4167 | 75.4167 | 71.2500 | 61.5000 | 65.4228 | 25.0000 | 4.1667 | 9.7500 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 0 | 2 | 6 | 6 | 53.3333 | 61.6667 | 61.6667 | N/A | N/A | 8.3333 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 10 | 2 | 6 | 16 | 53.3333 | 61.6667 | 56.6667 | 53.0000 | 54.7720 | 8.3333 | 5.0000 | 3.6667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 2 | 2 | 6 | 8 | 53.3333 | 61.6667 | 60.0000 | 45.0000 | 51.3750 | 8.3333 | 1.6667 | 15.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 20 | 2 | 6 | 26 | 53.3333 | 61.6667 | 55.0000 | 42.5000 | 47.6078 | 8.3333 | 6.6667 | 12.5000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 5 | 2 | 6 | 11 | 53.3333 | 61.6667 | 56.6667 | 50.0000 | 52.2675 | 8.3333 | 5.0000 | 7.3333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 0 | 2 | 6 | 6 | 74.1667 | 74.1667 | 74.1667 | N/A | N/A | 0.0000 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 10 | 2 | 6 | 16 | 74.1667 | 74.1667 | 71.6667 | 57.5000 | 63.7520 | 0.0000 | 2.5000 | 14.1667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 2 | 2 | 6 | 8 | 74.1667 | 74.1667 | 74.1667 | 37.5000 | 49.8127 | 0.0000 | 0.0000 | 36.6667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 20 | 2 | 6 | 26 | 74.1667 | 74.1667 | 66.6667 | 55.5000 | 60.5564 | 0.0000 | 7.5000 | 11.1667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 5 | 2 | 6 | 11 | 74.1667 | 74.1667 | 72.5000 | 50.0000 | 59.0656 | 0.0000 | 1.6667 | 22.5000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 0 | 2 | 6 | 6 | 72.5000 | 76.6667 | 76.6667 | N/A | N/A | 4.1667 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 10 | 2 | 6 | 16 | 72.5000 | 76.6667 | 72.5000 | 62.7500 | 67.1389 | 4.1667 | 4.1667 | 9.7500 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 2 | 2 | 6 | 8 | 72.5000 | 76.6667 | 76.2500 | 51.2500 | 60.5880 | 4.1667 | 0.4167 | 25.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 20 | 2 | 6 | 26 | 72.5000 | 76.6667 | 71.6667 | 63.1250 | 67.1195 | 4.1667 | 5.0000 | 8.5417 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 5 | 2 | 6 | 11 | 72.5000 | 76.6667 | 73.7500 | 60.0000 | 65.8065 | 4.1667 | 2.9167 | 13.7500 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 0 | 2 | 6 | 6 | 71.6667 | 63.3333 | 63.3333 | N/A | N/A | -8.3333 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 10 | 2 | 6 | 16 | 71.6667 | 63.3333 | 58.3333 | 38.0000 | 45.6785 | -8.3333 | 5.0000 | 20.3333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 2 | 2 | 6 | 8 | 71.6667 | 63.3333 | 63.3333 | 35.0000 | 40.8845 | -8.3333 | 0.0000 | 35.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 20 | 2 | 6 | 26 | 71.6667 | 63.3333 | 53.3333 | 43.5000 | 46.3053 | -8.3333 | 10.0000 | 20.1667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 5 | 2 | 6 | 11 | 71.6667 | 63.3333 | 60.0000 | 36.0000 | 44.9553 | -8.3333 | 3.3333 | 24.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 0 | 2 | 6 | 6 | 58.3333 | 73.3333 | 73.3333 | N/A | N/A | 15.0000 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 10 | 2 | 6 | 16 | 58.3333 | 73.3333 | 65.8333 | 56.5000 | 60.8100 | 15.0000 | 7.5000 | 9.3333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 2 | 2 | 6 | 8 | 58.3333 | 73.3333 | 71.6667 | 62.5000 | 66.6121 | 15.0000 | 1.6667 | 9.1667 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 20 | 2 | 6 | 26 | 58.3333 | 73.3333 | 65.0000 | 56.2500 | 60.2946 | 15.0000 | 8.3333 | 8.7500 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 5 | 2 | 6 | 11 | 58.3333 | 73.3333 | 68.3333 | 58.0000 | 62.5343 | 15.0000 | 5.0000 | 10.3333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 0 | 2 | 6 | 6 | 52.9167 | 76.6667 | 76.6667 | N/A | N/A | 23.7500 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 10 | 2 | 6 | 16 | 52.9167 | 76.6667 | 70.8333 | 60.5000 | 64.9331 | 23.7500 | 5.8333 | 10.3333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 2 | 2 | 6 | 8 | 52.9167 | 76.6667 | 76.2500 | 51.2500 | 59.7072 | 23.7500 | 0.4167 | 25.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 20 | 2 | 6 | 26 | 52.9167 | 76.6667 | 69.1667 | 60.1250 | 64.1369 | 23.7500 | 7.5000 | 9.0417 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 5 | 2 | 6 | 11 | 52.9167 | 76.6667 | 72.0833 | 57.5000 | 62.9280 | 23.7500 | 4.5833 | 14.5833 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 0 | 2 | 6 | 6 | 55.0000 | 63.3333 | 63.3333 | N/A | N/A | 8.3333 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 10 | 2 | 6 | 16 | 55.0000 | 63.3333 | 58.3333 | 56.0000 | 57.1390 | 8.3333 | 5.0000 | 2.3333 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 2 | 2 | 6 | 8 | 55.0000 | 63.3333 | 60.0000 | 55.0000 | 57.0000 | 8.3333 | 3.3333 | 5.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 20 | 2 | 6 | 26 | 55.0000 | 63.3333 | 55.0000 | 50.0000 | 52.3441 | 8.3333 | 8.3333 | 5.0000 |
| D92-AffineJointLocalRidge-v1 | oof | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 5 | 2 | 6 | 11 | 55.0000 | 63.3333 | 56.6667 | 52.0000 | 53.6641 | 8.3333 | 6.6667 | 6.0000 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 0 | 2 | 6 | 6 | 70.8333 | 58.1481 | 58.1481 | N/A | N/A | -12.6852 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 10 | 2 | 6 | 16 | 70.8333 | 58.1481 | 52.3148 | 31.6111 | 38.8836 | -12.6852 | 5.8333 | 21.0741 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 2 | 2 | 6 | 8 | 70.8333 | 58.1481 | 56.7593 | 14.7222 | 19.9145 | -12.6852 | 1.3889 | 43.1481 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 20 | 2 | 6 | 26 | 70.8333 | 58.1481 | 50.4630 | 29.3333 | 36.3460 | -12.6852 | 7.6852 | 22.7963 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 5 | 2 | 6 | 11 | 70.8333 | 58.1481 | 55.5556 | 27.8889 | 36.2713 | -12.6852 | 2.5926 | 28.2593 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 0 | 2 | 6 | 6 | 70.0000 | 57.3904 | 57.3904 | N/A | N/A | -12.6096 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 10 | 2 | 6 | 16 | 70.0000 | 57.3904 | 49.4737 | 29.4605 | 36.1325 | -12.6096 | 7.9167 | 21.2237 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 2 | 2 | 6 | 8 | 70.0000 | 57.3904 | 55.8333 | 15.1974 | 21.6159 | -12.6096 | 1.5570 | 42.2588 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 20 | 2 | 6 | 26 | 70.0000 | 57.3904 | 47.8070 | 29.5855 | 35.9997 | -12.6096 | 9.5833 | 19.1952 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 5 | 2 | 6 | 11 | 70.0000 | 57.3904 | 52.8070 | 28.1842 | 35.5163 | -12.6096 | 4.5833 | 25.6404 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 0 | 2 | 6 | 6 | 68.3333 | 54.5833 | 54.5833 | N/A | N/A | -13.7500 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 10 | 2 | 6 | 16 | 68.3333 | 54.5833 | 50.0000 | 32.7500 | 38.6764 | -13.7500 | 4.5833 | 18.4167 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 2 | 2 | 6 | 8 | 68.3333 | 54.5833 | 54.1667 | 18.7500 | 20.7090 | -13.7500 | 0.4167 | 44.5833 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 20 | 2 | 6 | 26 | 68.3333 | 54.5833 | 49.5833 | 29.5000 | 35.8506 | -13.7500 | 5.0000 | 21.9167 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 5 | 2 | 6 | 11 | 68.3333 | 54.5833 | 52.5000 | 30.0000 | 37.2540 | -13.7500 | 2.0833 | 22.8333 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 0 | 2 | 6 | 6 | 54.1667 | 49.5370 | 49.5370 | N/A | N/A | -4.6296 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 10 | 2 | 6 | 16 | 54.1667 | 49.5370 | 44.6296 | 33.6111 | 37.6485 | -4.6296 | 4.9074 | 13.3148 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 2 | 2 | 6 | 8 | 54.1667 | 49.5370 | 48.1481 | 26.9444 | 31.3964 | -4.6296 | 1.3889 | 24.1667 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 20 | 2 | 6 | 26 | 54.1667 | 49.5370 | 42.2222 | 30.0278 | 34.6442 | -4.6296 | 7.3148 | 13.9537 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 5 | 2 | 6 | 11 | 54.1667 | 49.5370 | 46.4815 | 35.0000 | 38.9933 | -4.6296 | 3.0556 | 12.4444 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 0 | 2 | 6 | 6 | 50.4167 | 47.3465 | 47.3465 | N/A | N/A | -3.0702 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 10 | 2 | 6 | 16 | 50.4167 | 47.3465 | 41.7325 | 34.6974 | 37.0409 | -3.0702 | 5.6140 | 10.2193 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 2 | 2 | 6 | 8 | 50.4167 | 47.3465 | 45.7018 | 31.1184 | 35.1818 | -3.0702 | 1.6447 | 16.4693 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 20 | 2 | 6 | 26 | 50.4167 | 47.3465 | 38.7281 | 30.6908 | 33.6601 | -3.0702 | 8.6184 | 10.3969 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 5 | 2 | 6 | 11 | 50.4167 | 47.3465 | 43.7719 | 32.6842 | 36.4219 | -3.0702 | 3.5746 | 12.3421 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 0 | 2 | 6 | 6 | 53.3333 | 45.8333 | 45.8333 | N/A | N/A | -7.5000 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 10 | 2 | 6 | 16 | 53.3333 | 45.8333 | 39.5833 | 34.5000 | 36.2728 | -7.5000 | 6.2500 | 7.7500 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 2 | 2 | 6 | 8 | 53.3333 | 45.8333 | 43.7500 | 30.0000 | 33.2127 | -7.5000 | 2.0833 | 13.7500 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 20 | 2 | 6 | 26 | 53.3333 | 45.8333 | 37.0833 | 31.2500 | 33.4403 | -7.5000 | 8.7500 | 7.4167 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 5 | 2 | 6 | 11 | 53.3333 | 45.8333 | 40.8333 | 36.0000 | 36.9348 | -7.5000 | 5.0000 | 11.3333 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 0 | 2 | 6 | 6 | 74.1667 | 56.2963 | 56.2963 | N/A | N/A | -17.8704 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 10 | 2 | 6 | 16 | 74.1667 | 56.2963 | 51.0185 | 33.4444 | 39.6663 | -17.8704 | 5.2778 | 18.9815 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 2 | 2 | 6 | 8 | 74.1667 | 56.2963 | 55.3704 | 17.2222 | 23.9437 | -17.8704 | 0.9259 | 38.3333 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 20 | 2 | 6 | 26 | 74.1667 | 56.2963 | 49.4444 | 30.4722 | 36.9530 | -17.8704 | 6.8519 | 20.9167 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 5 | 2 | 6 | 11 | 74.1667 | 56.2963 | 54.2593 | 30.6667 | 38.0196 | -17.8704 | 2.0370 | 24.3704 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 0 | 2 | 6 | 6 | 72.5000 | 57.3904 | 57.3904 | N/A | N/A | -15.1096 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 10 | 2 | 6 | 16 | 72.5000 | 57.3904 | 50.9649 | 31.0658 | 37.7741 | -15.1096 | 6.4254 | 21.2412 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 2 | 2 | 6 | 8 | 72.5000 | 57.3904 | 56.3158 | 15.8553 | 22.6336 | -15.1096 | 1.0746 | 42.0833 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 20 | 2 | 6 | 26 | 72.5000 | 57.3904 | 48.9035 | 29.8684 | 36.4923 | -15.1096 | 8.4868 | 20.1623 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 5 | 2 | 6 | 11 | 72.5000 | 57.3904 | 53.8816 | 31.4211 | 38.4198 | -15.1096 | 3.5088 | 23.7500 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 0 | 2 | 6 | 6 | 71.6667 | 56.6667 | 56.6667 | N/A | N/A | -15.0000 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 10 | 2 | 6 | 16 | 71.6667 | 56.6667 | 52.9167 | 32.5000 | 39.4708 | -15.0000 | 3.7500 | 21.9167 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 2 | 2 | 6 | 8 | 71.6667 | 56.6667 | 56.2500 | 21.2500 | 21.2974 | -15.0000 | 0.4167 | 45.8333 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 20 | 2 | 6 | 26 | 71.6667 | 56.6667 | 51.2500 | 29.3750 | 36.0855 | -15.0000 | 5.4167 | 24.3750 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 5 | 2 | 6 | 11 | 71.6667 | 56.6667 | 54.1667 | 30.5000 | 38.1285 | -15.0000 | 2.5000 | 23.6667 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 0 | 2 | 6 | 6 | 58.3333 | 48.1481 | 48.1481 | N/A | N/A | -10.1852 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 10 | 2 | 6 | 16 | 58.3333 | 48.1481 | 43.7037 | 33.5556 | 37.2939 | -10.1852 | 4.4444 | 12.1852 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 2 | 2 | 6 | 8 | 58.3333 | 48.1481 | 48.2407 | 25.0000 | 29.5447 | -10.1852 | -0.0926 | 24.9074 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 20 | 2 | 6 | 26 | 58.3333 | 48.1481 | 40.7407 | 30.5278 | 34.2286 | -10.1852 | 7.4074 | 13.2870 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 5 | 2 | 6 | 11 | 58.3333 | 48.1481 | 45.8333 | 32.2222 | 37.0726 | -10.1852 | 2.3148 | 13.9444 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 0 | 2 | 6 | 6 | 52.9167 | 47.7632 | 47.7632 | N/A | N/A | -5.1535 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 10 | 2 | 6 | 16 | 52.9167 | 47.7632 | 41.6447 | 34.6842 | 37.0239 | -5.1535 | 6.1184 | 9.9167 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 2 | 2 | 6 | 8 | 52.9167 | 47.7632 | 45.7018 | 31.1184 | 34.7271 | -5.1535 | 2.0614 | 17.5219 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 20 | 2 | 6 | 26 | 52.9167 | 47.7632 | 38.5526 | 30.6711 | 33.6534 | -5.1535 | 9.2105 | 9.9254 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 5 | 2 | 6 | 11 | 52.9167 | 47.7632 | 43.4649 | 33.6316 | 36.7859 | -5.1535 | 4.2982 | 11.7368 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 0 | 2 | 6 | 6 | 55.0000 | 46.6667 | 46.6667 | N/A | N/A | -8.3333 | 0.0000 | N/A |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 10 | 2 | 6 | 16 | 55.0000 | 46.6667 | 42.5000 | 35.5000 | 38.0281 | -8.3333 | 4.1667 | 9.8333 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 2 | 2 | 6 | 8 | 55.0000 | 46.6667 | 47.9167 | 40.0000 | 39.8141 | -8.3333 | -1.2500 | 17.0833 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 20 | 2 | 6 | 26 | 55.0000 | 46.6667 | 38.7500 | 33.6250 | 35.5446 | -8.3333 | 7.9167 | 6.7917 |
| D92-AffineJointLocalRidge-v1 | proxy | R_AFFINE_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 5 | 2 | 6 | 11 | 55.0000 | 46.6667 | 44.5833 | 39.0000 | 39.6762 | -8.3333 | 2.0833 | 12.4167 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 0 | 2 | 6 | 6 | 70.8333 | 71.6667 | 71.6667 | N/A | N/A | 0.8333 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 10 | 2 | 6 | 16 | 70.8333 | 71.6667 | 67.5000 | 60.0000 | 63.4486 | 0.8333 | 4.1667 | 7.5000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 2 | 2 | 6 | 8 | 70.8333 | 71.6667 | 70.0000 | 47.5000 | 56.3830 | 0.8333 | 1.6667 | 22.5000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 20 | 2 | 6 | 26 | 70.8333 | 71.6667 | 66.6667 | 56.0000 | 60.6493 | 0.8333 | 5.0000 | 11.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 5 | 2 | 6 | 11 | 70.8333 | 71.6667 | 68.3333 | 54.0000 | 60.2924 | 0.8333 | 3.3333 | 14.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 0 | 2 | 6 | 6 | 70.0000 | 77.5000 | 77.5000 | N/A | N/A | 7.5000 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 10 | 2 | 6 | 16 | 70.0000 | 77.5000 | 68.3333 | 61.0000 | 64.4321 | 7.5000 | 9.1667 | 7.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 2 | 2 | 6 | 8 | 70.0000 | 77.5000 | 73.7500 | 61.2500 | 66.8211 | 7.5000 | 3.7500 | 12.5000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 20 | 2 | 6 | 26 | 70.0000 | 77.5000 | 67.5000 | 63.2500 | 65.2549 | 7.5000 | 10.0000 | 5.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 5 | 2 | 6 | 11 | 70.0000 | 77.5000 | 71.2500 | 58.0000 | 63.9406 | 7.5000 | 6.2500 | 13.2500 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 0 | 2 | 6 | 6 | 68.3333 | 65.0000 | 65.0000 | N/A | N/A | -3.3333 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 10 | 2 | 6 | 16 | 68.3333 | 65.0000 | 55.0000 | 45.0000 | 49.2982 | -3.3333 | 10.0000 | 10.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 2 | 2 | 6 | 8 | 68.3333 | 65.0000 | 58.3333 | 65.0000 | 52.1429 | -3.3333 | 6.6667 | 46.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 20 | 2 | 6 | 26 | 68.3333 | 65.0000 | 53.3333 | 44.0000 | 47.2907 | -3.3333 | 11.6667 | 16.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 5 | 2 | 6 | 11 | 68.3333 | 65.0000 | 58.3333 | 54.0000 | 55.3191 | -3.3333 | 6.6667 | 13.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 0 | 2 | 6 | 6 | 54.1667 | 70.8333 | 70.8333 | N/A | N/A | 16.6667 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 10 | 2 | 6 | 16 | 54.1667 | 70.8333 | 60.0000 | 56.5000 | 57.9864 | 16.6667 | 10.8333 | 7.8333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 2 | 2 | 6 | 8 | 54.1667 | 70.8333 | 65.0000 | 70.0000 | 65.8355 | 16.6667 | 5.8333 | 18.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 20 | 2 | 6 | 26 | 54.1667 | 70.8333 | 57.5000 | 52.7500 | 54.7987 | 16.6667 | 13.3333 | 8.4167 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 5 | 2 | 6 | 11 | 54.1667 | 70.8333 | 61.6667 | 64.0000 | 62.6762 | 16.6667 | 9.1667 | 4.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 0 | 2 | 6 | 6 | 50.4167 | 75.8333 | 75.8333 | N/A | N/A | 25.4167 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 10 | 2 | 6 | 16 | 50.4167 | 75.8333 | 65.4167 | 63.7500 | 64.5593 | 25.4167 | 10.4167 | 2.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 2 | 2 | 6 | 8 | 50.4167 | 75.8333 | 69.1667 | 63.7500 | 66.2638 | 25.4167 | 6.6667 | 6.2500 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 20 | 2 | 6 | 26 | 50.4167 | 75.8333 | 63.7500 | 60.7500 | 62.2113 | 25.4167 | 12.0833 | 3.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 5 | 2 | 6 | 11 | 50.4167 | 75.8333 | 65.4167 | 63.0000 | 64.1597 | 25.4167 | 10.4167 | 2.4167 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 0 | 2 | 6 | 6 | 53.3333 | 63.3333 | 63.3333 | N/A | N/A | 10.0000 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 10 | 2 | 6 | 16 | 53.3333 | 63.3333 | 50.0000 | 54.0000 | 51.4937 | 10.0000 | 13.3333 | 6.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 2 | 2 | 6 | 8 | 53.3333 | 63.3333 | 51.6667 | 70.0000 | 52.1849 | 10.0000 | 11.6667 | 35.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 20 | 2 | 6 | 26 | 53.3333 | 63.3333 | 45.0000 | 47.0000 | 45.5504 | 10.0000 | 18.3333 | 6.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 5 | 2 | 6 | 11 | 53.3333 | 63.3333 | 53.3333 | 54.0000 | 53.5436 | 10.0000 | 10.0000 | 4.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 0 | 2 | 6 | 6 | 74.1667 | 72.5000 | 72.5000 | N/A | N/A | -1.6667 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 10 | 2 | 6 | 16 | 74.1667 | 72.5000 | 65.8333 | 59.0000 | 61.9928 | -1.6667 | 6.6667 | 9.8333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 2 | 2 | 6 | 8 | 74.1667 | 72.5000 | 66.6667 | 55.0000 | 58.1751 | -1.6667 | 5.8333 | 25.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 20 | 2 | 6 | 26 | 74.1667 | 72.5000 | 64.1667 | 57.5000 | 60.1946 | -1.6667 | 8.3333 | 12.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 5 | 2 | 6 | 11 | 74.1667 | 72.5000 | 65.8333 | 53.0000 | 58.5112 | -1.6667 | 6.6667 | 12.8333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 0 | 2 | 6 | 6 | 72.5000 | 77.0833 | 77.0833 | N/A | N/A | 4.5833 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 10 | 2 | 6 | 16 | 72.5000 | 77.0833 | 71.2500 | 62.0000 | 66.2645 | 4.5833 | 5.8333 | 9.2500 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 2 | 2 | 6 | 8 | 72.5000 | 77.0833 | 73.7500 | 61.2500 | 66.7881 | 4.5833 | 3.3333 | 12.5000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 20 | 2 | 6 | 26 | 72.5000 | 77.0833 | 70.4167 | 62.2500 | 66.0809 | 4.5833 | 6.6667 | 8.1667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 5 | 2 | 6 | 11 | 72.5000 | 77.0833 | 72.5000 | 61.0000 | 66.1719 | 4.5833 | 4.5833 | 11.5000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 0 | 2 | 6 | 6 | 71.6667 | 61.6667 | 61.6667 | N/A | N/A | -10.0000 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 10 | 2 | 6 | 16 | 71.6667 | 61.6667 | 48.3333 | 44.0000 | 44.7705 | -10.0000 | 13.3333 | 15.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 2 | 2 | 6 | 8 | 71.6667 | 61.6667 | 56.6667 | 65.0000 | 48.5581 | -10.0000 | 5.0000 | 51.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 20 | 2 | 6 | 26 | 71.6667 | 61.6667 | 46.6667 | 45.5000 | 42.7194 | -10.0000 | 15.0000 | 21.1667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 5 | 2 | 6 | 11 | 71.6667 | 61.6667 | 53.3333 | 42.0000 | 46.4935 | -10.0000 | 8.3333 | 14.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 0 | 2 | 6 | 6 | 58.3333 | 73.3333 | 73.3333 | N/A | N/A | 15.0000 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 10 | 2 | 6 | 16 | 58.3333 | 73.3333 | 59.1667 | 59.5000 | 59.1415 | 15.0000 | 14.1667 | 6.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 2 | 2 | 6 | 8 | 58.3333 | 73.3333 | 67.5000 | 67.5000 | 66.3629 | 15.0000 | 5.8333 | 16.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 20 | 2 | 6 | 26 | 58.3333 | 73.3333 | 58.3333 | 57.7500 | 57.9153 | 15.0000 | 15.0000 | 5.0833 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 5 | 2 | 6 | 11 | 58.3333 | 73.3333 | 61.6667 | 62.0000 | 61.7947 | 15.0000 | 11.6667 | 2.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 0 | 2 | 6 | 6 | 52.9167 | 75.4167 | 75.4167 | N/A | N/A | 22.5000 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 10 | 2 | 6 | 16 | 52.9167 | 75.4167 | 66.2500 | 59.0000 | 62.3543 | 22.5000 | 9.1667 | 7.2500 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 2 | 2 | 6 | 8 | 52.9167 | 75.4167 | 70.4167 | 57.5000 | 63.1709 | 22.5000 | 5.0000 | 12.9167 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 20 | 2 | 6 | 26 | 52.9167 | 75.4167 | 64.1667 | 59.5000 | 61.7449 | 22.5000 | 11.2500 | 4.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 5 | 2 | 6 | 11 | 52.9167 | 75.4167 | 66.2500 | 58.5000 | 62.0736 | 22.5000 | 9.1667 | 7.7500 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 0 | 2 | 6 | 6 | 55.0000 | 58.3333 | 58.3333 | N/A | N/A | 3.3333 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 10 | 2 | 6 | 16 | 55.0000 | 58.3333 | 45.0000 | 58.0000 | 49.1993 | 3.3333 | 13.3333 | 13.0000 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 2 | 2 | 6 | 8 | 55.0000 | 58.3333 | 51.6667 | 60.0000 | 54.7368 | 3.3333 | 6.6667 | 8.3333 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 20 | 2 | 6 | 26 | 55.0000 | 58.3333 | 43.3333 | 54.0000 | 47.1645 | 3.3333 | 15.0000 | 10.6667 |
| D92-ConditionalJointLocalRidge-v1 | oof | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 5 | 2 | 6 | 11 | 55.0000 | 58.3333 | 48.3333 | 56.0000 | 51.8808 | 3.3333 | 10.0000 | 7.6667 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 0 | 2 | 6 | 6 | 70.8333 | 58.1481 | 58.1481 | N/A | N/A | -12.6852 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 10 | 2 | 6 | 16 | 70.8333 | 58.1481 | 45.3704 | 36.8889 | 39.2261 | -12.6852 | 12.7778 | 15.2963 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 2 | 2 | 6 | 8 | 70.8333 | 58.1481 | 51.2963 | 35.2778 | 34.6850 | -12.6852 | 6.8519 | 35.8333 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 20 | 2 | 6 | 26 | 70.8333 | 58.1481 | 44.0741 | 32.1389 | 35.7974 | -12.6852 | 14.0741 | 17.3981 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 10 | 5 | 2 | 6 | 11 | 70.8333 | 58.1481 | 48.6111 | 34.8889 | 38.8483 | -12.6852 | 9.5370 | 21.2407 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 0 | 2 | 6 | 6 | 70.0000 | 57.3904 | 57.3904 | N/A | N/A | -12.6096 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 10 | 2 | 6 | 16 | 70.0000 | 57.3904 | 41.6228 | 34.8289 | 36.0878 | -12.6096 | 15.7675 | 14.9079 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 2 | 2 | 6 | 8 | 70.0000 | 57.3904 | 48.6842 | 37.4342 | 34.5119 | -12.6096 | 8.7061 | 35.1096 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 20 | 2 | 6 | 26 | 70.0000 | 57.3904 | 40.9868 | 32.0395 | 34.6397 | -12.6096 | 16.4035 | 14.5789 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 20 | 5 | 2 | 6 | 11 | 70.0000 | 57.3904 | 45.0658 | 35.9474 | 37.2674 | -12.6096 | 12.3246 | 19.5746 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 0 | 2 | 6 | 6 | 68.3333 | 54.5833 | 54.5833 | N/A | N/A | -13.7500 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 10 | 2 | 6 | 16 | 68.3333 | 54.5833 | 44.1667 | 39.0000 | 40.1239 | -13.7500 | 10.4167 | 12.5000 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 2 | 2 | 6 | 8 | 68.3333 | 54.5833 | 50.0000 | 46.2500 | 32.9250 | -13.7500 | 4.5833 | 52.0833 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 20 | 2 | 6 | 26 | 68.3333 | 54.5833 | 42.5000 | 33.1250 | 35.4550 | -13.7500 | 12.0833 | 17.2083 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx1-cvs-daot-rc4-s2026092701 | 5 | 5 | 2 | 6 | 11 | 68.3333 | 54.5833 | 46.2500 | 42.5000 | 41.5664 | -13.7500 | 8.3333 | 18.0833 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 0 | 2 | 6 | 6 | 54.1667 | 49.5370 | 49.5370 | N/A | N/A | -4.6296 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 10 | 2 | 6 | 16 | 54.1667 | 49.5370 | 33.5185 | 38.0000 | 33.2492 | -4.6296 | 16.0185 | 10.6296 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 2 | 2 | 6 | 8 | 54.1667 | 49.5370 | 37.5926 | 48.0556 | 33.8380 | -4.6296 | 11.9444 | 29.3519 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 20 | 2 | 6 | 26 | 54.1667 | 49.5370 | 30.2778 | 33.0000 | 29.4621 | -4.6296 | 19.2593 | 10.5556 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 10 | 5 | 2 | 6 | 11 | 54.1667 | 49.5370 | 34.5370 | 43.8889 | 36.2135 | -4.6296 | 15.0000 | 12.5741 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 0 | 2 | 6 | 6 | 50.4167 | 47.3465 | 47.3465 | N/A | N/A | -3.0702 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 10 | 2 | 6 | 16 | 50.4167 | 47.3465 | 32.6974 | 38.8158 | 34.0224 | -3.0702 | 14.6491 | 9.8289 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 2 | 2 | 6 | 8 | 50.4167 | 47.3465 | 38.3772 | 51.6447 | 39.8272 | -3.0702 | 8.9693 | 21.0746 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 20 | 2 | 6 | 26 | 50.4167 | 47.3465 | 30.2412 | 33.2434 | 30.6525 | -3.0702 | 17.1053 | 7.6820 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 20 | 5 | 2 | 6 | 11 | 50.4167 | 47.3465 | 34.9123 | 40.9211 | 36.0141 | -3.0702 | 12.4342 | 10.3947 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 0 | 2 | 6 | 6 | 53.3333 | 45.8333 | 45.8333 | N/A | N/A | -7.5000 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 10 | 2 | 6 | 16 | 53.3333 | 45.8333 | 32.0833 | 39.7500 | 33.8945 | -7.5000 | 13.7500 | 11.8333 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 2 | 2 | 6 | 8 | 53.3333 | 45.8333 | 36.6667 | 45.0000 | 37.1190 | -7.5000 | 9.1667 | 17.5000 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 20 | 2 | 6 | 26 | 53.3333 | 45.8333 | 27.0833 | 33.3750 | 28.6340 | -7.5000 | 18.7500 | 8.8750 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092701 | rx3-cvs-daot-rc4-s2026092701 | 5 | 5 | 2 | 6 | 11 | 53.3333 | 45.8333 | 32.0833 | 45.5000 | 36.0222 | -7.5000 | 13.7500 | 15.5833 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 0 | 2 | 6 | 6 | 74.1667 | 56.2963 | 56.2963 | N/A | N/A | -17.8704 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 10 | 2 | 6 | 16 | 74.1667 | 56.2963 | 44.2593 | 38.9444 | 40.0338 | -17.8704 | 12.0370 | 13.2037 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 2 | 2 | 6 | 8 | 74.1667 | 56.2963 | 49.7222 | 33.8889 | 32.2479 | -17.8704 | 6.5741 | 36.0185 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 20 | 2 | 6 | 26 | 74.1667 | 56.2963 | 42.8704 | 33.2500 | 35.9191 | -17.8704 | 13.4259 | 16.1019 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 10 | 5 | 2 | 6 | 11 | 74.1667 | 56.2963 | 47.6852 | 37.4444 | 39.9327 | -17.8704 | 8.6111 | 19.5000 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 0 | 2 | 6 | 6 | 72.5000 | 57.3904 | 57.3904 | N/A | N/A | -15.1096 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 10 | 2 | 6 | 16 | 72.5000 | 57.3904 | 43.2237 | 36.2368 | 37.5434 | -15.1096 | 14.1667 | 15.0570 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 2 | 2 | 6 | 8 | 72.5000 | 57.3904 | 48.9693 | 37.7632 | 34.7884 | -15.1096 | 8.4211 | 35.7237 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 20 | 2 | 6 | 26 | 72.5000 | 57.3904 | 41.6228 | 32.5526 | 35.0929 | -15.1096 | 15.7675 | 14.7456 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 20 | 5 | 2 | 6 | 11 | 72.5000 | 57.3904 | 45.7895 | 38.2895 | 38.8814 | -15.1096 | 11.6009 | 18.8158 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 0 | 2 | 6 | 6 | 71.6667 | 56.6667 | 56.6667 | N/A | N/A | -15.0000 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 10 | 2 | 6 | 16 | 71.6667 | 56.6667 | 45.0000 | 38.5000 | 39.9594 | -15.0000 | 11.6667 | 13.5000 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 2 | 2 | 6 | 8 | 71.6667 | 56.6667 | 51.2500 | 47.5000 | 33.3362 | -15.0000 | 5.4167 | 54.5833 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 20 | 2 | 6 | 26 | 71.6667 | 56.6667 | 42.9167 | 32.5000 | 34.9240 | -15.0000 | 13.7500 | 19.2500 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx1-cvs-daot-rc4-s2026092702 | 5 | 5 | 2 | 6 | 11 | 71.6667 | 56.6667 | 47.9167 | 37.0000 | 38.9615 | -15.0000 | 8.7500 | 21.9167 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 0 | 0 | 6 | 6 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 10 | 0 | 6 | 16 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 2 | 0 | 6 | 8 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 20 | 0 | 6 | 26 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 1 | 5 | 0 | 6 | 11 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 0 | 2 | 6 | 6 | 58.3333 | 48.1481 | 48.1481 | N/A | N/A | -10.1852 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 10 | 2 | 6 | 16 | 58.3333 | 48.1481 | 31.7593 | 38.7778 | 32.6882 | -10.1852 | 16.3889 | 11.2407 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 2 | 2 | 6 | 8 | 58.3333 | 48.1481 | 37.6852 | 45.8333 | 34.3471 | -10.1852 | 10.4630 | 25.1852 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 20 | 2 | 6 | 26 | 58.3333 | 48.1481 | 28.9815 | 34.0000 | 29.4437 | -10.1852 | 19.1667 | 10.0370 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 10 | 5 | 2 | 6 | 11 | 58.3333 | 48.1481 | 33.6111 | 42.2222 | 35.1136 | -10.1852 | 14.5370 | 12.3889 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 0 | 2 | 6 | 6 | 52.9167 | 47.7632 | 47.7632 | N/A | N/A | -5.1535 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 10 | 2 | 6 | 16 | 52.9167 | 47.7632 | 30.7675 | 39.6053 | 33.2393 | -5.1535 | 16.9956 | 10.1272 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 2 | 2 | 6 | 8 | 52.9167 | 47.7632 | 36.3596 | 51.3158 | 38.5409 | -5.1535 | 11.4035 | 20.7456 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 20 | 2 | 6 | 26 | 52.9167 | 47.7632 | 27.7193 | 33.3750 | 29.2756 | -5.1535 | 20.0439 | 7.7215 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 20 | 5 | 2 | 6 | 11 | 52.9167 | 47.7632 | 32.6096 | 42.4211 | 35.2198 | -5.1535 | 15.1535 | 11.4956 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 0 | 2 | 6 | 6 | 55.0000 | 46.6667 | 46.6667 | N/A | N/A | -8.3333 | 0.0000 | N/A |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 10 | 2 | 6 | 16 | 55.0000 | 46.6667 | 32.5000 | 38.2500 | 33.7896 | -8.3333 | 14.1667 | 9.4167 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 2 | 2 | 6 | 8 | 55.0000 | 46.6667 | 36.6667 | 52.5000 | 41.1866 | -8.3333 | 10.0000 | 17.5000 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 20 | 2 | 6 | 26 | 55.0000 | 46.6667 | 25.0000 | 35.0000 | 28.2637 | -8.3333 | 21.6667 | 10.0000 |
| D92-ConditionalJointLocalRidge-v1 | proxy | R_CONDITIONAL_seq | 2026092702 | rx3-cvs-daot-rc4-s2026092702 | 5 | 5 | 2 | 6 | 11 | 55.0000 | 46.6667 | 33.7500 | 48.0000 | 38.1218 | -8.3333 | 12.9167 | 15.4167 |

资源口径

唯一supervisor程序墙钟224.820874s，CPU进程峰值RSS为1243791360B；包含配对验证与日志。单条分类头评分计时不代表完整配对耗时。
原分类头每模型权重3840B，实际packet文件8737B/8735B。文件/数组字节不等于传输字节或部署常驻内存；实际星地传输、GPU峰值、能耗未测，为N/A。
| 方法 | 行 | 原始实测资源 |
|---|---|---|
| D92-AffineJointLocalRidge-v1 | affine-rx3-cvs-daot-rc4-s2026092701 | {"packet_load_seconds": 0.002801285998430103, "raw_cache_read_seconds": 0.17171347700059414, "float32_scalar_list_bridge_seconds": 0.027391060211812146, "native_single_record_score_seconds": 0.14244416494329926, "wall_seconds": 0.35795195199898444, "score_call_count": 1040, "scored_physical_count": 1040, "packet_total_local_file_bytes": 8737, "head_weight_file_bytes": 3840, "raw_z_id_numeric_bytes": 665600, "score_numeric_bytes": 24960, "incremental_transmission_bytes": null, "deployment_peak_memory_bytes": null, "gpu_peak_memory_bytes": null, "energy": null, "unmeasured_reason": "Local packet bytes and CPU timings measured; no network transfer, deployment peak or energy measured", "hardware": {"device": "cpu", "machine": "x86_64", "platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39", "torch_version": "2.1.0+cu121"}, "byte_scope": "Actual local packet/cache/prediction files and numeric arrays; not measured transmission or resident deployment memory"} |
| D92-AffineJointLocalRidge-v1 | affine-rx3-cvs-daot-rc4-s2026092702 | {"packet_load_seconds": 0.0008140549907693639, "raw_cache_read_seconds": 0.05154261499410495, "float32_scalar_list_bridge_seconds": 0.02728782783378847, "native_single_record_score_seconds": 0.14103048690594733, "wall_seconds": 0.23394189499958884, "score_call_count": 1040, "scored_physical_count": 1040, "packet_total_local_file_bytes": 8735, "head_weight_file_bytes": 3840, "raw_z_id_numeric_bytes": 665600, "score_numeric_bytes": 24960, "incremental_transmission_bytes": null, "deployment_peak_memory_bytes": null, "gpu_peak_memory_bytes": null, "energy": null, "unmeasured_reason": "Local packet bytes and CPU timings measured; no network transfer, deployment peak or energy measured", "hardware": {"device": "cpu", "machine": "x86_64", "platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39", "torch_version": "2.1.0+cu121"}, "byte_scope": "Actual local packet/cache/prediction files and numeric arrays; not measured transmission or resident deployment memory"} |
| D92-AffineJointLocalRidge-v1 | affine-rx1-cvs-daot-rc4-s2026092701 | {"packet_load_seconds": 0.0008052190096350387, "raw_cache_read_seconds": 0.01850890000059735, "float32_scalar_list_bridge_seconds": 0.027587452190346085, "native_single_record_score_seconds": 0.14269433497975115, "wall_seconds": 0.20327826699940488, "score_call_count": 1040, "scored_physical_count": 1040, "packet_total_local_file_bytes": 8737, "head_weight_file_bytes": 3840, "raw_z_id_numeric_bytes": 665600, "score_numeric_bytes": 24960, "incremental_transmission_bytes": null, "deployment_peak_memory_bytes": null, "gpu_peak_memory_bytes": null, "energy": null, "unmeasured_reason": "Local packet bytes and CPU timings measured; no network transfer, deployment peak or energy measured", "hardware": {"device": "cpu", "machine": "x86_64", "platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39", "torch_version": "2.1.0+cu121"}, "byte_scope": "Actual local packet/cache/prediction files and numeric arrays; not measured transmission or resident deployment memory"} |
| D92-AffineJointLocalRidge-v1 | affine-rx1-cvs-daot-rc4-s2026092702 | {"packet_load_seconds": 0.000829609009088017, "raw_cache_read_seconds": 0.018608960992423818, "float32_scalar_list_bridge_seconds": 0.027548866855795495, "native_single_record_score_seconds": 0.14300926381838508, "wall_seconds": 0.2037334940105211, "score_call_count": 1040, "scored_physical_count": 1040, "packet_total_local_file_bytes": 8735, "head_weight_file_bytes": 3840, "raw_z_id_numeric_bytes": 665600, "score_numeric_bytes": 24960, "incremental_transmission_bytes": null, "deployment_peak_memory_bytes": null, "gpu_peak_memory_bytes": null, "energy": null, "unmeasured_reason": "Local packet bytes and CPU timings measured; no network transfer, deployment peak or energy measured", "hardware": {"device": "cpu", "machine": "x86_64", "platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39", "torch_version": "2.1.0+cu121"}, "byte_scope": "Actual local packet/cache/prediction files and numeric arrays; not measured transmission or resident deployment memory"} |
| D92-ConditionalJointLocalRidge-v1 | conditional-rx3-cvs-daot-rc4-s2026092701 | {"packet_load_seconds": 0.0008288349927170202, "raw_cache_read_seconds": 0.05144345998996869, "float32_scalar_list_bridge_seconds": 0.026503090717596933, "native_single_record_score_seconds": 0.13834261006559245, "wall_seconds": 0.2299070809967816, "score_call_count": 1040, "scored_physical_count": 1040, "packet_total_local_file_bytes": 8737, "head_weight_file_bytes": 3840, "raw_z_id_numeric_bytes": 665600, "score_numeric_bytes": 24960, "incremental_transmission_bytes": null, "deployment_peak_memory_bytes": null, "gpu_peak_memory_bytes": null, "energy": null, "unmeasured_reason": "Local packet bytes and CPU timings measured; no network transfer, deployment peak or energy measured", "hardware": {"device": "cpu", "machine": "x86_64", "platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39", "torch_version": "2.1.0+cu121"}, "byte_scope": "Actual local packet/cache/prediction files and numeric arrays; not measured transmission or resident deployment memory"} |
| D92-ConditionalJointLocalRidge-v1 | conditional-rx3-cvs-daot-rc4-s2026092702 | {"packet_load_seconds": 0.0008092749922070652, "raw_cache_read_seconds": 0.05146239399618935, "float32_scalar_list_bridge_seconds": 0.027600326036917977, "native_single_record_score_seconds": 0.14109537182957865, "wall_seconds": 0.23410935400170274, "score_call_count": 1040, "scored_physical_count": 1040, "packet_total_local_file_bytes": 8735, "head_weight_file_bytes": 3840, "raw_z_id_numeric_bytes": 665600, "score_numeric_bytes": 24960, "incremental_transmission_bytes": null, "deployment_peak_memory_bytes": null, "gpu_peak_memory_bytes": null, "energy": null, "unmeasured_reason": "Local packet bytes and CPU timings measured; no network transfer, deployment peak or energy measured", "hardware": {"device": "cpu", "machine": "x86_64", "platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39", "torch_version": "2.1.0+cu121"}, "byte_scope": "Actual local packet/cache/prediction files and numeric arrays; not measured transmission or resident deployment memory"} |
| D92-ConditionalJointLocalRidge-v1 | conditional-rx1-cvs-daot-rc4-s2026092701 | {"packet_load_seconds": 0.0008188920037355274, "raw_cache_read_seconds": 0.018197686003986746, "float32_scalar_list_bridge_seconds": 0.026703388823079877, "native_single_record_score_seconds": 0.13859946072625462, "wall_seconds": 0.197234669001773, "score_call_count": 1040, "scored_physical_count": 1040, "packet_total_local_file_bytes": 8737, "head_weight_file_bytes": 3840, "raw_z_id_numeric_bytes": 665600, "score_numeric_bytes": 24960, "incremental_transmission_bytes": null, "deployment_peak_memory_bytes": null, "gpu_peak_memory_bytes": null, "energy": null, "unmeasured_reason": "Local packet bytes and CPU timings measured; no network transfer, deployment peak or energy measured", "hardware": {"device": "cpu", "machine": "x86_64", "platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39", "torch_version": "2.1.0+cu121"}, "byte_scope": "Actual local packet/cache/prediction files and numeric arrays; not measured transmission or resident deployment memory"} |
| D92-ConditionalJointLocalRidge-v1 | conditional-rx1-cvs-daot-rc4-s2026092702 | {"packet_load_seconds": 0.0008219099981943145, "raw_cache_read_seconds": 0.018569780004327185, "float32_scalar_list_bridge_seconds": 0.027150421155965887, "native_single_record_score_seconds": 0.13992719585075974, "wall_seconds": 0.19956754200393334, "score_call_count": 1040, "scored_physical_count": 1040, "packet_total_local_file_bytes": 8735, "head_weight_file_bytes": 3840, "raw_z_id_numeric_bytes": 665600, "score_numeric_bytes": 24960, "incremental_transmission_bytes": null, "deployment_peak_memory_bytes": null, "gpu_peak_memory_bytes": null, "energy": null, "unmeasured_reason": "Local packet bytes and CPU timings measured; no network transfer, deployment peak or energy measured", "hardware": {"device": "cpu", "machine": "x86_64", "platform": "Linux-6.8.0-142-generic-x86_64-with-glibc2.39", "torch_version": "2.1.0+cu121"}, "byte_scope": "Actual local packet/cache/prediction files and numeric arrays; not measured transmission or resident deployment memory"} |

来源：`E:\type10-7\automation_reports\CV-SincNet\20261001-phase2-d92-ground-a-support-m2-r01\results\raw_support_pairing`；实际runtime commit：`9b761828abc171b8e6437556aaae36935208e0cf`。
