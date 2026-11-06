# 遮挡先验源域同记录检索诊断

固定源域遮挡先验、固定随机初始化的同结构网络和原 DURA，比较 672 条已探索源 pilot 的 clean、受损输入和 DURA 输出。所有输入统一遮挡 112–143 采样点；网络使用最后隐藏层按时间的均值与标准差拼接并单位化，直接 IQ 对照使用同一遮挡后的单位化波形。按公共 RX 和源模拟场景分为 21 组，在同组 clean 波形中检索对应的同一物理记录。

仅使用源 IQ 配对和公共 RX/场景，未读 F、TX 标签、V 或目标，也未更新模型。随机网络并非先验训练时的确切初始权重。先验训练 ID 与 pilot 互斥，但 pilot 已探索且原 DURA 可能见过同源域。CPU 真实 checkpoint 8 条 smoke 通过，完整诊断 6.89 秒。

| 表征 | 输入 | 受损源记录数 | 同记录 top-1 | 匹配与其他记录平均余弦差 |
| --- | --- | ---: | ---: | ---: |
| masked_prior | received | 576 | 43.92% | 0.000279 |
| masked_prior | dura | 576 | 76.22% | 0.000376 |
| random_prior | received | 576 | 56.60% | 0.005356 |
| random_prior | dura | 576 | 81.08% | 0.006588 |
| raw_iq | received | 576 | 7.12% | 0.009464 |
| raw_iq | dura | 576 | 90.63% | 0.280177 |

此具体池化方式下，遮挡先验对 DURA 输出的同记录匹配低于随机网络和直接 IQ，匹配与其他记录之间的平均余弦差也很小；不能因此断定先验所有层、所有表征或恢复用途均无效。clean 表征协方差有效秩：遮挡先验 9.283，随机网络 7.530。

同记录检索可能利用报文、波形内容或接收机线索，不能作为 TX 身份准确率、星地识别性能或统计显著性的证据。已登记恢复候选继续按固定训练和完整七景流程执行，不能因源域诊断取消正式测试。

## 逐组结果

| 表征／RX／源场景／输入 | 数量 | 同记录 top-1 | 同记录平均余弦 | 其他记录平均余弦 |
| --- | ---: | ---: | ---: | ---: |
| masked_prior/RX4/clean/received | 28 | 100.00% | 1.000000 | 0.999787 |
| masked_prior/RX4/clean/dura | 28 | 100.00% | 1.000000 | 0.999788 |
| random_prior/RX4/clean/received | 28 | 100.00% | 1.000000 | 0.997743 |
| random_prior/RX4/clean/dura | 28 | 100.00% | 1.000000 | 0.997748 |
| raw_iq/RX4/clean/received | 28 | 100.00% | 1.000000 | 0.886011 |
| raw_iq/RX4/clean/dura | 28 | 100.00% | 0.999991 | 0.886059 |
| masked_prior/RX4/practical_high/received | 30 | 43.33% | 0.999827 | 0.999543 |
| masked_prior/RX4/practical_high/dura | 30 | 83.33% | 0.999986 | 0.999685 |
| random_prior/RX4/practical_high/received | 30 | 40.00% | 0.999133 | 0.996939 |
| random_prior/RX4/practical_high/dura | 30 | 90.00% | 0.999949 | 0.997366 |
| raw_iq/RX4/practical_high/received | 30 | 6.67% | 0.005284 | 0.006843 |
| raw_iq/RX4/practical_high/dura | 30 | 100.00% | 0.996571 | 0.823685 |
| masked_prior/RX4/practical_high_urban/received | 18 | 38.89% | 0.999742 | 0.999628 |
| masked_prior/RX4/practical_high_urban/dura | 18 | 88.89% | 0.999994 | 0.999804 |
| random_prior/RX4/practical_high_urban/received | 18 | 50.00% | 0.998728 | 0.996748 |
| random_prior/RX4/practical_high_urban/dura | 18 | 100.00% | 0.999933 | 0.997676 |
| raw_iq/RX4/practical_high_urban/received | 18 | 5.56% | 0.007960 | 0.002065 |
| raw_iq/RX4/practical_high_urban/dura | 18 | 100.00% | 0.997259 | 0.883762 |
| masked_prior/RX4/practical_low_suburban/received | 27 | 37.04% | 0.999631 | 0.999494 |
| masked_prior/RX4/practical_low_suburban/dura | 27 | 62.96% | 0.999994 | 0.999795 |
| random_prior/RX4/practical_low_suburban/received | 27 | 29.63% | 0.998373 | 0.996489 |
| random_prior/RX4/practical_low_suburban/dura | 27 | 74.07% | 0.999936 | 0.997504 |
| raw_iq/RX4/practical_low_suburban/received | 27 | 7.41% | 0.007998 | 0.002965 |
| raw_iq/RX4/practical_low_suburban/dura | 27 | 96.30% | 0.996434 | 0.879093 |
| masked_prior/RX4/practical_low_urban/received | 24 | 16.67% | 0.997628 | 0.997592 |
| masked_prior/RX4/practical_low_urban/dura | 24 | 33.33% | 0.999976 | 0.999797 |
| random_prior/RX4/practical_low_urban/received | 24 | 16.67% | 0.993702 | 0.992878 |
| random_prior/RX4/practical_low_urban/dura | 24 | 41.67% | 0.999783 | 0.998109 |
| raw_iq/RX4/practical_low_urban/received | 24 | 8.33% | 0.012636 | 0.005749 |
| raw_iq/RX4/practical_low_urban/dura | 24 | 50.00% | 0.987934 | 0.892248 |
| masked_prior/RX4/practical_mid/received | 26 | 23.08% | 0.999674 | 0.999457 |
| masked_prior/RX4/practical_mid/dura | 26 | 53.85% | 0.999992 | 0.999741 |
| random_prior/RX4/practical_mid/received | 26 | 30.77% | 0.998548 | 0.996250 |
| random_prior/RX4/practical_mid/dura | 26 | 65.38% | 0.999925 | 0.997292 |
| raw_iq/RX4/practical_mid/received | 26 | 11.54% | 0.007000 | 0.004967 |
| raw_iq/RX4/practical_mid/dura | 26 | 92.31% | 0.995735 | 0.866780 |
| masked_prior/RX4/practical_mid_urban/received | 24 | 20.83% | 0.998566 | 0.998429 |
| masked_prior/RX4/practical_mid_urban/dura | 24 | 45.83% | 0.999982 | 0.999792 |
| random_prior/RX4/practical_mid_urban/received | 24 | 16.67% | 0.995063 | 0.993369 |
| random_prior/RX4/practical_mid_urban/dura | 24 | 58.33% | 0.999842 | 0.997753 |
| raw_iq/RX4/practical_mid_urban/received | 24 | 4.17% | 0.016993 | 0.014751 |
| raw_iq/RX4/practical_mid_urban/dura | 24 | 66.67% | 0.990120 | 0.874885 |
| masked_prior/RX6/clean/received | 20 | 100.00% | 1.000000 | 0.999700 |
| masked_prior/RX6/clean/dura | 20 | 100.00% | 1.000000 | 0.999703 |
| random_prior/RX6/clean/received | 20 | 100.00% | 1.000000 | 0.996691 |
| random_prior/RX6/clean/dura | 20 | 100.00% | 0.999999 | 0.996715 |
| raw_iq/RX6/clean/received | 20 | 100.00% | 1.000000 | 0.804637 |
| raw_iq/RX6/clean/dura | 20 | 100.00% | 0.999957 | 0.804716 |
| masked_prior/RX6/practical_high/received | 18 | 27.78% | 0.999806 | 0.999673 |
| masked_prior/RX6/practical_high/dura | 18 | 88.89% | 0.999996 | 0.999830 |
| random_prior/RX6/practical_high/received | 18 | 38.89% | 0.999105 | 0.996586 |
| random_prior/RX6/practical_high/dura | 18 | 94.44% | 0.999954 | 0.996914 |
| raw_iq/RX6/practical_high/received | 18 | 11.11% | 0.010624 | -0.012624 |
| raw_iq/RX6/practical_high/dura | 18 | 100.00% | 0.997917 | 0.847496 |
| masked_prior/RX6/practical_high_urban/received | 30 | 23.33% | 0.999564 | 0.999341 |
| masked_prior/RX6/practical_high_urban/dura | 30 | 76.67% | 0.999980 | 0.999723 |
| random_prior/RX6/practical_high_urban/received | 30 | 46.67% | 0.998268 | 0.994795 |
| random_prior/RX6/practical_high_urban/dura | 30 | 83.33% | 0.999816 | 0.995057 |
| raw_iq/RX6/practical_high_urban/received | 30 | 20.00% | 0.025539 | -0.009942 |
| raw_iq/RX6/practical_high_urban/dura | 30 | 93.33% | 0.993825 | 0.768434 |
| masked_prior/RX6/practical_low_suburban/received | 21 | 33.33% | 0.999669 | 0.999434 |
| masked_prior/RX6/practical_low_suburban/dura | 21 | 80.95% | 0.999991 | 0.999728 |
| random_prior/RX6/practical_low_suburban/received | 21 | 52.38% | 0.998575 | 0.995313 |
| random_prior/RX6/practical_low_suburban/dura | 21 | 90.48% | 0.999868 | 0.995648 |
| raw_iq/RX6/practical_low_suburban/received | 21 | 14.29% | 0.014421 | -0.009587 |
| raw_iq/RX6/practical_low_suburban/dura | 21 | 95.24% | 0.993940 | 0.794003 |
| masked_prior/RX6/practical_low_urban/received | 24 | 8.33% | 0.997293 | 0.997343 |
| masked_prior/RX6/practical_low_urban/dura | 24 | 20.83% | 0.999922 | 0.999765 |
| random_prior/RX6/practical_low_urban/received | 24 | 29.17% | 0.992941 | 0.990957 |
| random_prior/RX6/practical_low_urban/dura | 24 | 33.33% | 0.999428 | 0.996333 |
| raw_iq/RX6/practical_low_urban/received | 24 | 8.33% | 0.007064 | -0.001468 |
| raw_iq/RX6/practical_low_urban/dura | 24 | 37.50% | 0.970627 | 0.819996 |
| masked_prior/RX6/practical_mid/received | 22 | 54.55% | 0.999778 | 0.999509 |
| masked_prior/RX6/practical_mid/dura | 22 | 86.36% | 0.999992 | 0.999720 |
| random_prior/RX6/practical_mid/received | 22 | 68.18% | 0.998942 | 0.995065 |
| random_prior/RX6/practical_mid/dura | 22 | 90.91% | 0.999905 | 0.995183 |
| raw_iq/RX6/practical_mid/received | 22 | 13.64% | 0.030789 | -0.002977 |
| raw_iq/RX6/practical_mid/dura | 22 | 100.00% | 0.995859 | 0.757922 |
| masked_prior/RX6/practical_mid_urban/received | 24 | 25.00% | 0.998578 | 0.998412 |
| masked_prior/RX6/practical_mid_urban/dura | 24 | 50.00% | 0.999976 | 0.999712 |
| random_prior/RX6/practical_mid_urban/received | 24 | 41.67% | 0.994967 | 0.992323 |
| random_prior/RX6/practical_mid_urban/dura | 24 | 58.33% | 0.999717 | 0.995937 |
| raw_iq/RX6/practical_mid_urban/received | 24 | 8.33% | 0.001194 | -0.001589 |
| raw_iq/RX6/practical_mid_urban/dura | 24 | 58.33% | 0.984355 | 0.815848 |
| masked_prior/RX8/clean/received | 48 | 100.00% | 1.000000 | 0.999491 |
| masked_prior/RX8/clean/dura | 48 | 100.00% | 1.000000 | 0.999492 |
| random_prior/RX8/clean/received | 48 | 100.00% | 1.000000 | 0.989668 |
| random_prior/RX8/clean/dura | 48 | 100.00% | 1.000000 | 0.989672 |
| raw_iq/RX8/clean/received | 48 | 100.00% | 1.000000 | 0.562596 |
| raw_iq/RX8/clean/dura | 48 | 100.00% | 0.999984 | 0.562754 |
| masked_prior/RX8/practical_high/received | 48 | 64.58% | 0.999760 | 0.999326 |
| masked_prior/RX8/practical_high/dura | 48 | 97.92% | 0.999985 | 0.999487 |
| random_prior/RX8/practical_high/received | 48 | 87.50% | 0.998376 | 0.989376 |
| random_prior/RX8/practical_high/dura | 48 | 97.92% | 0.999744 | 0.989261 |
| raw_iq/RX8/practical_high/received | 48 | 0.00% | -0.054941 | -0.054279 |
| raw_iq/RX8/practical_high/dura | 48 | 100.00% | 0.991499 | 0.582666 |
| masked_prior/RX8/practical_high_urban/received | 48 | 68.75% | 0.999698 | 0.999280 |
| masked_prior/RX8/practical_high_urban/dura | 48 | 93.75% | 0.999981 | 0.999498 |
| random_prior/RX8/practical_high_urban/received | 48 | 89.58% | 0.998237 | 0.989247 |
| random_prior/RX8/practical_high_urban/dura | 48 | 97.92% | 0.999812 | 0.990005 |
| raw_iq/RX8/practical_high_urban/received | 48 | 4.17% | -0.031099 | -0.043302 |
| raw_iq/RX8/practical_high_urban/dura | 48 | 100.00% | 0.991808 | 0.572899 |
| masked_prior/RX8/practical_low_suburban/received | 48 | 60.42% | 0.999566 | 0.999078 |
| masked_prior/RX8/practical_low_suburban/dura | 48 | 95.83% | 0.999972 | 0.999254 |
| random_prior/RX8/practical_low_suburban/received | 48 | 87.50% | 0.998072 | 0.987530 |
| random_prior/RX8/practical_low_suburban/dura | 48 | 97.92% | 0.999764 | 0.987156 |
| raw_iq/RX8/practical_low_suburban/received | 48 | 0.00% | -0.041506 | -0.039290 |
| raw_iq/RX8/practical_low_suburban/dura | 48 | 100.00% | 0.986732 | 0.543908 |
| masked_prior/RX8/practical_low_urban/received | 48 | 39.58% | 0.998798 | 0.998500 |
| masked_prior/RX8/practical_low_urban/dura | 48 | 66.67% | 0.999915 | 0.999479 |
| random_prior/RX8/practical_low_urban/received | 48 | 43.75% | 0.994231 | 0.989193 |
| random_prior/RX8/practical_low_urban/dura | 48 | 64.58% | 0.998014 | 0.990857 |
| raw_iq/RX8/practical_low_urban/received | 48 | 6.25% | -0.009308 | -0.022581 |
| raw_iq/RX8/practical_low_urban/dura | 48 | 97.92% | 0.945150 | 0.573622 |
| masked_prior/RX8/practical_mid/received | 48 | 66.67% | 0.999739 | 0.999301 |
| masked_prior/RX8/practical_mid/dura | 48 | 95.83% | 0.999987 | 0.999494 |
| random_prior/RX8/practical_mid/received | 48 | 87.50% | 0.998558 | 0.990152 |
| random_prior/RX8/practical_mid/dura | 48 | 97.92% | 0.999772 | 0.990402 |
| raw_iq/RX8/practical_mid/received | 48 | 8.33% | -0.044516 | -0.048680 |
| raw_iq/RX8/practical_mid/dura | 48 | 100.00% | 0.991801 | 0.606890 |
| masked_prior/RX8/practical_mid_urban/received | 48 | 52.08% | 0.999029 | 0.998737 |
| masked_prior/RX8/practical_mid_urban/dura | 48 | 83.33% | 0.999963 | 0.999445 |
| random_prior/RX8/practical_mid_urban/received | 48 | 56.25% | 0.995369 | 0.987381 |
| random_prior/RX8/practical_mid_urban/dura | 48 | 81.25% | 0.999259 | 0.988322 |
| raw_iq/RX8/practical_mid_urban/received | 48 | 6.25% | -0.016702 | -0.031250 |
| raw_iq/RX8/practical_mid_urban/dura | 48 | 95.83% | 0.972034 | 0.582215 |

远端证据：N607 runs/20261107-phase1-masked-prior-representation-source-s2026092701-r01/result.json，执行 commit 16555cb8792b6c83da7d21ce08136d56dd194374。
