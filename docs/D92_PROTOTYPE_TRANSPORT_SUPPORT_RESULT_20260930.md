# BranchLocalRidge 与原型条件微调联合：完整 support 结果

**本文件是 support-only 诊断，不是 query 成绩或新增独立数据验收。** 四行、160 parent 全部完成后，独立汇总核验了物理样本、继承状态和实际优化轨迹。**本轮联合主候选只获得小幅收益，未实现全面改善，保留原 BranchLocalRidge 主线，不启动本候选 query 晋级评分。** 相对 R0，C 旧类提高 0.069 个百分点，新类提高 0.414 个百分点，H 提高 0.163 个百分点；但 B 旧类下降 0.208 个百分点。 不依据部分结果选取子集。

run：`20260930-phase2-d92-prototype-transport-support-m2-r02`；runtime：`9ebdcd7dd7ecd804c440111d3a2f40d3f6685d27`；analysis：`7a66ab6e65f0f25f2805bd6dcfdc9a8074254ac7`。技术失败 r01 及所有产物保留，新 r02 使用同方法、数据和参数独立恢复。

## 方法与三阶段口径

最终分类器保持 BranchLocalRidge。在冻结 Phase1 表征上，使用合法目标 support 原型训练 10 个参数（9 个约束自由度）：5 个切向旋转系数与 5 个零和注意力系数。原始距离和适配距离各占 0.5。目标为内部 support 物理样本损失按类池化后的 RMS，加按物理样本数归一化的近端项。最多 4 次投影梯度迭代，每次最多 3 次 Armijo 试探，保留最后接受状态，无最好步选模。

B 仅使用旧 6 类 support。顺序 C 继承实际 B 参数、锚点与旧类原型，并用全部注册 support 更新后拟合全类 LocalRidge。reset 仅作机制对照，不覆盖顺序候选；新增 0 类时 C 复用 B。每个内层头与原型仅来自其内层训练样本。保留原距离、旧原型及参数近端锚点不等于固定旧类分数；全类分类头重新拟合和新增类竞争仍可能造成旧类下降。≤1 个百分点必须由配对测量确认，结构本身没有该保证。

固定 practical residual：route=residual、mode=post_sync、equalization=false、fs=25 MHz。源域样本、源域逐样本特征和 query 均未用于本轮拟合。数据复用经用户授权，新增独立验证暂缓；结果不能称为全新独立确认。

K 是每类物理 support 数。旧类 6 类，新增 0、2、5、10、20 类，因此实际完整注册 support 数为 (6+新增类数)×K。OOF 使用外层留出，每个物理样本只评分一次；K=5/10/20 的外层每类训练数为 3/4、6/7、13/14。proxy 每类只取一个锚点，先按 parent 平均；它不能代表真实 K=1 query。真实 K=1 没有可分离留出指标，记 N/A。

A 应为同批地面模型适应前的旧类指标；该 pilot 缺少可配对 A，因此 A 和 B−A 全部 N/A。B0 是原 LocalRidge 用同一旧类 support 拟合后的基线，不能充当 A。B 与 C 在同一外层旧类留出记录上配对，C 对全部 6+新增类统一竞争。

## 总体与完整矩阵

总体包含 K=5/10/20 且新增类大于 0 的 96 个 parent。准确率/H 为百分数，差值为百分点。H 与绝对新旧差先逐 parent 计算再平均；重复旧类 support 不能当独立样本证明显著性。

| 诊断 | 路径 | A旧 | B0旧 | B旧 | C旧 | C新 | H | B−B0 | 注册旧类下降 | 新旧绝对差 | ΔC旧 | ΔC新 | ΔH |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oof | R0 | N/A | 71.042 | 71.042 | 63.767 | 54.846 | 58.417 | 0.000 | 7.274 | 10.740 | 0.000 | 0.000 | 0.000 |
| oof | R_transport_seq | N/A | 71.042 | 70.833 | 63.837 | 55.260 | 58.579 | -0.208 | 6.997 | 11.080 | 0.069 | 0.414 | 0.163 |
| oof | R_transport_reset | N/A | 71.042 | 70.833 | 63.924 | 55.448 | 58.777 | -0.208 | 6.910 | 10.769 | 0.156 | 0.602 | 0.360 |
| proxy | R0 | N/A | 52.147 | 52.147 | 44.354 | 34.108 | 36.275 | 0.000 | 7.793 | 15.853 | 0.000 | 0.000 | 0.000 |
| proxy | R_transport_seq | N/A | 52.147 | 52.147 | 44.354 | 34.108 | 36.275 | 0.000 | 7.793 | 15.853 | 0.000 | 0.000 | 0.000 |
| proxy | R_transport_reset | N/A | 52.147 | 52.147 | 44.354 | 34.108 | 36.275 | 0.000 | 7.793 | 15.853 | 0.000 | 0.000 | 0.000 |

完整矩阵共 120 行：2 种诊断×3 路径×4 个 K×5 个新增类数。旧类数始终为 6，new0 的新类/H/新旧差为 N/A；保留负下降值，不截断旧类收益。

| 诊断 | 路径 | K | 旧类数 | 新增类数 | A旧 | B0旧 | B旧 | C旧 | C新 | H | B−B0 | 注册旧类下降 | 新旧绝对差 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oof | R0 | 1 | 6 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 6 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 6 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 6 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 1 | 6 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R0 | 5 | 6 | 0 | N/A | 63.333 | 63.333 | 63.333 | N/A | N/A | 0.000 | 0.000 | N/A |
| oof | R0 | 5 | 6 | 2 | N/A | 63.333 | 63.333 | 61.250 | 50.000 | 53.522 | 0.000 | 2.083 | 17.083 |
| oof | R0 | 5 | 6 | 5 | N/A | 63.333 | 63.333 | 54.583 | 50.000 | 51.918 | 0.000 | 8.750 | 7.083 |
| oof | R0 | 5 | 6 | 10 | N/A | 63.333 | 63.333 | 53.333 | 48.250 | 50.361 | 0.000 | 10.000 | 8.083 |
| oof | R0 | 5 | 6 | 20 | N/A | 63.333 | 63.333 | 48.750 | 46.250 | 46.539 | 0.000 | 14.583 | 10.500 |
| oof | R0 | 10 | 6 | 0 | N/A | 73.542 | 73.542 | 73.542 | N/A | N/A | 0.000 | 0.000 | N/A |
| oof | R0 | 10 | 6 | 2 | N/A | 73.542 | 73.542 | 71.667 | 51.875 | 59.451 | 0.000 | 1.875 | 20.208 |
| oof | R0 | 10 | 6 | 5 | N/A | 73.542 | 73.542 | 67.917 | 56.500 | 61.409 | 0.000 | 5.625 | 11.833 |
| oof | R0 | 10 | 6 | 10 | N/A | 73.542 | 73.542 | 65.625 | 57.875 | 61.456 | 0.000 | 7.917 | 8.083 |
| oof | R0 | 10 | 6 | 20 | N/A | 73.542 | 73.542 | 64.167 | 56.500 | 60.024 | 0.000 | 9.375 | 7.750 |
| oof | R0 | 20 | 6 | 0 | N/A | 76.250 | 76.250 | 76.250 | N/A | N/A | 0.000 | 0.000 | N/A |
| oof | R0 | 20 | 6 | 2 | N/A | 76.250 | 76.250 | 74.688 | 53.125 | 61.590 | 0.000 | 1.562 | 21.562 |
| oof | R0 | 20 | 6 | 5 | N/A | 76.250 | 76.250 | 70.104 | 61.250 | 65.085 | 0.000 | 6.146 | 8.854 |
| oof | R0 | 20 | 6 | 10 | N/A | 76.250 | 76.250 | 68.021 | 63.625 | 65.703 | 0.000 | 8.229 | 4.396 |
| oof | R0 | 20 | 6 | 20 | N/A | 76.250 | 76.250 | 65.104 | 62.906 | 63.944 | 0.000 | 11.146 | 3.448 |
| oof | R_transport_seq | 1 | 6 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_transport_seq | 1 | 6 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_transport_seq | 1 | 6 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_transport_seq | 1 | 6 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_transport_seq | 1 | 6 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_transport_seq | 5 | 6 | 0 | N/A | 63.333 | 62.917 | 62.917 | N/A | N/A | -0.417 | 0.000 | N/A |
| oof | R_transport_seq | 5 | 6 | 2 | N/A | 63.333 | 62.917 | 60.833 | 52.500 | 53.915 | -0.417 | 2.083 | 20.000 |
| oof | R_transport_seq | 5 | 6 | 5 | N/A | 63.333 | 62.917 | 54.583 | 52.000 | 52.805 | -0.417 | 8.333 | 8.083 |
| oof | R_transport_seq | 5 | 6 | 10 | N/A | 63.333 | 62.917 | 52.500 | 48.750 | 50.374 | -0.417 | 10.417 | 5.917 |
| oof | R_transport_seq | 5 | 6 | 20 | N/A | 63.333 | 62.917 | 50.000 | 47.125 | 47.633 | -0.417 | 12.917 | 10.125 |
| oof | R_transport_seq | 10 | 6 | 0 | N/A | 73.542 | 73.125 | 73.125 | N/A | N/A | -0.417 | 0.000 | N/A |
| oof | R_transport_seq | 10 | 6 | 2 | N/A | 73.542 | 73.125 | 72.292 | 51.250 | 59.272 | -0.417 | 0.833 | 21.042 |
| oof | R_transport_seq | 10 | 6 | 5 | N/A | 73.542 | 73.125 | 68.125 | 55.250 | 60.663 | -0.417 | 5.000 | 13.292 |
| oof | R_transport_seq | 10 | 6 | 10 | N/A | 73.542 | 73.125 | 66.250 | 58.125 | 61.826 | -0.417 | 6.875 | 8.958 |
| oof | R_transport_seq | 10 | 6 | 20 | N/A | 73.542 | 73.125 | 64.792 | 56.812 | 60.441 | -0.417 | 8.333 | 8.437 |
| oof | R_transport_seq | 20 | 6 | 0 | N/A | 76.250 | 76.458 | 76.458 | N/A | N/A | 0.208 | 0.000 | N/A |
| oof | R_transport_seq | 20 | 6 | 2 | N/A | 76.250 | 76.458 | 74.062 | 53.125 | 61.358 | 0.208 | 2.396 | 20.938 |
| oof | R_transport_seq | 20 | 6 | 5 | N/A | 76.250 | 76.458 | 69.792 | 61.250 | 64.956 | 0.208 | 6.667 | 8.542 |
| oof | R_transport_seq | 20 | 6 | 10 | N/A | 76.250 | 76.458 | 67.604 | 63.625 | 65.504 | 0.208 | 8.854 | 4.437 |
| oof | R_transport_seq | 20 | 6 | 20 | N/A | 76.250 | 76.458 | 65.208 | 63.312 | 64.207 | 0.208 | 11.250 | 3.188 |
| oof | R_transport_reset | 1 | 6 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_transport_reset | 1 | 6 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_transport_reset | 1 | 6 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_transport_reset | 1 | 6 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_transport_reset | 1 | 6 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| oof | R_transport_reset | 5 | 6 | 0 | N/A | 63.333 | 62.917 | 62.917 | N/A | N/A | -0.417 | 0.000 | N/A |
| oof | R_transport_reset | 5 | 6 | 2 | N/A | 63.333 | 62.917 | 60.833 | 52.500 | 54.051 | -0.417 | 2.083 | 20.000 |
| oof | R_transport_reset | 5 | 6 | 5 | N/A | 63.333 | 62.917 | 54.583 | 52.500 | 53.144 | -0.417 | 8.333 | 7.750 |
| oof | R_transport_reset | 5 | 6 | 10 | N/A | 63.333 | 62.917 | 53.333 | 48.250 | 50.515 | -0.417 | 9.583 | 6.250 |
| oof | R_transport_reset | 5 | 6 | 20 | N/A | 63.333 | 62.917 | 50.000 | 47.250 | 47.859 | -0.417 | 12.917 | 8.917 |
| oof | R_transport_reset | 10 | 6 | 0 | N/A | 73.542 | 73.125 | 73.125 | N/A | N/A | -0.417 | 0.000 | N/A |
| oof | R_transport_reset | 10 | 6 | 2 | N/A | 73.542 | 73.125 | 72.083 | 51.875 | 59.611 | -0.417 | 1.042 | 20.208 |
| oof | R_transport_reset | 10 | 6 | 5 | N/A | 73.542 | 73.125 | 67.917 | 56.250 | 61.286 | -0.417 | 5.208 | 12.083 |
| oof | R_transport_reset | 10 | 6 | 10 | N/A | 73.542 | 73.125 | 65.833 | 58.625 | 61.954 | -0.417 | 7.292 | 8.042 |
| oof | R_transport_reset | 10 | 6 | 20 | N/A | 73.542 | 73.125 | 65.000 | 57.063 | 60.701 | -0.417 | 8.125 | 7.938 |
| oof | R_transport_reset | 20 | 6 | 0 | N/A | 76.250 | 76.458 | 76.458 | N/A | N/A | 0.208 | 0.000 | N/A |
| oof | R_transport_reset | 20 | 6 | 2 | N/A | 76.250 | 76.458 | 74.479 | 53.125 | 61.513 | 0.208 | 1.979 | 21.354 |
| oof | R_transport_reset | 20 | 6 | 5 | N/A | 76.250 | 76.458 | 69.792 | 60.625 | 64.576 | 0.208 | 6.667 | 9.167 |
| oof | R_transport_reset | 20 | 6 | 10 | N/A | 76.250 | 76.458 | 67.917 | 63.875 | 65.779 | 0.208 | 8.542 | 4.500 |
| oof | R_transport_reset | 20 | 6 | 20 | N/A | 76.250 | 76.458 | 65.312 | 63.438 | 64.331 | 0.208 | 11.146 | 3.021 |
| proxy | R0 | 1 | 6 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 6 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 6 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 6 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 1 | 6 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R0 | 5 | 6 | 0 | N/A | 50.938 | 50.938 | 50.938 | N/A | N/A | 0.000 | 0.000 | N/A |
| proxy | R0 | 5 | 6 | 2 | N/A | 50.938 | 50.938 | 49.271 | 33.125 | 33.343 | 0.000 | 1.667 | 27.187 |
| proxy | R0 | 5 | 6 | 5 | N/A | 50.938 | 50.938 | 45.729 | 39.625 | 41.070 | 0.000 | 5.208 | 13.437 |
| proxy | R0 | 5 | 6 | 10 | N/A | 50.938 | 50.938 | 42.292 | 37.438 | 38.783 | 0.000 | 8.646 | 10.312 |
| proxy | R0 | 5 | 6 | 20 | N/A | 50.938 | 50.938 | 38.438 | 33.281 | 34.149 | 0.000 | 12.500 | 12.281 |
| proxy | R0 | 10 | 6 | 0 | N/A | 53.032 | 53.032 | 53.032 | N/A | N/A | 0.000 | 0.000 | N/A |
| proxy | R0 | 10 | 6 | 2 | N/A | 53.032 | 53.032 | 51.019 | 25.556 | 29.944 | 0.000 | 2.014 | 29.537 |
| proxy | R0 | 10 | 6 | 5 | N/A | 53.032 | 53.032 | 47.199 | 36.361 | 39.869 | 0.000 | 5.833 | 14.199 |
| proxy | R0 | 10 | 6 | 10 | N/A | 53.032 | 53.032 | 43.773 | 36.931 | 39.069 | 0.000 | 9.259 | 11.741 |
| proxy | R0 | 10 | 6 | 20 | N/A | 53.032 | 53.032 | 39.699 | 32.465 | 34.548 | 0.000 | 13.333 | 13.285 |
| proxy | R0 | 20 | 6 | 0 | N/A | 52.473 | 52.473 | 52.473 | N/A | N/A | 0.000 | 0.000 | N/A |
| proxy | R0 | 20 | 6 | 2 | N/A | 52.473 | 52.473 | 49.644 | 28.980 | 33.078 | 0.000 | 2.829 | 25.323 |
| proxy | R0 | 20 | 6 | 5 | N/A | 52.473 | 52.473 | 45.482 | 37.053 | 39.598 | 0.000 | 6.990 | 12.357 |
| proxy | R0 | 20 | 6 | 10 | N/A | 52.473 | 52.473 | 41.634 | 36.089 | 37.686 | 0.000 | 10.839 | 10.398 |
| proxy | R0 | 20 | 6 | 20 | N/A | 52.473 | 52.473 | 38.070 | 32.388 | 34.167 | 0.000 | 14.402 | 10.174 |
| proxy | R_transport_seq | 1 | 6 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_transport_seq | 1 | 6 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_transport_seq | 1 | 6 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_transport_seq | 1 | 6 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_transport_seq | 1 | 6 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_transport_seq | 5 | 6 | 0 | N/A | 50.938 | 50.938 | 50.938 | N/A | N/A | 0.000 | 0.000 | N/A |
| proxy | R_transport_seq | 5 | 6 | 2 | N/A | 50.938 | 50.938 | 49.271 | 33.125 | 33.343 | 0.000 | 1.667 | 27.187 |
| proxy | R_transport_seq | 5 | 6 | 5 | N/A | 50.938 | 50.938 | 45.729 | 39.625 | 41.070 | 0.000 | 5.208 | 13.437 |
| proxy | R_transport_seq | 5 | 6 | 10 | N/A | 50.938 | 50.938 | 42.292 | 37.438 | 38.783 | 0.000 | 8.646 | 10.312 |
| proxy | R_transport_seq | 5 | 6 | 20 | N/A | 50.938 | 50.938 | 38.438 | 33.281 | 34.149 | 0.000 | 12.500 | 12.281 |
| proxy | R_transport_seq | 10 | 6 | 0 | N/A | 53.032 | 53.032 | 53.032 | N/A | N/A | 0.000 | 0.000 | N/A |
| proxy | R_transport_seq | 10 | 6 | 2 | N/A | 53.032 | 53.032 | 51.019 | 25.556 | 29.944 | 0.000 | 2.014 | 29.537 |
| proxy | R_transport_seq | 10 | 6 | 5 | N/A | 53.032 | 53.032 | 47.199 | 36.361 | 39.869 | 0.000 | 5.833 | 14.199 |
| proxy | R_transport_seq | 10 | 6 | 10 | N/A | 53.032 | 53.032 | 43.773 | 36.931 | 39.069 | 0.000 | 9.259 | 11.741 |
| proxy | R_transport_seq | 10 | 6 | 20 | N/A | 53.032 | 53.032 | 39.699 | 32.465 | 34.548 | 0.000 | 13.333 | 13.285 |
| proxy | R_transport_seq | 20 | 6 | 0 | N/A | 52.473 | 52.473 | 52.473 | N/A | N/A | 0.000 | 0.000 | N/A |
| proxy | R_transport_seq | 20 | 6 | 2 | N/A | 52.473 | 52.473 | 49.644 | 28.980 | 33.078 | 0.000 | 2.829 | 25.323 |
| proxy | R_transport_seq | 20 | 6 | 5 | N/A | 52.473 | 52.473 | 45.482 | 37.053 | 39.598 | 0.000 | 6.990 | 12.357 |
| proxy | R_transport_seq | 20 | 6 | 10 | N/A | 52.473 | 52.473 | 41.634 | 36.089 | 37.686 | 0.000 | 10.839 | 10.398 |
| proxy | R_transport_seq | 20 | 6 | 20 | N/A | 52.473 | 52.473 | 38.070 | 32.388 | 34.167 | 0.000 | 14.402 | 10.174 |
| proxy | R_transport_reset | 1 | 6 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_transport_reset | 1 | 6 | 2 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_transport_reset | 1 | 6 | 5 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_transport_reset | 1 | 6 | 10 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_transport_reset | 1 | 6 | 20 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| proxy | R_transport_reset | 5 | 6 | 0 | N/A | 50.938 | 50.938 | 50.938 | N/A | N/A | 0.000 | 0.000 | N/A |
| proxy | R_transport_reset | 5 | 6 | 2 | N/A | 50.938 | 50.938 | 49.271 | 33.125 | 33.343 | 0.000 | 1.667 | 27.187 |
| proxy | R_transport_reset | 5 | 6 | 5 | N/A | 50.938 | 50.938 | 45.729 | 39.625 | 41.070 | 0.000 | 5.208 | 13.437 |
| proxy | R_transport_reset | 5 | 6 | 10 | N/A | 50.938 | 50.938 | 42.292 | 37.438 | 38.783 | 0.000 | 8.646 | 10.312 |
| proxy | R_transport_reset | 5 | 6 | 20 | N/A | 50.938 | 50.938 | 38.438 | 33.281 | 34.149 | 0.000 | 12.500 | 12.281 |
| proxy | R_transport_reset | 10 | 6 | 0 | N/A | 53.032 | 53.032 | 53.032 | N/A | N/A | 0.000 | 0.000 | N/A |
| proxy | R_transport_reset | 10 | 6 | 2 | N/A | 53.032 | 53.032 | 51.019 | 25.556 | 29.944 | 0.000 | 2.014 | 29.537 |
| proxy | R_transport_reset | 10 | 6 | 5 | N/A | 53.032 | 53.032 | 47.199 | 36.361 | 39.869 | 0.000 | 5.833 | 14.199 |
| proxy | R_transport_reset | 10 | 6 | 10 | N/A | 53.032 | 53.032 | 43.773 | 36.931 | 39.069 | 0.000 | 9.259 | 11.741 |
| proxy | R_transport_reset | 10 | 6 | 20 | N/A | 53.032 | 53.032 | 39.699 | 32.465 | 34.548 | 0.000 | 13.333 | 13.285 |
| proxy | R_transport_reset | 20 | 6 | 0 | N/A | 52.473 | 52.473 | 52.473 | N/A | N/A | 0.000 | 0.000 | N/A |
| proxy | R_transport_reset | 20 | 6 | 2 | N/A | 52.473 | 52.473 | 49.644 | 28.980 | 33.078 | 0.000 | 2.829 | 25.323 |
| proxy | R_transport_reset | 20 | 6 | 5 | N/A | 52.473 | 52.473 | 45.482 | 37.053 | 39.598 | 0.000 | 6.990 | 12.357 |
| proxy | R_transport_reset | 20 | 6 | 10 | N/A | 52.473 | 52.473 | 41.634 | 36.089 | 37.686 | 0.000 | 10.839 | 10.398 |
| proxy | R_transport_reset | 20 | 6 | 20 | N/A | 52.473 | 52.473 | 38.070 | 32.388 | 34.167 | 0.000 | 14.402 | 10.174 |

## 完整资源与证据

10 个 float64 参数为 80 bytes，但不等于完整模型。原始/适配 support、标签、继承绑定、两个原型表与 LocalRidge 头均计入实测数值状态。单独星载硬件时延、部署总包和总星地传输没有测量，记 N/A；新增地面样本/统计 payload 为 0 bytes。CPU 结果不能冒称星载实测。

| 实测/登记资源字段 | 数值 |
|---|---:|
| additional_ground_data_payload_bytes | 0 |
| additional_ground_statistics_bytes | 0 |
| base_fit_effective_degrees_of_freedom_seconds_sum | 0.34285218504192017 |
| base_fit_fit_and_score_seconds_sum | 207.72631922898108 |
| base_fit_fit_seconds_sum | 50.752573107996795 |
| base_fit_gram_seconds_sum | 49.921627272005935 |
| base_fit_maximum_persistent_state_bytes | 2224800 |
| base_fit_score_seconds_sum | 153.5545406021174 |
| base_fit_solve_seconds_sum | 0.4864282479866233 |
| candidate_fit_baseline_binding_distance_evaluation_count_sum | 4576 |
| candidate_fit_baseline_binding_factorization_count_sum | 0 |
| candidate_fit_baseline_binding_seconds_sum | 83.61003705901567 |
| candidate_fit_fit_seconds_sum | 2365.790673078989 |
| candidate_fit_maximum_adapter_state_bytes | 80 |
| candidate_fit_maximum_head_state_bytes | 2224800 |
| candidate_fit_maximum_lineage_state_bytes | 4289376 |
| candidate_fit_maximum_optimizer_state_bytes | 0 |
| candidate_fit_maximum_persistent_state_bytes | 6820440 |
| candidate_fit_maximum_prototype_state_bytes | 306184 |
| candidate_fit_score_seconds_sum | 395.3144432999288 |
| deployment_package_bytes | N/A |
| final_head_forward_seconds_sum | 96.18687157102067 |
| incremental_transmission_bytes | N/A |
| inner_adjoint_seconds_sum | 927.0931101270016 |
| inner_forward_seconds_sum | 1146.9122725450761 |
| lane_wall_seconds_sum | 3606.411731518001 |
| maximum_lane_peak_rss_bytes | 693604352 |
| objective_backward_seconds_sum | 932.538933933949 |
| objective_forward_seconds_sum | 1154.11741956504 |
| parent_wall_seconds_sum | 3428.721644711999 |
| peak_gpu_memory_bytes | N/A |
| prototype_construction_seconds_sum | 45.969334370980505 |
| remote_code_release_archive_bytes | N/A |
| run_wall_seconds | 1839.6391611099243 |
| separate_inner_kernel_seconds | N/A |
| training_steps_seconds_sum | 2009.7692033930325 |
| transport_preparation_maximum_prepared_numeric_state_bytes | 36299776 |
| transport_preparation_maximum_transient_distance_bytes | 3179904 |
| transport_preparation_prepare_seconds_sum | 258.75422792997233 |

计数为整个三路径诊断矩阵的实际工作量，阶段耗时求和不等于并发墙钟；共享准备只计一次。具体线程、硬件、状态范围与未测项原因见 summary.json。

| 全矩阵实际计数 | 数值 |
|---|---:|
| accepted_trial_count | 3656 |
| backward_evaluation_count | 3728 |
| baseline_factorization_count | 3168 |
| baseline_head_fit_count | 3168 |
| derivative_triangular_solve_count | 22368 |
| diagnostic_fit_count | 0 |
| episodes | 160 |
| factorization_count | 19821 |
| final_factorization_count | 936 |
| final_head_fit_count | 936 |
| final_score_evaluation_count | 7744 |
| final_score_physical_count | 1020600 |
| head_fit_count | 19821 |
| inner_factorization_count | 15717 |
| inner_head_fit_count | 15717 |
| inner_objective_evaluation_count | 5239 |
| k1_episodes | 40 |
| oof_episodes | 120 |
| optimizer_iterations | 3728 |
| optimizer_steps | 3656 |
| prepared_distance_evaluation_count | 10224 |
| prototype_construction_count | 5112 |
| prototype_distance_evaluation_count | 15336 |
| proxy_anchor_count | 1400 |
| rejected_trial_count | 647 |
| sequence_paths | 1760 |
| trained_transport_stage_count | 936 |
| transport_forward_evaluation_count | 16653 |
| transport_preparation_count | 3168 |
| transport_stage_count | 4576 |
| trial_attempt_count | 4303 |
| trial_count | 4303 |

- [独立完整汇总](../automation_reports/CV-SincNet/20260930-phase2-d92-prototype-transport-support-m2-r02/results/support_summary/report.md)
- [机器可读结果与资源](../automation_reports/CV-SincNet/20260930-phase2-d92-prototype-transport-support-m2-r02/results/support_summary/summary.json)
- [接收机与场景分层](../automation_reports/CV-SincNet/20260930-phase2-d92-prototype-transport-support-m2-r02/results/support_summary/by_receiver_scene.csv)
- [模型与 cohort 分层](../automation_reports/CV-SincNet/20260930-phase2-d92-prototype-transport-support-m2-r02/results/support_summary/by_model_cohort.csv)
- [逐阶段优化目标](../automation_reports/CV-SincNet/20260930-phase2-d92-prototype-transport-support-m2-r02/results/support_summary/training_objectives.csv)
- [技术恢复与验证](D92_PROTOTYPE_TRANSPORT_RECOVERY_20260930.md)

## 结果解释与后续方向

顺序主候选 B=70.833%，C 旧=63.837%、新=55.260%、H=58.579%。注册旧类下降 6.997 个百分点，逐任务新旧绝对差 11.080 个百分点。A 缺失，不能验收适应提升≥10 个百分点；另外两个理想方向也没有全面达到。

| K | B旧 | C旧 | C新 | H | ΔB | ΔC旧 | ΔC新 | ΔH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 62.917 | 54.479 | 50.094 | 51.182 | -0.417 | 0.000 | 1.469 | 0.597 |
| 10 | 73.125 | 67.865 | 55.359 | 60.550 | -0.417 | 0.521 | -0.328 | -0.035 |
| 20 | 76.458 | 69.167 | 60.328 | 64.006 | 0.208 | -0.312 | 0.102 | -0.074 |

上表各 K 均等权平均新增 2、5、10、20 类。K 增大时绝对新类表现上升，但候选相对原方法的 H 收益没有单调上升：K=10 和 20 的平均 H 反而略低。更多 support 同时帮助原方法，微调增益不必随 K 单调增大；不同新增类数还会改变竞争强度。

K=10、新增 5 类时，新类下降 1.250 个百分点；K=20、新增 2/5/10 类时旧类分别下降 0.625/0.312/0.417 个百分点。K=20、新增 20 类时，旧类 65.208%、新类 63.312%，两组均值差为 1.896 个百分点，但逐任务绝对差平均为 3.188 个百分点，不能称为达到≤3 个百分点。

旧类竞争分解：顺序 B=70.833%，C 仅比较旧类列时为 71.406%，对全部注册类竞争后降为 63.837%。新增类竞争损失为 7.569 个百分点，比 R0 扩大 0.078 个百分点。总注册下降略小不能全部归因于保持旧类，因为 B 起点也下降了。

B 阶段相对 R0 的原正确→错误比例为 0.694%，原错误→正确为 0.486%；间隔均值有所改善，仍未转化为旧类准确率净增。C 新类中两项分别为 1.190% 和 1.604%。这些配对变化说明真实预测发生改变，不能将小收益解释为没有执行微调。

重置 C 是预登记机制对照：B 与顺序方法共用，C 旧=63.924%、新=55.448%、H=58.777%。它不能因本次 support 排名较高就被悄然改称原预声明的顺序主候选。完整三路径均保留。

本轮全部三路径 pilot 墙钟为 30.661 min，单 lane 峰值 RSS 为 661.473 MiB，最大完整数值状态为 6,820,440 bytes（包含 80 bytes adapter、306,184 bytes 原型、头与继承绑定）。新增地面数据/统计 payload 为 0 bytes。服务器 CPU 的试验耗时不等于星载实测时延。

相比之前通道联合 pilot，本轮头拟合从 29376 降到 19821，反向三角求解从 44928 降到 22368，实测整矩阵墙钟从约 42.596 min 降到 30.661 min；但峰值 RSS 从约 536.473 MiB 增到 661.473 MiB。两次运行没有构成受控星载速度测试，不能据此宣称单个部署方法的加速比。

原型条件旋转确实进行了监督优化，但 10 参数结构与 50% 原几何保护仍没有解决旧类适应和全类竞争。表达能力受限、训练间隔与外层 top-1 不完全一致是机制推断，不能把它们写成已证明原因。下一轮继续联合 LocalRidge 与真实 support 微调，优先增加可控表征能力与旧类保持约束，而非只加步数或扫描学习率。

- [完整训练轨迹与计数](../automation_reports/CV-SincNet/20260930-phase2-d92-prototype-transport-support-m2-r02/results/training_diagnostics/report.md)
- [全部实际阶段与变化](../automation_reports/CV-SincNet/20260930-phase2-d92-prototype-transport-support-m2-r02/results/training_diagnostics/stages.csv)
- [全部梯度、接受/拒绝试探与曲线](../automation_reports/CV-SincNet/20260930-phase2-d92-prototype-transport-support-m2-r02/results/training_diagnostics/stage_curves.csv)

- [完整训练损失、梯度、停止与去重解释](D92_PROTOTYPE_TRANSPORT_TRAINING_FINDINGS_20260930.md)

实际运维 release archive 为 583680 bytes，包含代码/配置，不属于地面样本或统计 payload；此值不能替代完整模型部署传输测量。

上述 6,820,440 bytes 是该 support pilot 的最大拟合数值状态。OOF 使用的实际每类训练量小于完整 K，不能将该数值当作完整 K=20 部署内存上限；模型总包与星载峰值仍未测量。

- [下一联合候选的冻结设计](D92_JOINT_AFTER_PROTOTYPE_DESIGN_20260930.md)（设计完成，尚未实施或运行，不代表收益）
