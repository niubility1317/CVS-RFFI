# CVS 组会汇报 PPT 内容方案

| 页码 | 标题 | 核心内容 | 推荐图示 | 讲解要点 |
|---|---|---|---|---|
| 1 | CVS 项目版本迭代与成果汇报 | 项目名称、汇报人、日期 | 标题页 | 本次汇报聚焦版本演进、最终消融与结论 |
| 2 | 研究问题 | RFFI 发射机识别；跨 day / rx 泛化 | Source -> Target 示意图 | 从同分布识别转向跨采集条件识别 |
| 3 | 数据设置 | WiSig；train day 0,1；test day 2,3；train rx 0..6；test rx 7..11 | day/rx 矩阵 | 三类 target split 是最终评估核心 |
| 4 | 项目结构 | type1 到 type15；type10-* 分支 | 目录树时间轴 | 无 git，版本按目录和日志推断 |
| 5 | Baseline | SincConv + CE + 简单增强 | Baseline pipeline | `type1` 建立基本分类流程 |
| 6 | 早期提升 | SupCon + Prototype + DAC loss | Loss 堆叠图 | `type3` ORACLE best 98.65%，但不是严格跨域 |
| 7 | 训练框架演进 | argparse、AMP、rollback、自适应增强 | 训练循环图 | `type6/type7` 解决强增强稳定性 |
| 8 | 跨域转折 | WiSig、day/rx domain、PA、sat channel | 数据域切换图 | `type9` 开始真正跨采集条件 |
| 9 | Dual CVSincNet | id backbone + dom backbone + GRL | 双分支网络图 | `z_id` 做分类并去域，`z_dom` 学域信息 |
| 10 | 物理感知模块 | DAC/PA 特征、强度预测、joint fusion | 物理分支结构图 | 利用硬件非理想特征提升鲁棒性 |
| 11 | 最终训练策略 | S1 core、S2 stabilize、S3 selective late | Stage 时间轴 | 分阶段避免辅助 loss 过早破坏 ID 特征 |
| 12 | 评价指标 | last、best-joint、best-test、三类 split | 指标定义表 | 强调不能混淆 best-test 与模型选择 |
| 13 | 最终消融总览 | A/B/C/D 系列 | 消融表热力图 | 对比 MixStyle、分支删除、lite_c |
| 14 | 关键结果 | A00 vs D02 | 柱状图 | last: 86.51 -> 89.67；best-joint: 86.25 -> 90.18 |
| 15 | MixStyle 分析 | B00/B04/D02 提升；B02 后期崩溃 | 折线图 + warning 标记 | MixStyle 有用，但 random/多层可能不稳 |
| 16 | 分支消融 | no time/freq/stats/PA/DAC | 模块删除对比表 | time/freq/stats 都重要，PA 对稳定性有帮助 |
| 17 | 失败实验 | B02、C00、C03、type10-4 | 问题-原因-结论表 | 失败实验说明辅助约束需要分阶段和稳定保护 |
| 18 | 最终结论 | D02_litec_mixstyle 最佳 | 三点结论图 | 当前推荐版本：best-joint 90.18%，last 89.67% |
| 19 | 局限与后续 | 多 seed、多 split、稳定性、最难 split | Roadmap | 下一步做统计复现和稳定性控制 |
| 20 | Q&A | 导师可能问题 | Q&A 卡片 | 准备 source/target、loss、best/last 解释 |

## 推荐核心图表数据

| 实验 | Last | Best-joint | Best-test | 最难 split best |
|---|---:|---:|---:|---:|
| A00_s1_core_base | 86.51 | 86.25 | 89.51 | 86.72 |
| B00_mixstyle_cd_td_t1 | 88.03 | 88.08 | 89.85 | 85.37 |
| C01_no_dac | 88.17 | 88.27 | 90.04 | 85.61 |
| D00_mixstyle_no_dac | 88.84 | 88.84 | 90.14 | 85.04 |
| D02_litec_mixstyle | 89.67 | 90.18 | 90.53 | 86.25 |

## 备注

当前环境未安装 `python-pptx`，因此未直接生成 `.pptx`。本文件可直接作为 PPT 制作提纲；建议报告时以表格和结构图为主，少放代码细节。
