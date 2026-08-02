# D112-SEAM-qKNN source-held G1报告（r3）

状态：`PREREGISTERED / LOCAL_VERIFIED / NOT_RUN / NO_PERFORMANCE_RESULT`

- run ID：`d112_g1_sourceheld_seam_20260802_r3`。
- 目标／矩阵：固定`M0/M_HEAD_GROUND/M_JOINT_SEAM`，63行／189个prediction单元；完整prediction后独立score。
- 输入：source-held archive`f2ceae1b47f84027f21c561bd58f50cc9df5c511e4b8d110e04e8062db6bee41`；manifest`155d6ed4f75ec5f236da5169229d355a2cbfccadaec60c5ede61ed1e81235b94`；tap`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`；checkpoint`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- 代码：runner`f43f0532`＋字段合同修复`1b6a9711`＋只读收据修复`049c4927`；三臂理论／surface／head均不变。
- 验证：`ssr-gpu`编译通过，26项聚焦测试通过。r1/r2的prediction row、manifest、truth-open和score均为0，不含性能数据。
- 输出：`E:\type10-7\automation_reports\CV-SincNet\d112_g1_sourceheld_seam_20260802_r3\artifacts`，全新不可覆盖；本地NumPy，不使用N607/GPU。
- 裁决：完整同row old BA、seen-new、H、old floor和negative tail；负收益关闭D112，不调参、不补seed、不跑125。
- 本次为第二轮发布修复后的最后一次当前runner尝试；若技术上仍不闭合，冻结更小独立入口，不再修补此runner。
