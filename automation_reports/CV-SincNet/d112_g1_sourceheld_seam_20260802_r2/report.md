# D112-SEAM-qKNN source-held G1报告（r2）

状态：`PREREGISTERED / LOCAL_VERIFIED / NOT_RUN / NO_PERFORMANCE_RESULT`

- run ID：`d112_g1_sourceheld_seam_20260802_r2`。
- 目标：在固定新source-held split上完整运行`M0/M_HEAD_GROUND/M_JOINT_SEAM`三臂63行／189个prediction单元。
- 输入：archive SHA256=`f2ceae1b47f84027f21c561bd58f50cc9df5c511e4b8d110e04e8062db6bee41`；manifest SHA256=`155d6ed4f75ec5f236da5169229d355a2cbfccadaec60c5ede61ed1e81235b94`；D106 tap SHA256=`48b92fa8defc1c7261ca80f9e0723662e3fe6e8c64ec0881c8ef13bab3cafa2f`；checkpoint SHA256=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`。
- 代码：surface`d8805391`、理论`eb260250`、head`ca6db16d`、runner`f43f0532`、r1字段合同修复`1b6a9711`。
- 验证：`ssr-gpu`中编译和25项聚焦测试通过；r1故障前truth-open=0、score=0。
- 输出根：`E:\type10-7\automation_reports\CV-SincNet\d112_g1_sourceheld_seam_20260802_r2\artifacts`，全新且不可覆盖。
- 执行：重新从固定输入运行`prepare→predict`；确认63行／189单元后才允许`score`。predict无truth参数，本地纯NumPy，不使用N607/GPU。
- 裁决：完整同row报告old BA、seen-new、H、old floor和negative tail；联合稳定负收益则关闭D112，不调参、不增加seed、不跑125。
- 技术停止：仅输入／SHA／覆盖、非有限数、确定性异常或零prediction；禁止按性能停止。
