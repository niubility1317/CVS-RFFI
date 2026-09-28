# D42 support-only技术诊断

只读取两失败row的固定support和新训练模型v2 ground。禁止query/truth或目标准确率参与。逐行配置见experiment.json。原失败输出保留。

状态：ARTIFACTS_COMPLETE，远端support-only复现VERIFIED。诊断代码commit 281b92099b76233c1d4dc341372426894006bd84；预登记提交2aad44ec35b6b86d6a9ccd544ca65bfcd37d6291；主Agent独立P0/P1 PASS。

两CPU诊断进程PID为2056306、2056307，已退出。各自诊断JSON和失败inner-support矩阵独立读回，原实验及checkpoint未修改。压缩NPZ读取可能解压整成员；算法和诊断仅选择support行，未消费query行、truth或目标准确率。

|模型seed|首个失败split|外层K/inner K|FP64与sklearn不一致数|FP32不一致数|support判别间隔|FP32评分最大误差|
|---|---|---|---|---|---|---|
|392005|6533d11a97e532c6b8fe719b|20/19|0|1|2.0202e-5|5.2424e-5|
|2026092703|ef8921d662af2517554b00a8|20/19|0|1|1.3811e-5|4.1063e-5|

两次失败均在旧6类的support内部留一拟合。显式lstsq系数与sklearn系数最大差约1e-9，排除解算截断差异；大公共仿射项转FP32后掩盖小类别间隔。FP64去掉所有类别共享的系数/截距项再转FP32，两个捕获矩阵均恢复sklearn预测一致。

修复仅在原部署guard将失败时触发。保留FP64/sklearn一致、公共项消除代数等价及新FP32/sklearn一致三项校验，不能证明时继续报错；未禁用guard。原成功分支的数值状态bitwise不变，仅新增诊断audit字段。因此恢复可校验后复用原成功prediction，旧输出保持不动。

本地验证：3个新增回归PASS；原D42与matched相关31个测试PASS；两个捕获inner-support矩阵通过verify_d42_support_fix.py回放。尚需新发布在原运行环境验证完整失败support链，不能把本地回放称为恢复矩阵完成。

原始support诊断矩阵只保留在`E:/type10-7/local_artifacts/20260928-diagnostic-d42-support-manytx-m2-r01/`和远端同名run；不进入Git。小型diagnostic JSON见evidence。
