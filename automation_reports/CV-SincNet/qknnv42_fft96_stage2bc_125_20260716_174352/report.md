# qKNNV42+FFT96 Stage2-B/C 125-bundle实验报告

Git镜像。权威运行记录见`E:\type10-7\automation_reports\CV-SincNet\qknnv42_fft96_stage2bc_125_20260716_174352\report.md`；本文件与其同步维护。

当前状态：已完成方法与矩阵代码实现；相关完整回归145项通过。真实ADV3B02 runtime、strict plan及25/25个target cache已核验。同步前发现远端共享代码树存在并行改动，因此不覆盖共享文件；改用Git提交`ec3dc84`的独立源码快照和`plan_v3.json`在实验run root内执行。等待N607四状态bwrap+strace smoke和125-bundle启动。

关键口径：

- 5 receiver×seed713101–713105×K={1,2,5,10,20}=125个bundle；
- 每bundle含物理独立before、after-new5、after-new10、after-new20；
- 实际100个package、500个状态cell、1500个场景行；
- seed713101是development，本轮不声明5个独立confirmation seed；
- 0训练参数、0epoch、0step、默认1-view；
- 禁止source/clean/query Oracle/类别配额/query-query图；
- qKNN主特征为ADV3B02 z_id160，FFT96分数权重0.34。

证据哈希：

- 执行计划`plan_v3.json`：`0e43380fa2689febcb91d8be7b8e23a0eb010901c88001a2cc0890f6ce60a767`
- 源码快照：`b9b95253b55eee0ab6da76294353fb45e34fb1449ba270816ab69ae895a31cd0`
- strict plan：`0939d1ef8c837e96f95275febb0c76e31b2d375004dab56688ed79f5ebf83676`
- ADV3B02 identity runtime：`b2021ca1ac97848a8cfda353a4070530bfa41bc08a711f746f329bd2d8d870d9`
