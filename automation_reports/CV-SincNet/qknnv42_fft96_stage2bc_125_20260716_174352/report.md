# qKNNV42+FFT96 Stage2-B/C 125-bundle实验报告

Git镜像。权威运行记录见`E:\type10-7\automation_reports\CV-SincNet\qknnv42_fft96_stage2bc_125_20260716_174352\report.md`；本文件与其同步维护。

当前状态：已完成方法与矩阵代码实现；相关完整回归145项通过。真实ADV3B02 runtime、strict plan及25/25个target cache已核验；`plan_v2.json`计数为100/125/500/1500且全部执行路径为绝对POSIX路径。运行实现已提交为`b04ee15`，等待N607四状态bwrap+strace smoke和125-bundle启动。

关键口径：

- 5 receiver×seed713101–713105×K={1,2,5,10,20}=125个bundle；
- 每bundle含物理独立before、after-new5、after-new10、after-new20；
- 实际100个package、500个状态cell、1500个场景行；
- seed713101是development，本轮不声明5个独立confirmation seed；
- 0训练参数、0epoch、0step、默认1-view；
- 禁止source/clean/query Oracle/类别配额/query-query图；
- qKNN主特征为ADV3B02 z_id160，FFT96分数权重0.34。

证据哈希：

- `plan_v2.json`：`a24f25657d0c96b3a3b0321e87f3c46a5debd64fb87248c53f463744709cba80`
- strict plan：`0939d1ef8c837e96f95275febb0c76e31b2d375004dab56688ed79f5ebf83676`
- ADV3B02 identity runtime：`b2021ca1ac97848a8cfda353a4070530bfa41bc08a711f746f329bd2d8d870d9`
