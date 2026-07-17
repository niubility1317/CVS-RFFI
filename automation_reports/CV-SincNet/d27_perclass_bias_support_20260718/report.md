# D27逐新类安全bias support-only实验

## 启动前记录

- experiment ID：`d27_perclass_bias_20260718/support_screen_v1`；日期：2026-07-18；operator：Codex；状态：`READY_FOR_N607_SUPPORT_SCREEN`。
- 目标：解决D26单一new-group bias在“旧类崩塌”和“新类崩塌”之间无可行点的问题，以每个新类1个独立安全bias同时优化旧类遗忘和新类floor。
- 比较：Z0、B3诊断、C0、D27-A `15+0`、D27-B `15+10`、D27-C `15+15`；6候选×3个LEO_weak场景×5个held-rank fold=90行。
- 数据与工作点：receiver `20-1`、开发seed `713101`、K=10、6个old+5个seen-new；直接复用D25/D26同一sealed enrollment-only LEO_weak support，不新增数据或信道view，query不打开。

## 方法锁

- 288D为同一唯一接收IQ的`z160+FFT96+RF32`拼接；每个物理support仍只有一个LEO_weak观测、一个support行，K不变。
- Stage2-B保持shared 288D diagonal+6个旧类weight，全批次15步；Stage2-C只训练新suffix 0/10/15步，旧weight、shared diagonal和旧raw score列冻结。
- 对每个新类`j`独立计算旧support安全上界：`b_j^safe=min_{i∈G_old}(winning_old_score_i-new_score_ij)-1e-4`。K>1只在`b_j^safe+[0,-0.5,-1,-2,-4]`内按固定类序做一次support LOO坐标选择；K=1直接使用cap，不伪造LOO。
- 选择目标词典序为`min_new_class_LOO→overall_LOO→worst_margin`；每个候选bias向量都必须保留所有Stage2-B old-only正确support行和逐旧类准确率，否则fail closed。
- 推理仍对全部注册类逐样本一次argmax，不读取query角色、类别数量、quota、排序或全局assignment。

## 本地版本与验证

- Git仓库：`E:/type10-7/github_publish/CVS-RFFI-repo`；根目录不是Git仓库，本报告根目录与Git镜像同步。
- D27核心提交：`67b9d2275782339e0ac07800652b997adbcca534`；runner提交：`00e89bb2`。
- runner SHA256：`9bb0deff5fa896da54947a7505eceb47e03a9d05d1a0b3d31490df36d0d9fd6b`；核心SHA256：`553d6361a728490c26963944df8353f1bc64bf1540b2ab6709f2f25bedd6f1ff`；launcher SHA256：`f67cbd548ff8d7c5082de1480da8e8c25976fc6e76214d9716474c5fee4b2f09`。
- 65项D27/D26/D25/C3相邻回归PASS；`py_compile`、`bash -n`、`git diff --check`PASS。
- 资源锁：正式档≤80,000活动参数、≤30epoch/step、≤256KB状态、无dense query图。D27 5类bias仅20B FP32预测状态且不参与梯度；20新类构造测试仍低于状态上限。

## N607计划

- 远端根：`/home/szu2070436088/2510044040/CV-SincNet`；Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；GPU优先0。
- output：`/home/szu2070436088/2510044040/CV-SincNet/runs/d27_perclass_bias_20260718/output/support_screen_v1`。
- log：`/home/szu2070436088/2510044040/CV-SincNet/logs/d27_perclass_bias_20260718/support_screen_v1.log`。
- 精确命令：`cd /home/szu2070436088/2510044040/CV-SincNet && D27_GPU=0 bash code/scripts/launch_d27_perclass_bias_support_20260718.sh`。
- 只同步runner、D27核心和launcher；不覆盖远端实际FFT96/RF32 operator。启动前重新做live inventory、远端SHA、`py_compile`、`bash -n`和output不存在门。
- 03:57 CST直连preflight PASS；8张RTX3090均0%利用、约10MiB显存。live inventory为`gpu_compute=[]`、`active_training_processes=[]`、`unknown_training_active=false`，允许使用GPU0。

## 判定与风险

- 晋级同时要求15fold旧support非退化、逐场景old/new pooled floor相对C0均提升≥10pp、任一旧/新类相对C0下降不超过10pp、H与forgetting不劣于C0；B3仅性能参考。
- 主要风险：独立安全cap仍可能对`cls_09f8/cls_f608`过严，或一次坐标搜索在LOO上乐观。若D27仍失败，依据失败分解再决定小幅support保护松弛或双原型，不继续盲扫全局bias/epoch。
- 当前仍是development support-only筛选，`formal_metric_claim_allowed=false`；正路线才进入joint bundle method lock及正式独立确认矩阵。
