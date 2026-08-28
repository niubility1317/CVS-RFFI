# D3+ERBT-IDR嵌套新类矩阵实验报告

- 状态：LOCAL_VERIFIED
- run ID：`stage2_sf_d3_erbt_idr_new1to20_rx20_1_s713101_m392002_20260828_r1`
- 科学声明：`DIAGNOSTIC_NON_FORMAL`；历史Stage2-C truth已用于旧研究，不用于正式晋级
- Git commit：`9236f747f19efd0cbd9fb67e2b5250ca9b8276aa`
- 环境/CWD：N607，`/home/szu2070436088/2510044040/CV-SincNet`，Conda环境`CVS-RFFI`
- checkpoint：`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`
- 数据：receiver=`20-1`，data seed=`713101`，method seed=`392002`，K=10，`p2_min_v1/VALIDATED_ONCE`
- 场景：`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`
- 注册规模：`N={1,2,3,5,10,15,20}`；6个旧类不变
- D3：327步单A段、4-fold support-only OOF温度、R16、`p1_head_norm`、cache关闭
- ERBT-IDR：`M29-FFT96-A4/D92-E0-NORF32`
- 四状态：`DA0_REG0`、`DA1_REG0`、`DA0_REG1`、`DA1_REG1`

## 矩阵与资源

D3每个场景只拟合一次，共3个support-only适配任务；每个D3 delta复用于同场景7个注册规模，共21个四状态预测/评分格。数据构建器从同一权威new20接收IQ按`(class,rank)`稳定排序并切出嵌套K10 support；query使用每个已注册类别的全部20条记录，因此各N的query量依次为140、160、180、220、320、420、520。

GPU预分配：D3三场景使用GPU0/1/2；预测阶段按实时preflight在GPU0–7中每卡最多两个任务。不得触碰无关进程。

## 命令与路径

- 配置：`configs/stage2_sf_d3_erbt_idr_new1to20_rx20_1_s713101_m392002_20260828.json`
- 运行根：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_d3_erbt_idr_new1to20_rx20_1_s713101_m392002_20260828_r1`
- 日志根：运行根下`logs/`
- 预期artifact：3个`sf_tapft_delta_bundle.pt`、3个数据构建audit、21个`support_state_receipt.json`、21个`predictions.npz`、21个`prediction_receipt.json`、21个`score.json`
- D3命令：`python code/scripts/run_sf_erbt_four_state.py adapt --plan <plan> --scenario <scene> --output-root <out> --device cuda:<gpu>`
- 预测命令：`python code/scripts/run_sf_erbt_four_state.py predict ...`
- 评分命令：仅在prediction完整后运行`python code/scripts/run_sf_erbt_four_state.py score ...`

## 技术停止规则

只在协议/query泄漏、错误receiver/seed/K/scene/split、输出覆盖、错误checkout、进程归属不清、无prediction闭合、scorer连接错误或同一确定性预prediction异常至少出现两次时停止相应run-owned进程树。低性能不得停止。所有已有artifact保留。

## 本地验证

- 聚焦协议/执行测试：66项通过。
- 新增D3/ERBT与嵌套构建测试：7项通过。
- 独立P0/P1审查：初审APPROVE；定点检查发现support交错排序风险，已用`(class,rank)`稳定排序与打乱输入测试修复。
- Git远端OID：`9236f747f19efd0cbd9fb67e2b5250ca9b8276aa`，与本地HEAD一致。

