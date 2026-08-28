# SF-TAPFT P1紧凑部署D0–D4工程回放实验报告

## 1.最小预登记

- run ID：`stage2_sf_tapft_p1_compact_deploy_replay_rx20_1_s392002_20260828_r1`
- 当前状态：`LOCAL_VERIFIED`
- 实现commit：`656af576ae86d408ff5a206712d81524edca765c`
- 方法边界：ADV3B02 CORE90；旧6类K=10，共60条support；不注册新类；仅报告`DA0_REG0`和`DA1_REG0`
- 数据句柄：`p2_min_v1/VALIDATED_ONCE`
- capsule：`d18-enrollment-before-rx20-1-seed713101-k10-smoke-reuse`
- split：`stage2b-rx20-1-seed713101-before-support-prefix`
- receiver/scene/seed：`rx20-1/leo_clear_weak/392002`
- 科学证据边界：该rx20 query truth已被历史实验使用，本轮只能验证实现、资源与同row回放，不得把结果作为“新未暴露capsule”晋级证据
- query边界：适配和选择不读取query、query truth或query role；prediction完整后才允许独立scorer连接既有Q180 truth

## 2.冻结矩阵

|行|方法|固定训练|cache|GPU|唯一目的|
|---|---|---|---|---:|---|
|D0|P0C H6 Compact|300/150/70|FP32 storage/compute，CompactH6Suffix|0|紧凑部署基线|
|D1|Q2A-Deploy|503/0/0|关闭|0|1248元素mixed norm候选|
|D2|Q2B-Deploy|231/0/0|关闭|1|1368元素校准优先候选|
|D3|R1-T|327/0/0+support-only OOF温度|关闭|1|1584元素性能档|
|D4|H6+class-CVaR|300/150/100，末30步Top2，权重0.03|FP32 storage/compute，CompactH6Suffix|2|floor增强候选|

所有行统一原位训练、delta v2 only、HardPair=0、无Adapter、无完整t3、无frequency/domain更新、无EMA。D1/D2/D3因真实训练范围超出仅`t3.norm`的H6缓存契约，显式关闭prefix cache。

## 3.本地验证与独立审查

- `ssr-gpu`环境115项聚焦测试通过；修改模块编译通过；`git diff --check`通过。
- 唯一一次独立P0/P1审查发现并修复两项启动P1：D1/D2/D3错误继承H6 cache；矩阵permission不符合runner既有枚举。定点复审结果为PASS。
- 当前RSS与process lifetime max RSS已分离；常驻模式增加CUDA free/min/allocated/reserved证据。真正冷启动尚未实现，接口会拒绝把同进程测量标记为cold-start。
- 低精度cache storage只允许一次性materialize为FP32 compute；未完成等价性验证的低精度suffix compute会被拒绝。

## 4.N607发布命令与路径

- 环境/CWD：N607 `ssr-gpu`；`/home/szu2070436088/2510044040/CV-SincNet`
- release根：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_sf_tapft_p1_compact_deploy_replay_rx20_1_s392002_20260828_r1`
- run根：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_sf_tapft_p1_compact_deploy_replay_rx20_1_s392002_20260828_r1`
- log根：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_sf_tapft_p1_compact_deploy_replay_rx20_1_s392002_20260828_r1`
- release归档：本地到远端只校验一次归档SHA，远端解压后执行一次Python编译
- 单行命令：`python code/scripts/run_sf_tapft_slim_matrix_row.py --matrix configs/stage2_sf_tapft_p1_compact_deploy_replay_rx20_1_s392002_20260828.json --row-id <D0|D1|D2|D3|D4> --mode deploy --deployment-inplace --delta-only --output-dir <run-root>/support/<row> --device cuda:0`
- GPU映射：D0/D1→GPU0，D2/D3→GPU1，D4→GPU2；每张卡最多两个实验，D0/D1和D2/D3按同卡顺序启动，D4独立
- smoke：launcher第一步只运行D0真实checkpoint无query smoke，PASS后立即继续其余行

## 5.停止规则与预期artifact

只在协议/query泄漏、错误stage/receiver/seed/K/scene/split、输出覆盖、错误checkout、重复确定性异常、无prediction闭合、scorer连接错误或进程归属不清时技术停止；不得因低性能停止。

每行预期生成：

- `support/<row>/selection.json`
- `support/<row>/sf_tapft_delta_bundle.pt`
- `support/<row>/deployment_receipt.json`
- GNU time与GPU采样日志
- 后续Q180 truth-blind prediction闭合后生成`prediction/<row>/da0_reg0.npz`、`da1_reg0.npz`和receipt

## 6.回放分析门槛

相对D0同row：BA不下降、floor不下降、最差类别准确率变化不低于-5pp、NLL不高于D0+0.02；可训练元素不超过1584、delta不超过10KB、适配wall-clock不超过20秒。由于本轮truth已历史暴露，达到门槛只记为`ENGINEERING_REPLAY_PASS`，不得晋级；正式晋级必须等待新的未暴露合法capsule。

