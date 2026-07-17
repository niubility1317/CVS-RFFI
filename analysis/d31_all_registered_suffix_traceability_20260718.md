# D31全注册类新后缀追溯表

D31是D27-D29强制回顾后的第2轮探索；D30为回顾后第1轮。本表在N607启动前建立，实验结果栏须由同一run的完整artifact回填。

|ID|来源要求|D31落地要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|D31-01|单IQ、LEO_weak-only|每个physical support只使用一个已经叠加的LEO_weak观测；z160、FFT96、RF32仅为该固定接收IQ的确定性拼接描述|runner、support audit|verified|统一60项测试通过；72状态压力测试通过|不重建IQ，不增加overlay，不从clean派生多view|
|D31-02|域适应与新类注册同等重要|Stage2-B用15步适应旧类几何；Stage2-C用全部已注册old+new support训练new suffix；同run保存注册前/后结果|D31 core、runner|implemented|候选锁和训练trace待runner回归|共享对角与旧类权重在Stage2-C冻结|
|D31-03|多模态拼接优先|使用辅助主导的`norm([z160,4*(FFT96||RF32)])`，保留更高维固定IQ多表征|runner|implemented|geometry artifact待实验回填|只使用同一物理IQ的确定性数学描述|
|D31-04|floor与遗忘保护|A为全注册类balanced CE；B加入top20%新类CVaR；C再加入旧类margin保护|D31 core|verified|D31+D26邻接测试23项通过|C的Stage2-C为15步，A/B为10步|
|D31-05|轻量快速适应|活动参数不超过2,016；A/B总步数15+10=25，C总步数15+15=30；无dense query图|D31 core、resource audit|implemented|资源artifact待实验回填|满足≤80k参数、≤30epoch、≤50step、≤256KB活动上限|
|D31-06|无query拟合与Oracle|query为测试集，适应/校准/选择/回滚不得读取query；逐样本在全部注册类上决策，无角色、真实batch类别数、配额或全局分配|D31 core、runner、receipt|implemented|接口与protocol测试待统一执行|support labels仅用于注册和support-only训练|
|D31-07|clean/source不可达|Phase2只读取密封LEO_weak support与用户授权的不可变Phase1 int8聚合模型知识|launcher、runner、runtime audit|implemented|启动前远端闭包与runtime evidence待核验|不得读取clean、source样本或未授权衍生信号|
|D31-08|K=1与统一K-shot|K=1执行质心注册且零梯度更新；开发K=10统一选参，后续必须覆盖K=1/5/10|D31 core、runner|verified|核心单测已覆盖K1旁路|本轮N607仍是K10 development support screen|
|D31-09|int8原型高效使用|当前support screen对历史84-cell组件只读；正式部署目标仅保留6×160固定medoid int8锚、scale和radius，约1.34KB|runner resource、bundle rebuild|deferred|需Phase1离线重封装后形式化|1.34KB是slim medoid目标口径，不冒充当前历史组件实测resident bytes|
|D31-10|完整实验与证据|6候选×3场景×5折=90行；输出逐类、逐receiver、floor、confusion、完整日志、资源、selection和receipt|runner、报告|verified|v2 90/90行、1,290个D31 trace、哈希/selection/resource闭环通过|正式5 receiver×≥5seed×3场景×2/5/10/20新类矩阵仍待正路线后执行|
|D31-11|Git与远端源闭包|本地先验证并提交；launcher校验runner、D31 core及继承依赖SHA，diag仅校验远端固定SHA而不上传|launcher、报告|verified|`bash -n`、最终runner/core SHA和`git diff --check`通过|N607同步与远端SHA仍待preflight后回填|

## 反向审计摘要

- 当前状态：verified 5项、implemented 5项、deferred 1项、rejected 0项、blocked 0项。
- 最高风险：v2证明D31-B仍依赖约-7logit事后bias，D31-C又以明显新类损失换旧类保护；09f8的new→old与new→wrong-new双重错误尚未解决。
- 设计一致性：D31核心机制是严格落地；slim medoid 1.34KB仍是正式bundle重建目标，当前support screen不会把它包装成已验证部署事实。
