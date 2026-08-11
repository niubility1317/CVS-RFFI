# P1-HNCCD冻结设计与实现追踪卡

状态：`IMPLEMENTED / LOCAL_TECHNICAL_VERIFIED / NO_PERFORMANCE_RESULT`。本卡仅承载P1-HNCCD在Phase1 source-L训练中的冻结实现合同和本地技术验证；它不构成性能、unknown、Phase2、Phase3或N607实验结论。尚未执行真实40E训练或任何N607实验。

## 冻结机制与边界

P1-HNCCD只在同一物理source-L批内，对单LEO`feat_joint`与其精确分类头的head/null residual实施线性交叉协方差去相关。运行时从sealed source-split receipt读取有序七个source receiver slot，local4类构成固定`7×4=28`个cell。每个cell样本数`n<2`时贡献可微零，所有cell始终按固定分母28聚合，不按active cell重归一。

令精确head权重为`W∈R^(4×d)`。它必须有限且full-row-rank。AMP外以可微FP32 Cholesky和triangular solve构造`Q=W^T L^{-T}`，其中`L=chol(WW^T)`；禁止`pinv`、epsilon、fallback或替代投影。对totalized-L2后的LEO特征`u`，定义`h=Q^Tu`和`b=u-Qh`。在每个`(receiver,class)`cell内，使用

```text
C=(H-Hbar)^T(B-Bbar)/n
L_HNCCD=(1/28)Σ||C||_F²
lambda_hnccd=.02
```

clean路径不参与该项；C臂辅助为`N/A/0`，G臂只在共同`L_base`上加`0.02L_HNCCD`。辅助项只读取source-known-train L。U不iterate、不forward；V、proxy、held、target、day、fold和score零训练、校准、选择或反馈。C/G保持同一warm-start、C/G物理顺序、seed、sampler、40E、新AdamW、AMP、单LEO和clear/low/rain三scene；资源峰值和step-time仅记录，不可用于反选。

## VJP、图释放与资源合同

对G中每个scene的首个正项批，raw、unscaled`L_HNCCD`只审计一次：LEO`z/feat_joint`、shared encoder和精确head`W`的VJP必须finite/nonzero；bias与clean VJP必须None或数值零。公共`L_base→W`路径保持live。实现只允许当前批的可微FP32布局`Q[d,4]`、`h[B,4]`、`b[B,d]`、`counts[28]`、`sum_h[28,4]`、`sum_b[28,d]`和`sum_hb[28,4,d]`，或等价逐cell`[4,d]`串流；禁止detach、`B×d²`、`B×28×d²`和跨批cache。

raw VJP可保留图，之后只能执行一次正常AMP backward、unscale、step/update及纯标量telemetry。任何finite或AMP-skip批都必须在下一forward前显式清空output、loss、VJP和log tensor的图根别名；不允许第二次backward、unscale、forward、`gc`或`empty_cache`。receipt、terminal、failure、warm-start和C/G共同绑定必须完整。sealed42、F6和后冻结门不在本实现面扩展。

## 可追溯性矩阵

|ID|来源|要求|目标文件|状态|验证|备注|
|---|---|---|---|---|---|---|
|HNCCD-01|冻结合同|固定`lambda=.02`、7×local4、B128、d160、固定28分母与三scene|`code/cvsrffi/phase1_hnccd.py`|implemented|纯函数与29项HNCCD本地测试|不重标定|
|HNCCD-02|冻结合同|exact`W[4,d]`有限/full-row-rank；FP32 Cholesky+triangular solve`Q=W^TL^-T`|`code/cvsrffi/phase1_hnccd.py`|implemented|纯VJP正例、singular-W拒绝和模块测试|禁`pinv`、epsilon、fallback|
|HNCCD-03|冻结合同|totalized-L2、`h/b/C`公式、`n<2`可微零、无active-cell重归一|`code/cvsrffi/phase1_hnccd.py`|implemented|纯函数/VJP测试|无clean辅助路径|
|HNCCD-04|权限与资源|source-L-only、同物理单LEO、current-batch FP32聚合、无跨批cache|`code/SSDG/train_ssdg.py`|implemented|训练入口静态锚点与模块测试；未运行真实数据训练|不改数据协议|
|HNCCD-05|C/G公平|CLI、strict warm-start、head/source receiver/physical-order/scene/AdamW/AMP绑定|`code/cvsrffi/phase1_hnccd.py`、`code/SSDG/train_ssdg.py`|implemented|`--help`、receipt合成和模块测试|C为N/A/0|
|HNCCD-06|VJP合同|每scene首个正项raw-unscaled VJP；LEO/encoder/W非零，clean/bias零|`code/cvsrffi/phase1_hnccd.py`、`code/SSDG/train_ssdg.py`|implemented|最小纯VJP与three-scene receipt tamper测试|公共`L_base→W`live绑定在common forward验证|
|HNCCD-07|AMP/图释放|一次正常backward/unscale/step；finite与skip批显式释放全部图根|`code/cvsrffi/phase1_hnccd.py`、`code/SSDG/train_ssdg.py`|implemented|AMP skip/raw-failure与无GC图根释放测试|禁第二次backward/forward/gc/cache清理|
|HNCCD-08|receipt终态|data-free receipt、failure、terminal、three-scene coverage与C/G共同绑定|`code/cvsrffi/phase1_hnccd.py`、`code/SSDG/train_ssdg.py`|implemented|terminal fail-closed、failure receipt和训练后terminal静态锚点|不扩展sealed42/F6|
|HNCCD-08a|terminal资源观测闭合|`hnccd_resource_observations`必须为严格list，逐common batch恰有一项；峰值显存为非负strict int、步时有限非负、无选择反馈|`code/cvsrffi/phase1_hnccd.py`|implemented|空、少一项、布尔/负峰值、非有限/负步时及反馈篡改7类负测|C/G共用同一terminal receipt，禁止以资源观测反选|

当前追踪计数：已验证9（本地技术级），已实现9，延期0，拒绝0，阻塞0。focused测试29项、HNCCD/HSCF/RCMMC共享回归64项均通过；真实数据训练、性能分析与任何发布结论均未执行。
