# D80地面跨域质心漂移协方差追溯

## 方法定位与预运行修正

D77的可逆地面对角坐标变换几乎被LDA抵消；D78/D79在D62后直接写入ground切向class-row residual，保护旧类但伤害新类。D80必须改变共享协方差本身，并在D62全部full/block、outer/held-support fit内一致生效，不能再做post-hoc row修补。

当前D22 v1 bundle没有radius、count或域内样本散度，只有84个int8域×类质心及scale。因此D80的合法统计是“class-centered cross-domain centroid drift covariance”，不是类内样本协方差。对`g_dc=s_dc q_dc`：

`r_dc=g_dc−mean_d(g_dc)`，

`G=sum_dc(r_dc r_dc^T)/[C_g(D−1)] + mean(s_dc^2/12)I`。

ground类中心在形成`r_dc`后丢弃。每个target support fit以`lambda=(D_eff−1)/[(D_eff−1)+C(K−1)]`把trace-matched`G`和target covariance闭式收缩，再解`W=Sigma_post^−1 mu`。全部类使用同一公式；ground不输出类别分数、anchor或row residual。

## 需求到实现追溯

|ID|要求|目标文件|状态|验证/停止条件|
|---|---|---|---|---|
|D80-R1|真实84-cell地面组件只读，入口/出口hash一致|D80 probe/D66 loader|verified|真实loader烟测84 cell；完整run核对入口/出口hash|
|D80-R2|只构造逐类去中心的共享域质心漂移协方差，不声称radius/count|D80 core/probe|verified|置换不变测试；真实rank78/effective rank13.6446；radius/count=false|
|D80-R3|固定量化底`mean(scale²/12)`，不裸逆低于量化精度的方向|D80 core/tests|verified|真实floor`5.2414e−7`，SPD测试通过|
|D80-R4|固定自由度EB权重，所有full/block outer/held fit一致注入|D80 core/probe|verified|factory patch→build D62 closure→restore；synthetic after权重13/90|
|D80-R5|所有target-old/new类同一公式，K1不伪造target covariance|D80 core/tests|verified|类置换等变；K1 `nu_t=0/lambda=1`有限确定|
|D80-R6|0新增step/parameter、含ground<256KB、query额外MAC/state0|probe/resource audit|verified|源码和单测闭包；真实资源待run|
|D80-R7|完整开发实验20-1/new5/K10/713101、3场景×5fold、105行|run/summarizer|planned|逐类/场景/混淆/INT8-FP32/资源全量解析|
|D80-R8|相对D62严格联合门|summarizer/report|planned|总体及每场景无退化、三类混淆不增、INT8/FP32无翻转|
|D80-R9|formal ground bundle需联合封存及外部authority签名|loader/report|blocked|当前只能development diagnostic|

## 停止条件

不扫描`lambda`、rank、量化ridge、类权重或场景权重。若D80仍产生旧/新交换、support-held与outer错配、量化翻转或完全identity，则关闭ground协方差决策路线；不启第二seed、125或N607。专项10/10、D62/D78/D79/D80相邻34/34已通过，但测试和资源达标不替代性能成功。
