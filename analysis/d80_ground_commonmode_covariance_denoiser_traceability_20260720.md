# D80地面公共域模态协方差去噪追溯

## 方法定位

D77证明地面对角可靠性预条件过弱，D78/D79证明把类相关地面域切向直接写入logit会保护旧类但压制新类。D80不再使用ground类中心作anchor，也不学习class-row residual；它只从完整地面域网格提取跨类平均的公共接收机扰动子空间。

对完整域`d`和ground类`c`的解量化中心`p_dc`，先做逐类域中心化并跨类平均：

`u_d=mean_c[p_dc−mean_d'(p_d'c)]`。

对`{u_d}`做SVD得到固定正交基`U`和投影`P=UU^T`。`P`只表示各ground类共同出现的接收机域扰动，不保留具体类中心或类分数。把其解释为单位噪声协方差增量`Sigma_g=I+P`，则精确逆为`Sigma_g^−1=I−0.5P`。对D62最终仿射头和全注册类等K target support均值`mu`做一次闭式编译：

`W'=W(I−0.5P)`，

`b'=b+0.5WPmu`，

所以`s'(x)=s(x)−0.5WP(x−mu)`，在target中心残差严格为0。全部注册类使用同一`P`、同一闭式系数0.5，不读取old/new角色或类ID，不训练、不扫描。

## 需求到实现追溯

|ID|要求|目标文件|状态|验证/停止条件|
|---|---|---|---|---|
|D80-R1|真实84-cell地面组件只读，入口/出口hash一致|D80 probe/loader|planned|完整域数、cell数、manifest/NPZ hash闭包|
|D80-R2|只提取跨类平均公共域模态，不保留类中心预测分支|D80 core|planned|类置换不变、公共模态构造等价、ground score access=false|
|D80-R3|固定`Sigma_g=I+P`及精确逆`I−0.5P`，无超参数|D80 core/tests|planned|投影幂等、特征值0.5/1、无扫描|
|D80-R4|中心保持编译，support均值处残差严格0|D80 core/tests|planned|直接式与单仿射等价、平移不变|
|D80-R5|所有target-old/new类同一公式，K1可定义|D80 core/tests|planned|类置换等变、无role/class branch、K1有限确定|
|D80-R6|正式资源上限，query额外MAC/state0|probe/resource audit|planned|0 optimizer step、闭式、含ground<256KB、单仿射|
|D80-R7|完整开发实验20-1/new5/K10/713101、3场景×5fold、105行|run/summarizer|planned|逐类/场景/混淆/INT8-FP32/资源全量解析|
|D80-R8|相对D62严格联合门|summarizer/report|planned|`A/N/H/min-A/min-N`不退化、`F`不升、至少一项严格改善且无混淆交换|
|D80-R9|formal ground bundle需联合封存及外部authority签名|loader/report|blocked|当前只能development diagnostic|

## 停止条件

不扫描投影系数、rank、中心倍率、类权重、场景权重或旧/新门。若公共模态去噪仍只保护旧类、伤害新类，或完全不改变outer决策，则关闭ground几何直接编译路线，转向重新训练Phase1 bundle时联合封存的类内协方差统计；当前未验证ground组件不得进入125或正式声明。
