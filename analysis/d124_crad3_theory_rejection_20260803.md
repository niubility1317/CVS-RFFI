# D124 CRAD-3理论拒绝

- 状态：`CLOSE_D124_THEORY_METRIC_REPACKAGING`；不实现、不发布G0/63/125。
- 候选原意：用6个old support-ground配对残差在D106 rank-3轴上的去均值离散度，构造参数无扫描的共享衰减metric，并统一作用于全部old/new support和query。
- 独立终审：`REJECT / P0=1 / P1=5 / P2=0`。

## 拒绝理由

1.同一非正交公共变换作用于support/query，等价于后端pullback PSD/角度metric；该族已被D93/D94/D110/C-id和D118边界关闭。更换为`q/(q+tau)`没有增加observable。
2.去均值残差`q`消除了共同receiver shift，剩余量混合类×域交互、ground误差和K-shot噪声；它不是receiver状态的可识别量。
3.K1的6类配对只使每轴方差数值可计算，并不能把receiver、anchor误差与单样本噪声分离。
4.`q`是support原型的跨类残差方差，`tau`是Phase1单样本类内scatter；虽同量纲，但K变化时统计尺度不匹配。
5.无clip时理论SPD但条件数无界；增加clip或floor会引入新的未识别超参数，不能作为局部修复。
6.old-only统计选择metric再作用于new类存在Stage2-C偏置；加入new scatter仍不能补足无ground残差，也会让append改变状态语义。

## 后续边界

不再研发support-local静态PSD、Fisher、对角衰减、白化或公共低秩metric换名路线。下一候选只有在Phase1新增经过receiver-held/class-LOCO约束学习并与checkpoint共同封存的轻型response dictionary/hyper-adapter时才可能获得新的可识别先验；Phase2仍只能由合法support估计少量系数，query零fit/零update。
