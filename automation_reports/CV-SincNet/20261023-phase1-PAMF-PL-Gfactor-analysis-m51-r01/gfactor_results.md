# PAMF普通PL与恢复数据权限：完整四格

17契约×3Fseed×4格=204读出。此次只补51个普通PL F×G-L预测、51独立评分和1CPU汇总；无F/G训练、新缓存或时延。F0使用已修复controls-binding来源。三F条件于固定同契约G-L/G-LU；native最终T1原生SSL不混入普通PL四格。

| F0 cell | 六景Accuracy（%） |
|---|---:|
| Sup+G-L | 76.8373±0.5315 |
| Sup+G-LU | 77.7774±0.2746 |
| PL+G-L | 76.7037±0.6620 |
| PL+G-LU | 77.6128±0.3688 |

| F0配对效应 | 六景差（pp） | 正seed数 |
|---|---:|---:|
| PL_minus_Sup_at_G_L | -0.1336±0.5513 | 1/3 |
| PL_minus_Sup_at_G_LU | -0.1646±0.1338 | 0/3 |
| G_LU_minus_L_at_Sup | 0.9401±0.3669 | 3/3 |
| G_LU_minus_L_at_PL | 0.9091±0.2949 | 3/3 |
| F_PL_x_G_U_interaction | -0.0310±0.4201 | 2/3 |

只Sup+G-L为完整L-only；PL+G-L仍经F用U。PL收益和G使用U收益因契约而异，不能从F0推出所有场景正交互。差分交互先seed内相减，再汇总三seed；SD不是显著性检验。全部逐RX/TX/day、Accuracy/Macro-F1及相对同F raw的净纠错保留。源F/G历史成本、共享缓存和本次分类成本分别保留。
