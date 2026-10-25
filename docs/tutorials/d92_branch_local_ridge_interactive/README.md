# D92-BranchLocalRidge-v1 交互教程

17 章、16 组交互算例、16 幅可放大矢量图解、10 道自测。从 Phase2 数据角色、向量与 Gram 开始，逐步推导五块特征、有限交互空间、稳定距离、径向核、训练中心化、trace 匹配、核岭回归与独立 query 判决。保留该版本历史编码器和 A/B/C 独立重拟合语义，不混入后续 margin、微调或 PAMF/DURA 新组合。

## 构建

```text
python build.py --export <绝对输出路径>
node verify.cjs
```

生成 `dist/index.html`。构建复用相邻 `pamf_dura_interactive` 的本地阅读器、样式、图解组件和 KaTeX 依赖，交付的 HTML 本身包含全部资源，无需上述目录即可离线打开。

数值例子采用缩小后的合成特征，不访问实际 IQ、support/query、checkpoint 或真值，不运行模型实验。完整版表达式和源码依据在各章记录。

验证覆盖公式保真、交互槽、交互核恒等式、近重复距离、二锚点解析解、中心化与 trace、Cholesky 残差、query 独立性、退化情况、物理分折与状态字节。页面排版、放大图解、滑块和符号表另在浏览器检查。
