# 独立星地观测恢复器：实现与当前证据

已有可复用的独立恢复器，训练使用源 L/U 的 IQ 和模拟接收退化，不读取 TX 标签、不使用 F 损失。冻结原 F50 的完整正式测试已证明统计识别收益；逐条生成细节的 TX 真实性仍未证明，clean 也存在小幅损伤。目标保持 active，不能把本轮正收益写成普遍身份认证保证。

## 统一输入与输出

源域冻结主线为 R03 deterministic：

\[
\widehat x=b(r)+s\,C(r,b(r)),\quad s=0.1169459182403555.
\]

b 是冻结 R02 复数时频粗恢复器；C 是新学的多尺度残差修正器。推理只接当前受损 IQ，输出一份 IQ；原 PAMF 从这份 IQ 实际计算 FFT、PA、时间等分支及原默认头。没有独立预测一份与 IQ 不相容的 FFT。

```mermaid
flowchart LR
  R[当前受损 IQ] --> B[冻结复时频粗恢复]
  R --> C[确定性残差修正]
  B --> C
  B --> X[一份恢复 IQ]
  C --> X
  X --> F[原 PAMF 全部分支和默认头]
```

训练目标是对应原地面接收 IQ，而非理想 TX 波形。L/U 分别6300/56700条，标签均不进入恢复训练；父 E20＋新增 E20，累计 E40。源生成使用原生同潜在信道的 full/residual，15%clean/60%residual/25%full，E20固定4940更新。源 V不更新持久状态，原模型选择在目标前冻结。它属于具有 clean 源波形的模拟自监督，不是完全无 clean 数据训练。

## 六类网络的实际覆盖

|报告类别|实际实现和训练|当前证据|
|---|---|---|
|R1 残差1D U-Net|R02 residual/full/mixed三版，独立scratch E20|完成源比较，未作为新增目标候选|
|R2 复数时频U-Net|R02 tf_unet_mixed，正确负频率镜像、统一IQ输出|源波形A冻结；完整正式读出R04/R09|
|R3 物理展开|R06真实/估计作用诊断；R07学习作用＋6步软CG，新增E20|R07平均源NMSE增加1.5535%，SOURCE_NOT_QUALIFIED，0目标访问|
|R4 条件扩散|R02多尺度v-prediction，64档cosine噪声、16步DDIM|源128条×3draw类别切换56.25%，未通过身份稳定性|
|R5 条件流匹配|R02直线配对路径、16步Euler；不是I2SB|源类别切换51.5625%，未通过身份稳定性|
|R6 粗恢复＋残差生成|R03 diffusion/flow及deterministic控制，新增E20|生成版切换31.25%/7.03125%，均高于5%；源A/B冻结deterministic控制|

生成式负结果保留，没有按目标真值挑生成 draw、改seed或选择性重跑。R07物理版本使用15%clean/85%residual且只监督作用/噪声，与R03配方不同，不能把它们当成仅一个模块不同的严格匹配消融。

## 原 F50 的完整结果

固定原 F50 CROSS_RX-s2026092701、原默认头，168000条正式物理记录；七视图和三组预测全部固定后独立 truth-last。新增模型/头/适配更新为0，恢复 IQ 直接复用原 R04/R05。

|场景|原F50（%）|coarse（%）|residual（%）|
|---|---:|---:|---:|
|clean|84.6762|83.8321|84.1786|
|高仰角郊区|40.7327|77.7238|77.3673|
|中仰角郊区|39.9167|76.7458|76.3125|
|低仰角郊区|39.0298|75.6357|75.3315|
|高仰角城区|39.6982|76.4631|76.2244|
|中仰角城区|34.4833|70.1131|70.3726|
|低仰角城区|30.2994|63.1500|63.9458|

低城区六 TX、七目标 RX均改善，但 TX14-7的准确率仍只有27.1571%。clean下降0.4976个百分点。配对原clean特征的接近度改善也只构成统计证据；512参考位置中配对更近的比例为44.5131%，不能据此宣称所有补出细节属于真实TX。

完整郊区 Macro-F1、修复/损坏、逐 TX/RX/day 与特征指标见 [R09正式报告](../automation_reports/CV-SincNet/20261019-phase1-restoration-formal-FcrossRX50-s2026092701-r09/report.md)。原E200 cleanF和E30 AsF属于不同读出，见 [R05报告](../automation_reports/CV-SincNet/20261019-phase1-restoration-formal-frozen-residual-s2026092701-r05/report.md)；其中AsF低城区两TX有负收益。R07独立负结果见 [源物理估计报告](../automation_reports/CV-SincNet/20261019-phase1-restoration-learned-action-source-s2026092701-r07/report.md)。

## 复用已有权重

在项目顶层和已验证 Phase1 Python 环境下使用。下列权重是本地私有artifact引用，未进入Git；R03 final.pt 包含冻结coarse和corrector的完整状态，不需要重新训练。

```python
from pathlib import Path
import torch
from experiments.pamf_residual_restore.runtime import load_checkpoint
from experiments.pamf_fcrossrx_readout.frozen import load_reader

artifacts = Path("local_artifacts/pamf_restoration")
restorer, _ = load_checkpoint(
    artifacts / "trained/20261019-phase1-restoration-residual-refinement-selfsup-m3-r03/residual_deterministic/final.pt",
    "residual_deterministic", "cpu",
)
frozen_F = load_reader(
    artifacts / "fcrossrx50_reference_20261019_r01",
    Path("local_artifacts/datasets/phase1_manysig_reference_20261019/source_cache"),
    "cpu",
)

@torch.inference_mode()
def restore_and_read(received):
    # 与原数据一致的float32 [B,2,256]当前接收IQ；不输入clean或信道真值。
    restored = restorer(received)
    spectrum = torch.fft.fft(
        torch.complex(restored[:, 0], restored[:, 1]), norm="ortho"
    )
    # FFT导出供检查；原PAMF各分支继续按原图从restored实际计算。
    return dict(iq=restored, fft=spectrum, logits=frozen_F(restored))
```

加载器严格检查原来源、数据角色和checkpoint配置，不替换分类头，不拟合参数。现有 CPU完整1800源V诊断及本次N607正式读出已覆盖这两个实际模型的复合前向。这里的 source_cache 仅供既有 Phase1 契约核对，不能把该示例当作 Phase2/星上部署访问source的许可。

当前最高风险是严重退化中的不可辨识细节和恢复不确定性；后续机制只能由合法源证据研发，目标结果不得回流调参、选模或选择性重跑。多seed、独立物理族与真实在轨身份验证仍未完成。

## 本轮验证与追踪

在已验证的本机 Phase1 Python 中实际分别执行 `python -X utf8 -m pytest -q tests/test_pamf_action_restore.py`（11项通过）及 `python -X utf8 -m pytest -q tests/test_pamf_formal_f50.py`（最终8项通过）。N607实际执行compile、模块入口、真实源checkpoint smoke、R07完整E20源评分及R09完整原F50独立truth-last评分，原进程均已退出且产物完成。

[R07追踪](PAMF_ACTION_RESTORE_R07_20261019.md)为7项已验证、1项延期；[R08/R09追踪](PAMF_FORMAL_F50_R08_20261019.md)为5项已验证、1项延期。合计12项已验证、2项延期、0项功能需求拒绝或阻断。R07属于条件性物理近似且未通过源科学标准；这些执行验证不等于最终逐细节身份真实性已证明。
