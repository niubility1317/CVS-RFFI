# ADV3B02-FCR Task9报告：完整FCR模型接线与输出schema

## Status

`DONE_WITH_CONCERNS`。Task9拥有的聚合、可选模型接线、关闭态回归、开启态单视图forward/backward和Task3-5直接回归已完成。未执行训练、真实checkpoint、N607或独立P0/P1审查。

关注点限定为Task4已提交响应算子对需梯度`s_hat`的原地固定基更新：直接组合会触发PyTorch autograd版本冲突。Task9未越权修改Task4模块，而是在聚合边界把指纹响应激励detach；内容仍经`s_hat→PhysicsOrderedDecoder`获得梯度，指纹算子、nuisance、Decoder、`z_id_raw`及身份主干仍由聚焦反传测试证明可达。Task10若需要`G_f→E_s`梯度，应在其拥有的训练路由中明确处理，不能静默假定当前Task4算子支持该边。

## Files changed

- `code/model_dual_cvsincnet.py`
- `code/cvsrffi/phase1_fcr_types.py`
- `code/tests/test_phase1_fcr_model_contract.py`
- `code/tests/test_phase1_fcr_forward.py`
- `docs/CVS_PHASE1_ADV3B02_FCR_TRACE_20260901.md`仅FCR-02至FCR-07行
- 本报告

## TDD red/green evidence

红测命令：

```text
conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_model_contract.py code/tests/test_phase1_fcr_forward.py -v
```

实现前结果为`1 passed,2 failed`：关闭态严格兼容测试通过；开启态分别因`assert model.fcr is not None`和`KeyError:'z_id_raw'`失败，确认缺失的是Task9聚合实例化与公开输出接口。

接线后的同一命令结果为`3 passed`。过程中同一backward测试稳定复现Task4响应基原地更新导致的`RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`；聚合边界最小detach后该测试通过。

## Aggregate call graph

```text
x
→ConservativeCanonicalizer(x)
→ContentFactorEncoder(canonical_iq)
→FingerprintFactorEncoder(z_id,canonical_iq,residual_iq,excitation(s_hat.detach()))
→ExcitationConditionedFingerprintOperator(s_hat.detach(),fingerprint)
→StructuredNuisanceEncoder(x,eta_hat)
→PhysicsOrderedDecoder(s_hat,delta_f,nuisance)
→FCRAggregateOutput
```

聚合只调用Task3-5已提交API，没有复制canonicalization、内容编码、固定响应基、nuisance或Decoder逻辑。`pair_context`仅为可选关键字上下文，当前单视图forward不依赖clean companion。

## Enabled output schema

对ADV3B02的160维身份配置，`return_aux=True`新增：

| Key | Shape/type | Contract |
|---|---|---|
| `z_id_raw` | `[B,160]` | 与既有`z_id`同一对象，不改值 |
| `z_f_id` | `[B,160]` | 逐样本unit L2 |
| `z_tx_state` | `[B,16]` | 独立慢变TX状态 |
| `z_s` | `[B,input_len/content_stride,32]` | 时序内容token |
| `z_n` | `dict` | 仅`channel/receiver/sync/gain`，维度16/8/6/3 |
| `fcr_decode` | `FCRDecodeOutput` | `mu_iq:[B,2,input_len]`、`log_variance:[B,input_len]`、complex64`delta_f:[B,input_len]` |
| `fcr_quality` | `dict[str,Tensor]` | canonical/content/fingerprint/nuisance/decode具名有限诊断 |
| `feature_schema` | `str` | 精确为`ADV3B02:FCR:z_f_id:unit_l2:160:v1` |

默认未传`fcr_config`时只将`FCRConfig.input_len`绑定模型`input_len`，其余Task1维度和方差默认值不变。显式`use_fcr=True`才实例化`fcr.*`参数。

## Compatibility and single-view proof

- `use_fcr=False`时`fcr_config=None`、`fcr=None`，state dict无`fcr.*`键。
- 关闭态测试比较legacy与显式关闭模型的完整顶层/嵌套输出键和全部张量，逐元素`rtol=0,atol=0`一致；输出无任何FCR-only key。
- 同seed的开启/关闭模型中，既有`tx_logits`和`z_id`逐元素完全一致；FCR在所有旧模块之后实例化，未改变旧参数初始化序列。
- `DualCVSincNetDisentangle.forward`和聚合`forward`签名都没有`clean_companion`；单个`x:[B,2,input_len]`即可生成全部FCR aux输出。

## Backward evidence

有限目标联合读取`z_f_id`、`z_tx_state`、`z_s`、`fcr_decode.mu_iq`和`log_variance`。测试确认：

- `z_id_raw.grad`存在、有限且非零；
- 至少一个FCR可训练参数梯度存在、有限且总绝对值非零；
- 既有`id_backbone`梯度存在、有限且总绝对值非零；
- 原`tx_logits`分类输出没有改路到`z_f_id`。

## Regressions and trace

Task3-5直接回归命令：

```text
conda run --no-capture-output -n ssr-gpu python -m pytest code/tests/test_phase1_fcr_canonicalizer.py code/tests/test_phase1_fcr_content.py code/tests/test_phase1_fcr_fingerprint.py code/tests/test_phase1_fcr_nuisance.py code/tests/test_phase1_fcr_decoder.py -v
```

结果：`15 passed`。

FCR-02至FCR-07保持`implemented`，Verification补充本地端到端可达证据；没有把训练、真实checkpoint、strict pair、真实数据或N607状态写成已验证。

## Self-review

- 关闭态：完整state/output兼容测试通过，无FCR参数和FCR-only输出。
- 旧语义：`z_id_raw is z_id`，`tx_logits/z_id`开启前后逐元素一致，未静默切换分类feature。
- 可选实例化：只有`use_fcr=True`存在`fcr.*`state key。
- 无旁路：Decoder仍只接收`s_hat/delta_f/nuisance`，不接收原始`x`。
- 单视图：无必需clean companion或pair参数。
- schema：精确常量`ADV3B02:FCR:z_f_id:unit_l2:160:v1`。
- 梯度：聚焦backward覆盖FCR、`z_id_raw`和身份上游；Task4的`G_f→E_s`限制已明确记录。
- 范围：未改Task3-8模块、数据、训练、checkpoint、launcher或其他测试/报告。

## Commit and publish

本报告与Task9拥有文件以`feat:wire-FCR-into-ADV3B02`同一提交发布。提交后的本地HEAD、push结果和远端OID将在任务完成回执中独立给出；不把未来OID回写到同一提交，避免改变提交对象。

## Interfaces for Tasks10/11

- Task10可从模型aux直接读取`z_id_raw/z_f_id/z_tx_state/z_s/z_n/fcr_decode/fcr_quality/feature_schema`，也可调用`model.fcr(...)`取得具名`FCRAggregateOutput`内部factor对象；配对损失仍须由合法`FCRPairBatch`显式提供。
- Task10必须保留`z_id/tx_logits`旧路径，feature选择只按精确schema显式切换到`z_f_id`。
- Task11可序列化`fcr_config`、`fcr.*`state和精确feature schema；单LEO IQ加载后不应要求clean companion。
- 当前FCR身份入口严格消费160维ADV3B02`z_id`；其它embedding宽度的模型变体不属于本Task9 schema，不得静默投影或伪装为同一schema。
