# ADV3B02可训练骨干CSIL与MoPC-HR全量对比v7

- 状态：`PREREGISTERED_NUMPY_TORCH_ABI_COMPAT_LOCAL_VERIFIED_SMOKE_PENDING`
- predictor使用`torch.frombuffer`兼容远端NumPy2.2.5/Torch2.1.0 ABI组合，保持float32/int64、shape、样本与方法参数不变。
- predictor SHA=`ade8612996d72f287750507172b7ef664bdfabb21a42813fc32ea81e9d8bb0d8`；32项focused test、`py_compile`、`git diff --check`PASS。
- 新远端root：`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_unfrozen_paperfull_ci_20260723_v7`；固定4cell smoke全链PASS后才授权100 package/800cell/2400行矩阵。
