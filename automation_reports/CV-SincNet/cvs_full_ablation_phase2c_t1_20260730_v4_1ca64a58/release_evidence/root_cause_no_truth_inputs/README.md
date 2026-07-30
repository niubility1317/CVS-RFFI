# v4根因行no-truth本地复现输入

## 边界

本目录只保存3个真实根因physical row的原始predictor request、只读feature cache payload和对应manifest sidecar，用于本地prediction-only/no-truth smoke。未复制数据集、clean/source IQ、query truth、scoring sidecar、checkpoint、prediction或评分结果；未重验数据，也未修改N607。

3份manifest均明确`query_truth_present=false`、`query_role_present=false`、`clean_source_samples_present=false`。3份NPZ只包含deployment prototypes、ground basis/spectral weights、3个`leo_*_weak`场景的query features/query tokens以及old/new support features/support labels；不存在query label、query truth或query role数组。

## 文件清单与原始路径

manifest由发布器以canonical JSON计算绑定SHA后再追加一个换行写盘。因此表中同时记录完整文件SHA256和去掉末尾LF后的绑定SHA256；request里的`feature_cache_manifest_sha256`对应后者。

|row|本地文件|原始N607路径|字节|完整文件SHA256|绑定SHA256|
|---|---|---|---:|---|---|
|`phys_1b9d0cee16897a454ddb3aa7`|`phys_1b9d0cee16897a454ddb3aa7.json`|`/home/szu2070436088/2510044040/CV-SincNet/requests/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58/predict/phys_1b9d0cee16897a454ddb3aa7.json`|1681|`f7f5afec075fa1521982fe0bd8d493db528fa73838f7760c5abff6a80d214bb8`|同完整文件SHA|
|同上|`features.manifest.json`|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features/rx_7_14/method_7282101/new20/k_10/stage2c/features.manifest.json`|12862|`c66756c1bdfac11b246b004125e5d084f43c9f124f7ad9c03864e9c63d4b62b0`|`383a7fba0a356a966843cadfb0f5b350c3d6b537da8728385646a9d9cfa17938`|
|同上|`features.npz`|同manifest目录下`features.npz`|3355594|`4e8dfa705dd74aa41b18dec94ac2c4ec562c81a4b391c68bc850a83e044c2ded`|同完整文件SHA|
|`phys_af88df635bf6b18beb105d08`|`phys_af88df635bf6b18beb105d08.json`|`/home/szu2070436088/2510044040/CV-SincNet/requests/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58/predict/phys_af88df635bf6b18beb105d08.json`|1664|`20f8911cfa6ed2f464c038a202e5fb7318d0c2ff35b9c180a18d086c929c638b`|同完整文件SHA|
|同上|`features.manifest.json`|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features/rx_3_19/method_7282102/new20/k_10/stage2c/features.manifest.json`|12862|`939dd7522aac201ec20c6ec71d6cc495b2ded27262fa78f27cd949373adf2fcc`|`06d366d61904ab0ce5b8f92e1239cf11d4c70cf9f925b0be5e38f9386ee65186`|
|同上|`features.npz`|同manifest目录下`features.npz`|3355594|`7ab9a893561e87be30321f965dd1474dfea57c316c2838bc0da84ce31db8a389`|同完整文件SHA|
|`phys_37cc012d2b44700e361a5a9c`|`phys_37cc012d2b44700e361a5a9c.json`|`/home/szu2070436088/2510044040/CV-SincNet/requests/cvs_full_ablation_phase2c_t1_20260730_v4_1ca64a58/predict/phys_37cc012d2b44700e361a5a9c.json`|1676|`0eeb460f77763fe5f450c9b240f0075c21c8a780bac923480e8dcef7cb05cb32`|同完整文件SHA|
|同上|`features.manifest.json`|`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/cvs_full_ablation_phase2c_feature_t1_20260730_v3_47212437/artifacts/features/rx_8_8/method_7282103/new20/k_10/stage2c/features.manifest.json`|12859|`e0baf3e18d8b0915d1f62d1f025dacb2bf5c2606b4a2aa2754fe6125df3b8352`|`6021619fc9a9efab942521011046ece1a8afafe8e5af5be856201777f5d2ce30`|
|同上|`features.npz`|同manifest目录下`features.npz`|3355594|`17b7a6ffd3ef73918ff136d8cb70e17f8f12d18dafa70c7d0c33192cbf1e60d7`|同完整文件SHA|

## 本地no-truth smoke依赖

- 工作树代码必须包含正式实现commit`1ca64a586b85c97fbaa2a677a6ca5776ffd239b3`对应的`code/`内容。
- 本地环境使用`ssr-gpu`；入口为`code/scripts/run_full_ablation_stage2_row.py`。
- 每个row只依赖其目录中的request、`features.manifest.json`和`features.npz`。prediction-only smoke不需要也不得打开scoring sidecar、query truth或数据集。
- 原始request作为证据保持不变。运行前在临时目录生成derived request，只替换：
  - `feature_cache_manifest`为本地绝对manifest路径；
  - `feature_cache_payload`为本地绝对NPZ路径；
  - `output_root`为新的空临时输出目录。
- 保持`feature_cache_manifest_sha256`、`feature_cache_payload_sha256`、row/config/capsule/split/seed字段不变。`device`可按本地smoke资源设为`cuda:0`；不得启动scorer。

从`E:\type10-7\github_publish\CVS-RFFI-repo`运行：

```powershell
conda activate ssr-gpu
$env:PYTHONPATH = (Resolve-Path '.\code').Path
python .\code\scripts\run_full_ablation_stage2_row.py --request <derived-request.json>
```

成功门为predictor自然返回0，输出`row_execution_receipt.json`和只读`predictions.cvspred`，receipt保持`query_truth_opened=false`且`fit_query_rows_used=0`。本任务只回收输入和给出命令，不执行该smoke。

本地`ssr-gpu`只读校验已通过：3个request的row绑定正确，payload SHA与request一致，manifest去掉发布器追加LF后的canonical SHA与request一致；每份NPZ均为21个数组，未发现query label/truth/role数组。
