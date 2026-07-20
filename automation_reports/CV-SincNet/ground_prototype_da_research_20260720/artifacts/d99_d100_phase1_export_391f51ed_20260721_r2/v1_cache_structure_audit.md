# D99/D100 Phase1 v1 cache结构与provenance只读审计

## 审计边界

- 审计对象：冻结`source_validation` cache set中的三个`cvs_leo_weak_iq_cache_v1` NPZ。
- 只读取ZIP成员名、NPY头部、非IQ元数据数组和标量`manifest_json`。
- 未读取、打印或下载`leo_weak_iq`数值；因此IQ行内容与逐行`post_channel_iq_sha256`的对应关系未重新计算。本审计对该层只依赖NPZ文件SHA256及既有逐行哈希。
- cache set SHA256：`125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74`。

## 精确有序member list、shape与dtype

三个NPZ的有序member集合完全相同，共17项；除`sat_scenarios`和`manifest_json`的Unicode宽度外，shape/dtype一致。

|序号|member|leo_clear_weak|leo_low_elev_weak|leo_rain_weak|
|---:|---|---|---|---|
|1|`leo_weak_iq`|`(8400,2,256) float32`|同左|同左|
|2|`raw_labels`|`(8400,) int64`|同左|同左|
|3|`domain_labels`|`(8400,) int64`|同左|同左|
|4|`tx_ids`|`(8400,) <U5`|同左|同左|
|5|`rx_ids`|`(8400,) <U4`|同左|同左|
|6|`day_ids`|`(8400,) <U10`|同左|同左|
|7|`eq_ids`|`(8400,) <U1`|同左|同左|
|8|`sig_ids`|`(8400,) <U3`|同左|同左|
|9|`dataset_role`|`(8400,) <U6`|同左|同左|
|10|`channel_views`|`(8400,) <U7`|同左|同左|
|11|`sat_scenarios`|`(8400,) <U14`|`(8400,) <U17`|`(8400,) <U13`|
|12|`satellite_seeds`|`(8400,) int64`|同左|同左|
|13|`overlay_applied`|`(8400,) bool`|同左|同左|
|14|`sample_ids`|`(8400,) <U34`|同左|同左|
|15|`post_channel_iq_sha256`|`(8400,) <U64`|同左|同左|
|16|`overlay_ids`|`(8400,) <U64`|同左|同左|
|17|`manifest_json`|`() <U2950`|`() <U2967`|`() <U2949`|

明确缺失且触发r2失败的v2成员为`source_dataset_sha256`和`source_record_indices`。旧v1未含`split_partition`或`split_rank`。

## 完整嵌入manifest字段

下表覆盖三个嵌入manifest的全部顶层字段；嵌套的`channel_config`、`channel_meta_keys`和`role_inputs`在后续代码块完整列出。

|字段|leo_clear_weak|leo_low_elev_weak|leo_rain_weak|
|---|---|---|---|
|`artifact_stage`|`phase1_offline_prechannel_export`|同左|同左|
|`build_spec_sha256`|`75a7d772aa9d39b209a9ddd2b6f310189ff43107d603d9bd8573fe6dd897256f`|同左|同左|
|`builder_sha256`|`b7dff031c8348207e7b017ac7a62459e391fbc37073dbaee26ec6d51cb99ce35`|同左|同左|
|`channel_config`|见下文|见下文|见下文|
|`channel_config_sha256`|`8ef111986fe5cb1196fffbc102defe28d2342f2249a4008d63e81a750c816380`|`b8bfc9e048422b1be84ee672bdb448d2ccfef0e471af90166f7673596e8deb2a`|`36be452fd725d208bad0d9e7af2aa9e3c8190cc8f8ceff4c5bebcf5370ea1a18`|
|`channel_meta_keys`|见下文，共18项|同左|同左|
|`channel_model`|`leo_residual`|同左|同左|
|`clean_derived_signal_access`|`false`|同左|同左|
|`clean_sample_access`|`false`|同左|同左|
|`contains_clean_rows`|`false`|同左|同左|
|`contains_post_channel_iq_only`|`true`|同左|同左|
|`iq_array_key`|`leo_weak_iq`|同左|同左|
|`output_roles`|`["source"]`|同左|同左|
|`overlay_applied_before_phase2`|`true`|同左|同左|
|`overlay_ids_sha256`|`539ca4f877adc69ea9617c66d0809a1f64703728a759c93c7ca66eed66ddf0b3`|`180824fc7a791e23f913583968ea0bcb119fe636fcb06b6174da1b97d4584f92`|`826f462dabab01972576d07d33b4653f1696333382cb7c4781af0fb6475900e0`|
|`phase2_sample_view_policy`|`leo_weak_only_no_clean_access`|同左|同左|
|`physical_sample_ids_sha256`|`d2def2acf96a9338f94b4626f77ca9b7b106a65f41615dd5c703b1b76461e1a3`|同左|同左|
|`post_channel_iq_sha256_root`|`eb658fc62342e6f6cec36339b541190238d2da8601143cb8de529e9a9154c4f6`|`8e389ff05b265d5e356a1d315a44586eabf1bd58b314e0198b2c5423216703ef`|`47eb3dd877fe0e966b75292b58e05a7e0b60593f73bde02abfabfa63cc3ce319`|
|`raw_or_clean_iq_key_present`|`false`|同左|同左|
|`role_inputs`|见下文|同左|同左|
|`role_satellite_seeds`|`{"source":4071391}`|`{"source":4071392}`|`{"source":4071393}`|
|`row_count`|`8400`|同左|同左|
|`sample_overlay_provenance_fields`|`["sample_ids","sat_scenarios","satellite_seeds","post_channel_iq_sha256","overlay_ids"]`|同左|同左|
|`scenario`|`leo_clear_weak`|`leo_low_elev_weak`|`leo_rain_weak`|
|`schema`|`cvs_leo_weak_iq_cache_v1`|同左|同左|
|`star_ground_channel_impl`|`simplified_leo_residual`|同左|同左|
|`target_channel_scenarios`|`["leo_clear_weak"]`|`["leo_low_elev_weak"]`|`["leo_rain_weak"]`|
|`target_channel_view`|`leo_weak_only`|同左|同左|

### 完整`channel_config`

```json
{
  "leo_clear_weak": {
    "K_db_range": [16.0, 24.0], "agc_resid_db": [-0.2, 0.2],
    "apply_path_loss_to_iq": false, "cfo_std_hz": 50.0,
    "channel_model": "leo_residual", "enable_atmospheric_fading": false,
    "enable_iq_imbalance": false, "enable_multipath": true,
    "fading_mode": "rician", "fc_hz": 2462000000.0,
    "fs_hz": 25000000.0, "loo_level": "light", "max_delay_samp": 2,
    "multipath_profile": "weak", "num_taps": [2, 2],
    "orbit_probs": {"GEO": 0.0, "LEO": 1.0, "MEO": 0.0},
    "phase_noise_inc_std": [0.0, 0.0005], "pwr_decay": 0.08,
    "scenario": "leo_residual", "snr_db": [22.0, 32.0],
    "star_ground_channel_impl": "simplified_leo_residual",
    "theta_deg": [35.0, 90.0], "use_residual_doppler": true,
    "weather": "clear"
  },
  "leo_low_elev_weak": {
    "K_db_range": [8.0, 18.0], "agc_resid_db": [-0.3, 0.3],
    "apply_path_loss_to_iq": false, "cfo_std_hz": 90.0,
    "channel_model": "leo_residual", "enable_atmospheric_fading": false,
    "enable_iq_imbalance": false, "enable_multipath": true,
    "fading_mode": "shadowed_rician", "fc_hz": 2462000000.0,
    "fs_hz": 25000000.0, "loo_level": "light", "max_delay_samp": 3,
    "multipath_profile": "weak", "num_taps": [2, 2],
    "orbit_probs": {"GEO": 0.0, "LEO": 1.0, "MEO": 0.0},
    "phase_noise_inc_std": [0.0001, 0.0008], "pwr_decay": 0.12,
    "scenario": "leo_residual", "snr_db": [16.0, 28.0],
    "star_ground_channel_impl": "simplified_leo_residual",
    "theta_deg": [10.0, 35.0], "use_residual_doppler": true,
    "weather": "clear"
  },
  "leo_rain_weak": {
    "K_db_range": [10.0, 20.0], "agc_resid_db": [-0.3, 0.3],
    "apply_path_loss_to_iq": false, "cfo_std_hz": 70.0,
    "channel_model": "leo_residual", "enable_atmospheric_fading": false,
    "enable_iq_imbalance": false, "enable_multipath": true,
    "fading_mode": "rician", "fc_hz": 2462000000.0,
    "fs_hz": 25000000.0, "loo_level": "light", "max_delay_samp": 3,
    "multipath_profile": "weak", "num_taps": [2, 2],
    "orbit_probs": {"GEO": 0.0, "LEO": 1.0, "MEO": 0.0},
    "phase_noise_inc_std": [0.0001, 0.0007], "pwr_decay": 0.1,
    "scenario": "leo_residual", "snr_db": [14.0, 26.0],
    "star_ground_channel_impl": "simplified_leo_residual",
    "theta_deg": [20.0, 80.0], "use_residual_doppler": true,
    "weather": "rain"
  }
}
```

### 完整`channel_meta_keys`

```json
["K_db","atmospheric_fading_applied","cfo_hz","channel_model","d_km","fD_hz","h_km","iq_imbalance_applied","multipath_profile","num_taps","orbit","orbital_doppler_applied","path_loss_iq_applied","pl_db","residual_cfo_hz","snr_db","state","theta_deg"]
```

### 完整`role_inputs`

三个场景的`role_inputs`相同：

```json
[
  {
    "dataset_seed": 4071391,
    "dataset_sha256": "2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f",
    "dataset_size_bytes": 2359341461,
    "physical_sample_count": 8400,
    "requested_days": null,
    "requested_rxs": "0,1,2,3,4,5,6",
    "requested_tx_ids": "0,1,2,3,4,5",
    "resolved_info": {
      "days": null,
      "pkl": "/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl",
      "role": "source",
      "rxs": "0,1,2,3,4,5,6",
      "size": 8400,
      "tx_idx": [0,1,2,3,4,5],
      "tx_labels": ["14-10","14-7","20-15","20-19","6-15","8-20"]
    },
    "role": "source"
  }
]
```

## provenance复算结果

旧v1实现的physical ID公式为：

```text
dataset_role|tx_id|rx_id|day_id|eq_id|sig_id
```

三个NPZ均可只用现有逐行成员精确复算，不需要`source_dataset_sha256`或`source_record_indices`：

|场景|行数|唯一复算ID数|逐行ID完全一致|与首场景顺序一致|physical root|
|---|---:|---:|---|---|---|
|`leo_clear_weak`|8400|8400|true|true|`d2def2acf96a9338f94b4626f77ca9b7b106a65f41615dd5c703b1b76461e1a3`|
|`leo_low_elev_weak`|8400|8400|true|true|同上|
|`leo_rain_weak`|8400|8400|true|true|同上|

每个场景的复算physical root均同时匹配嵌入manifest与外层`cache_audits`。`sample_ids`长度范围为30至34。

使用复算sample ID、逐行`scenario`、`satellite_seed`、`channel_config_sha256`及既有`post_channel_iq_sha256`，三个场景的8400个`overlay_ids`均逐行完全复算一致，overlay root同时匹配manifest和`cache_audits`。既有IQ哈希列表的root也匹配两处声明：

|场景|IQ哈希列表root|overlay root|嵌入manifest canonical SHA256|
|---|---|---|---|
|`leo_clear_weak`|`eb658fc62342e6f6cec36339b541190238d2da8601143cb8de529e9a9154c4f6`|`539ca4f877adc69ea9617c66d0809a1f64703728a759c93c7ca66eed66ddf0b3`|`e098566428e59e4e1320ed42e78ff73864f15b8052460c05a5cd6ca8c0858211`|
|`leo_low_elev_weak`|`8e389ff05b265d5e356a1d315a44586eabf1bd58b314e0198b2c5423216703ef`|`180824fc7a791e23f913583968ea0bcb119fe636fcb06b6174da1b97d4584f92`|`811f7ca882505c836dd84fe3a7bc35a2dda0de58c4b44ad5597a1a71462d4c6b`|
|`leo_rain_weak`|`47eb3dd877fe0e966b75292b58e05a7e0b60593f73bde02abfabfa63cc3ce319`|`826f462dabab01972576d07d33b4653f1696333382cb7c4781af0fb6475900e0`|`d0a64adcd3afd193ec64c92231d603d862044b605078265c71924151211374b6`|

## 结论与修复边界

1. v1并非只有不可复算的`sample_ids`；现有`dataset_role/tx/rx/day/eq/sig`足以按历史v1公式逐行复算physical ID、唯一性与固定root。
2. `cache_audits`为三方对照提供了固定physical/IQ-hash-list/overlay/manifest roots；三个NPZ均自洽。
3. v1无法满足当前v2的“数据集SHA+原始record index逐行绑定”语义，因为这两个逐行成员确实不存在。manifest虽然含单一source dataset SHA，但没有逐行record index。
4. r2失败不是旧v1 provenance完全不可验证，而是`accepted_schemas=(v1,)`仍在member检查前共用v2的`_REQUIRED_ARRAY_KEYS`和v2 provenance字段顺序，导致兼容入口未真正分支。
5. 若主线选择修复，应为v1显式冻结历史公式并校验上述三方roots；不得伪造`source_record_indices`，也不得把v1声明升级成v2语义。
