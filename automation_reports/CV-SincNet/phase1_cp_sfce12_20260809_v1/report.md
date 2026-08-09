# phase1_cp_sfce12_20260809_v1实验报告

## 1. 预注册

- 状态：`LOCAL_VERIFIED / READY_FOR_N607_RELEASE / NO_PERFORMANCE_RESULT`
- 日期：2026-08-09
- 负责人：`/root`；唯一N607 runner：`/root/n607_geosat_lite_runner`
- 目标：检验固定CB-SFCE有标签LEO风险在冲突投影后，能否保留其广泛LEO增益并消除F1/F5尾部负迁移。
- 来源：第3轮回顾后唯一允许候选；不调`lambda=0.10`、`gamma=1`、场景、采样、数据或矩阵。
- 边界：source-only；无RX/day/domain条件、teacher、表示对齐、新head、拒识阈值、proxy/held训练或选参。

## 2. 冻结机制

C保持GeoSat-C共同base续训。G保留CB-SFCE损失，仅在共享identity encoder与精确classifier head上处理新增梯度：若`dot(a,b)<0`且`||b||²>0`，则`a'=a-dot(a,b)/||b||²*b`；否则`a'=a`。其中`b`是共同base梯度，`a`是`0.10*L_SFCE`梯度。

- AMP：base只做一次scaled backward+unscale；aux用同scale做VJP再除scale；未投影aux不写入`.grad`。
- all-trainable VJP证明scope外aux为None/零；scope内None/disconnected/nonfinite失败。
- 无epsilon近似；`b=0`是合法无冲突。
- 每个G batch必须逐一闭合projection、outside audit、optimizer state step；no-step为0。
- 终态local4×3全覆盖，encoder/head各至少一次真实conflict，否则`REJECT_CP_SFCE_INERT`。

## 3. 本地版本

- Git仓库：`E:\type10-7\code\snapshots\phase3_responsibility_20260807_wt`
- 实现commit：`20b166b69acf7537c85037f1543007577c22ab9f`
- 独立复核：`P0=0,P1=0,ALLOW`

|文件|SHA256|
|---|---|
|`code/SSDG/train_ssdg.py`|`1dd3af3e744aa804cd45dcdef36eec9f8f6909f7f568ec09f8e7aae51c240a4e`|
|`code/cvsrffi/phase1_cp_sfce.py`|`37686279e3a12852ed8bace0020ac962ba10e2c1f55fa36b219aa5177f382b1c`|
|`code/tests/test_phase1_cp_sfce.py`|`2f66d8e46c0693040956fbd01a96cde1e9ef50d1841d75d5ced0f49d7664359b`|
|`code/scripts/launch_phase1_cp_sfce12_20260809.sh`|`ca53b5bf0a64db84b640ed55146ff114e7005b021847abcdc32307dc7d9c0810`|
|`analysis/phase1_cp_sfce_design_20260809.md`|`04e31a6c20c3eb259c315f8e412e6e2fec5b126b4c28bd3714016ce652d6bf09`|

验证：py_compile通过；CP+CB必要回归17项通过；真实`lite_d`无query等价训练环通过；`bash -n`、dry-run=12、`git diff --check`通过。

## 4. N607发布

- run ID：`phase1_cp_sfce12_20260809_v1`
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cp_sfce12_20260809_v1_20b166b6`
- run/log：`/home/szu2070436088/2510044040/CV-SincNet/{runs,logs}/phase1_cp_sfce12_20260809_v1`
- outer：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cp_sfce12_20260809_v1.launch.out`
- CWD：`<release>/code`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- 数据：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`
- 基线：`runs/phase1_loto_clsgeo12_20260808_v1/F1C...F6C/final_ssdg.pth`

```bash
cd /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cp_sfce12_20260809_v1_20b166b6/code && nohup setsid env RUN_ID=phase1_cp_sfce12_20260809_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cp_sfce12_20260809_v1_20b166b6/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python bash /home/szu2070436088/2510044040/CV-SincNet/releases/phase1_cp_sfce12_20260809_v1_20b166b6/code/scripts/launch_phase1_cp_sfce12_20260809.sh > /home/szu2070436088/2510044040/CV-SincNet/logs/phase1_cp_sfce12_20260809_v1.launch.out 2>&1 < /dev/null & echo $!
```

GPU0=`F1C+F5G`，GPU1=`F1G+F5C`，GPU2=`F2C+F6G`，GPU3=`F2G+F6C`，GPU4=`F3C`，GPU5=`F3G`，GPU6=`F4C`，GPU7=`F4G`；每卡不超过2个。

## 5. 健康、产物与判据

- 12臂40E、final-only；不做额外1E预飞。
- 仅P0、执行异常、无进展或至少两个arm同一确定性异常停止；不按性能停止。
- expected：12套metrics/final/checkpoint/config/terminal/resource/heldout receipt；G另有CP终态合同。
- `NON_PROMOTABLE_P0_DISABLED/exit8`是预期终态，不是技术失败。
- 训练完整后复用已验证的CB-SFCE 42步postfreeze结构，仅替换training root/candidate名；非补偿门完全相同。
- 任一clean/LEO/proxy门失败或CP inert即`REJECT_CP_SFCE_PERMANENT`，不调参、不重试、不进入Phase3。

## 6. 运行回填

待runner回填archive/member SHA、launcher/child PID、GPU、epochs/exits、逐batchCP合同、artifact SHA与清理状态；当前无性能结果。
