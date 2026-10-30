# PAMF 固定 GMM 已知类竞争与 OE 背景

PLANNED，无新性能。Run `20261031-phase3-class-competition-s2026092701-r01`，actualcode `b30db356419342dd15ecf811e444916aa1e5c4df`，完整 Phase3 目标 ACTIVE。唯一主 hybrid_competition、原 pooled_ratio/predicted_ratio 精确控制和 class_competition 诊断，共四列×三 DURA。公式、固定权重、矩阵和科学门槛见 `docs/PAMF_PHASE3_CLASS_COMPETITION_PLAN_20261031.md`，逐 row/六类 seed/命令路径在 experiment.json 唯一维护。

原 F0 Tri-SIG-TIME SSL E50/11100/default6、source-only r03/dc01a402、DURA R69 E20/4940、classdensity/4af known90/OE32/normalizer 和两旧六τ保持。ℓ_j是每旧类15等质量模态logdensity，b_c=logmeanother。诊断ℓ_c−b_c；主ℓ_c−logaddexp(ℓ_OE,b_c)+log2，即0.5OE+0.5其他五旧类背景，T1/权重0.5不搜索。F0预测c固定，不用密度argmax改身份；旧OVA/logits已有竞争，新问题仅该GMM明确公式。

actual三CP synthetic2各首步smokePASS后继续→核原sourceonly/cache→原density及新公式冻结→V27000两新六τ+旧two6τexact→全部τ冻结→NV3150、cleanQ864、starQ864全四列→独立CUDAempty CPU三truth。阈值只源已知V按原known12 CK90/同G原F0闭集−AK2pp，30交叉组诊断；不U/真K/配额/NV或Qfit/cleanlearning。

原source契约392005/VALIDATED_ONCE：L6300/Us56700/OE6134均祖先、本轮0；V27000额外knownlabels，NV三个unknown19-15/19-16/7-1各1050。sourceRX1/3/4/6/8=1-19/18-2/19-2/2-19/3-19、2021-03-08/15/23。旧六14-10/14-7/20-15/20-19/6-15/8-20。crossRX24-1/index26/03-15同864物理Q、old576/nonold288各类96，两条件均DURA；U384/03-08及正式RX24-18未读。G10 MIX01/600km/25MHz/2.45GHz residual/post_sync/noEQ/refSNR30dB/reflect、原六景权重/郊区城区各0.5、receiver2027/scene2026103001/channel2026104001，一record一received，不重抽。

源known12、cleanRU95/starRU90、CK90、sameDURAF0闭集损2pp、逐主景完整roster及主starAK/RU均不低于原predicted是预登记gate，H无独立门槛；source失败不取消Q。历史NV/Q曝光、单seed/RX/source日期重合、high缺19-15导致RU/H NA，不删景/补抽/换主/选择性重跑。无DA/发现/注册，AN/HOW=N/A，全目标ACTIVE。

newNN/真实F0Gsignal/sourceIQ0，新density/normalizer/statisticalparameters0，Vonly12新τ实际fit，old12exact，总τ24/192B。原density364340B/NN834665、原source训练和缓存成本继承；actualCP synthetic2each另计，无source/U逐行bank部署；进程时间非B1。

N607解释器 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-class-competition-s2026092701-r01-release`，唯一owner `codex-phase3-class-competition-20261031`；不可覆盖root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-class-competition-s2026092701-r01`。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_class_competition_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-class-competition-s2026092701-r01-release/runtime.json`。单producerGPU空闲≥4096MiB/每卡新增本任务≤2，CPU评分CUDAempty；不停止/修改/重启其他owner。

聚焦core/producer→CPU/launcher协议负测、唯一直接P0/P1通过；具体测试数及证据在private verification.json。一次preflight/archive SHA/compile，启动核PID/CWD/argv/GPU/log增长。非法角色/祖先/非有限/输出碰撞/无法合法pred保留，无partial truth/覆盖重启；UNKNOWN先核原PID/root，科学低分不停健康任务。IQ/weights/preds/rawlogs/private runtime不进Git。

## 最终结果（ANALYZED_DEVELOPMENT / NO_PROMOTION）

唯一主hybrid_competition，source/clean/star AK89.688889/68.576389/62.673611%、CK96.514815/85.069444/83.506944%、RU61.301587/100/76.736111%、H72.826729/81.359423/68.995607%。
主−predicted：clean正确0增13失；star正确0增8失、未知拒对1增2失，AK/RU/H下降1.388889/.347222/.976718pp。class_competition星地AK64.583333但RU22.569444，不能换主或称无OE训练。
同DURA原F0闭集损source1.292593/clean11.458333/star10.243056pp；sourceknown12通过但42未全过，未来CK/2pp/星地RU失败，high缺19-15相关RU/H N/A。旧predicted重点候选/pooled冻结对照保留，NO_PROMOTION/目标ACTIVE。
固定GMM新公式为lc−logaddexp(lOE,logmeanother)+log2；不是此前未做类竞争。NN/Gaussian/normalizer新fit0，但额外V27000标签实际新校准12τ，旧12exact，24FP64=192B；已存下一tail受RX3损2.0185185pp约束，无新扫描。
真实GFsignal/sourceL/OE/U0，三CP synthetic各2；NN834665/density364340B及原训练/cache成本继承，无prior/sourcebank，AN/HOW N/A、无UDA/REG。
predict/score/pipeline=17.873421/15.735417/33.693221s；独立driver/CPUlive-CWD-argv-CUDAempty已捕获，predictor仅pipeline GPU4/产物，显存UNKNOWN；终态live/GPU空foreign0，三生产stdout无warning/error。
9unique+30sub、一次review，单包661files/data0、SHA/compile/runtime VERIFIED。完整12点/全分组/三pair见[公报](../../../docs/PAMF_PHASE3_CLASS_COMPETITION_RESULTS_20261031.md)，未新增实验或重评分。
