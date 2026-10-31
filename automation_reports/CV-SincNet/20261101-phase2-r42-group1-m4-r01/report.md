# R42 Phase2 对比组补全

用户明确固定 R42，使用 Tri-SIG-TIME SSL 四个原始种子与 DURA v1.0，完成此前 Group1 原生方法对比。R42／原版 LocalRidge 的全部8套 MIX 扩展为 K=1/5/10/20、N=2/5/10/20，共672单元；原三种单场景各5 episode 的1260单元保留 K5/N20。support raw/DURA 严格分开，R42 query raw/G-only/同记录融合各列；原 LocalRidge 保留四个纯路径。含共用 A0 共167496行、1932单元。

已有 R42 主点及全KN先导、原 LocalRidge 所有已完成 prediction/metrics 依据 capsule/split/Fseed/路径配对复用，无目标分数选择。4个 R42 source NN 与源K1规则均复用，新增 NN/F/G 更新为0。seed2–4的 source-stage runtime 按已提交 main source() 的真实调用重建（单seed、crossrx、main/sources/sseed），明确记录重建来源；读取原 source selection 的 source-only 更新历史验证，而非伪造历史记录。

全体新 prediction 固定后，独立评分仅新行，原完成指标直接复用。逐KN输出 C_old/C_new/宏H、配对Δ和四seed统计，并额外对各K query的真实交集评分。每格+5个百分点是验收目标，不预先承诺；融合对原raw和G分别比较，不冒充同输入路径。开发基准已曝光，不宣称新盲测。

外部原生方法已有训练/预测与健康独立评分队列保持，由同一既有owner完成；ISSL遵守用户先停止的具体指令。MoPC缺失的作者encoder将作为有明确输入适配身份的独立run补齐；不得以旧generic encoder代替。旧29种任意组合链不恢复。

启动命令、1932逐行矩阵、环境、不可变root、checkpoint来源、停止规则见 experiment.json。GPU0/2/3/4/6每卡最多两个新增本任务预测进程、source每卡一个，启动前剩余显存至少6000MiB，外部owner GPU1/5/7及其他健康任务不改。任一技术/权限/输出错误停止新入队并排空健康本任务，不盲重试；科学低分不停止。真实源checkpoint smoke首步通过即继续，无额外seal/hash链。

当前状态：FAILED（外部 CUDA 环境故障，见末尾最新证据）。代码提交、一次P0/P1、真实source smoke、归档传输、远端编译及启动后PID/CWD/argv/GPU/log证据按项目最小流程完成。

启动后独立读回（2026-11-01T01:31:29.902910+08:00）：VERIFIED RUNNING: four actual source checkpoint smokes PASS, newNN/F/Gupdates0; controller2148792; predictions7/1932, active10, no failures; complete208row newly computed units observed. silent logs0 with actual fit/prediction/resource artifact growth. Code14af25d0ff68d0d4de80357849d8a7c69018ebd3.

归档传输SHA比对、远端编译、代码双仓库push及remote OID均VERIFIED；真实PID/CWD/argv、GPU和日志/产物增长已核实。最终分数尚待完整冻结与独立评分，不宣称+5pp达标。

最新故障核实（2026-11-01T05:38:20.760136+08:00）：VERIFIED EXTERNAL_CUDA_ENVIRONMENT_FAILURE: GPU7 handle C2:00.0 unavailable/NVML failure; fresh UUID-pinned CUDA probes on0/2/4 all returned0 devices and allocation failed; own controllers/workers all exited, no new admissions or blind restart. {"R42_completed_units": 37, "R42_prediction_rows": 7696, "MoPC_completed": {"source": 4, "predict": 398, "score": 397}}

故障来自 N607 GPU/NVML/CUDA 环境：GPU7（PCI C2:00.0）设备句柄不可用；分别绑定剩余GPU0/2/4 UUID的新进程探测均无法初始化CUDA。不能凭剩余显存空闲判断可运行，也不将其归因于方法性能。两条本run均已按停止规则退出，无存活owned worker；无需新增终止操作。

已完成prediction、评分、四个MoPC源模型及R42四源状态全部保留，未按分数改变矩阵。等待GPU环境恢复后新建不可变恢复run，复用全部闭合产物，不覆盖旧root，不重训已完成源模型。本任务不执行管理员GPU重置、驱动操作或服务器重启；其他owner的输出和控制器未修改。原r10控制器独立读回也已退出，其旧RUNNING计数已过时，不冒充仍在评分。
