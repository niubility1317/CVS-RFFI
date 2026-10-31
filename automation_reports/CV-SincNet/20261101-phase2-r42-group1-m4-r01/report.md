# R42 Phase2 对比组补全

用户明确固定 R42，使用 Tri-SIG-TIME SSL 四个原始种子与 DURA v1.0，完成此前 Group1 原生方法对比。R42／原版 LocalRidge 的全部8套 MIX 扩展为 K=1/5/10/20、N=2/5/10/20，共672单元；原三种单场景各5 episode 的1260单元保留 K5/N20。support raw/DURA 严格分开，R42 query raw/G-only/同记录融合各列；原 LocalRidge 保留四个纯路径。含共用 A0 共167496行、1932单元。

已有 R42 主点及全KN先导、原 LocalRidge 所有已完成 prediction/metrics 依据 capsule/split/Fseed/路径配对复用，无目标分数选择。4个 R42 source NN 与源K1规则均复用，新增 NN/F/G 更新为0。seed2–4的 source-stage runtime 按已提交 main source() 的真实调用重建（单seed、crossrx、main/sources/sseed），明确记录重建来源；读取原 source selection 的 source-only 更新历史验证，而非伪造历史记录。

全体新 prediction 固定后，独立评分仅新行，原完成指标直接复用。逐KN输出 C_old/C_new/宏H、配对Δ和四seed统计，并额外对各K query的真实交集评分。每格+5个百分点是验收目标，不预先承诺；融合对原raw和G分别比较，不冒充同输入路径。开发基准已曝光，不宣称新盲测。

外部原生方法已有训练/预测与健康独立评分队列保持，由同一既有owner完成；ISSL遵守用户先停止的具体指令。MoPC缺失的作者encoder将作为有明确输入适配身份的独立run补齐；不得以旧generic encoder代替。旧29种任意组合链不恢复。

启动命令、1932逐行矩阵、环境、不可变root、checkpoint来源、停止规则见 experiment.json。GPU0/2/3/4/6每卡最多两个新增本任务预测进程、source每卡一个，启动前剩余显存至少6000MiB，外部owner GPU1/5/7及其他健康任务不改。任一技术/权限/输出错误停止新入队并排空健康本任务，不盲重试；科学低分不停止。真实源checkpoint smoke首步通过即继续，无额外seal/hash链。

当前状态：PREPARED，尚未发布。代码提交、一次P0/P1、真实source smoke、归档传输、远端编译及启动后PID/CWD/argv/GPU/log证据按项目最小流程完成。
