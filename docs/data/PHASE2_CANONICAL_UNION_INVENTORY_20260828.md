# Phase2 canonical union清点（2026-08-28）

## 结论

Task 9以`equalized=1`只读扫描N607上的`ManySig.pkl`、`ManyTx.pkl`、`ManyRx.pkl`和`SingleDay.pkl`。四份资产共含1,268,812条source record，跨资产canonical union为1,139,612条物理记录；129,200条重复来源被合并，identity conflict为0，因此本次清点中的1,139,612条canonical record均标为eligible。

确定性scene/support seed为`713101`。22个new TX候选的完整排序、嵌套`Y_new5/10/20`、17个target receiver的资格和8份split计数均已由真实SQLite inventory与manifest闭合。本次工作只证明inventory与split enumeration，不创建`VALIDATED_ONCE`，也不证明固定received IQ、support/query数据验证、训练性能或模型有效性。

## 审计身份

|字段|值|
|---|---|
|Run ID|`P2_CANONICAL_UNION_AUDIT_V1_20260828`|
|审计代码提交|`3fd56b4ecd336b877c0cb74ce733d138f038b207`|
|protocol schema|`p2_min_v1`|
|source profile|`SRC5_MAXP2`|
|equalized|`1`|
|scene/support seed|`713101`|
|release SHA256|`545aca71f852dd0ccde836c29dbd6ff359fd3d2066addf01820b423d76a2de0e`|
|远端环境|用户授权范围内的`CVS-RFFI`，Python`3.10.19`、NumPy`2.2.5`|
|本地解析环境|`ssr-gpu`，Python`3.10.19`、NumPy`2.2.6`|
|remote root|`/home/szu2070436088/2510044040/CV-SincNet/runs/P2_CANONICAL_UNION_AUDIT_V1_20260828`|
|本地artifact|`E:\type10-7\local_artifacts\P2_CANONICAL_UNION_AUDIT_V1_20260828`|

## 资产与去重

`source record`按`record_sources.asset_name`计数；`preferred canonical record`按`canonical_records.preferred_asset`计数。后者是canonical记录的首选读取来源，不是新的物理样本。

|资产|source record|preferred canonical record|被合并的重复来源|
|---|---:|---:|---:|
|ManySig|288,000|288,000|0|
|ManyTx|509,128|464,328|44,800|
|ManyRx|247,684|196,884|50,800|
|SingleDay|224,000|190,400|33,600|
|合计|1,268,812|1,139,612|129,200|

|全局字段|精确值|
|---|---:|
|`source_record_count`|1,268,812|
|`canonical_record_count`|1,139,612|
|`merged_duplicate_count`|129,200|
|`conflict_count`|0|
|`eligible_record_count`|1,139,612|

## TX×RX×day覆盖

`coverage.csv`包含11,024个非空`TX×RX×day`cell，覆盖150个TX、35个RX和4天；所有cell的`record_count`之和为1,139,612，与canonical总数严格一致。

|day|canonical record|
|---|---:|
|`2021_03_01`|238,749|
|`2021_03_08`|240,654|
|`2021_03_15`|240,341|
|`2021_03_23`|419,868|
|合计|1,139,612|

下表只列profile中的17个target receiver候选。`MAXQ K1/5/10/20`和`BAL4D K1/5/10/20`按顺序给出4个资格位；`Y`表示eligible，`N`表示ineligible或不属于该policy的候选tier。

|receiver|tier|record|TX|day|非空cell|MAXQ K1/5/10/20|BAL4D K1/5/10/20|
|---|---|---:|---:|---:|---:|---|---|
|`1-1`|dense|70,479|149|4|575|Y/Y/Y/Y|Y/Y/Y/Y|
|`14-7`|dense|70,009|147|4|570|Y/Y/Y/Y|Y/Y/Y/Y|
|`2-1`|dense|70,455|149|4|581|Y/Y/Y/Y|Y/Y/Y/Y|
|`20-1`|dense|70,838|150|4|596|Y/Y/Y/Y|Y/Y/Y/Y|
|`7-14`|dense|69,045|143|4|556|Y/Y/Y/Y|Y/Y/Y/Y|
|`7-7`|dense|70,068|148|4|571|Y/Y/Y/Y|Y/Y/Y/Y|
|`8-8`|dense|69,674|146|4|560|Y/Y/Y/Y|Y/Y/Y/Y|
|`13-13`|single_day|22,400|28|1|28|Y/Y/Y/Y|N/N/N/N|
|`2-20`|single_day|22,400|28|1|28|Y/Y/Y/Y|N/N/N/N|
|`8-13`|single_day|22,400|28|1|28|Y/Y/Y/Y|N/N/N/N|
|`1-20`|many_tx|33,746|145|4|561|Y/Y/Y/Y|N/N/N/N|
|`13-7`|many_tx|34,218|146|4|584|N/N/N/N|N/N/N/N|
|`18-19`|many_tx|33,959|150|4|578|Y/Y/Y/Y|N/N/N/N|
|`19-1`|many_tx|34,364|149|4|583|Y/Y/Y/Y|N/N/N/N|
|`20-19`|many_tx|33,661|149|4|571|Y/Y/Y/Y|N/N/N/N|
|`8-14`|many_tx|35,021|148|4|588|Y/Y/Y/Y|N/N/N/N|
|`8-7`|many_tx|34,589|147|4|584|Y/Y/Y/Y|N/N/N/N|

MAXQ在4个K下均保留同一16个receiver，仅`13-7`未满足“全部26个registered TX×3个scene均至少K条”的资格公式。BAL4D只以dense tier为候选，并按`K_max=20`判定，因此4个K均保留7个dense receiver。

## 22个new TX候选排序

排序只使用eligible canonical inventory coverage与seed=`713101`的scene assignment，不读取prediction、metric或query truth。

|rank|TX|rank|TX|
|---:|---|---:|---|
|1|`11-1`|12|`3-13`|
|2|`7-11`|13|`5-5`|
|3|`10-11`|14|`6-1`|
|4|`10-7`|15|`7-10`|
|5|`11-4`|16|`8-18`|
|6|`11-7`|17|`8-3`|
|7|`15-1`|18|`13-3`|
|8|`16-16`|19|`4-11`|
|9|`2-19`|20|`3-18`|
|10|`20-12`|21|`11-17`|
|11|`20-7`|22|`1-11`|

- `Y_new5=[11-1,7-11,10-11,10-7,11-4]`
- `Y_new10=[11-1,7-11,10-11,10-7,11-4,11-7,15-1,16-16,2-19,20-12]`
- `Y_new20=[11-1,7-11,10-11,10-7,11-4,11-7,15-1,16-16,2-19,20-12,20-7,3-13,5-5,6-1,7-10,8-18,8-3,13-3,4-11,3-18]`

三组严格嵌套：`Y_new5=Y_new10[:5]`且`Y_new10=Y_new20[:10]`。

## MAXQ/BAL4D精确计数

8份manifest均注册6个old TX和`Y_new20`中的20个new TX，共26类；全部使用同一`capsule_id=536fb610302e0298fe98b4708d2e6d51eb81aef676126c01d8de6ff1a67985f2`。

|policy|K|receiver|support|query|row|split ID|
|---|---:|---:|---:|---:|---:|---|
|MAXQ_ALL_UNIQUE|1|16|1,248|424,694|425,942|`bcd52ba49651b7fa332fe85351e1c9ef211bd752a6e305a23d2a56ddab9492ce`|
|MAXQ_ALL_UNIQUE|5|16|6,240|419,702|425,942|`64b1638b6a1b29e3e7c92dd1951f9b964c7ebd8c69662c87b5b12b4c3287bace`|
|MAXQ_ALL_UNIQUE|10|16|12,480|413,462|425,942|`1c4d689f8c37c43336b9ec222ed2c7e1d76a72fe71d2610b801a558b4e70102e`|
|MAXQ_ALL_UNIQUE|20|16|24,960|400,982|425,942|`6da04fe3507ad0c23ce090fcfd0c745e08c133d318f26f9004fbe6016bdefefe`|
|BALANCED_4DAY_CORE|1|7|546|28,392|28,938|`fe9b406c9ef27adc24fae41b86b9fc45715d6ecbfdd6bd766c0d337541c0b8d8`|
|BALANCED_4DAY_CORE|5|7|2,730|28,392|31,122|`d36082ebc746797f855f84ea445baf2fa93c211a69b19931f7829bc087a2e3c7`|
|BALANCED_4DAY_CORE|10|7|5,460|28,392|33,852|`65148e866ea3becb76babab36072f936471c7ae60dcceea139142cc39a8c5523`|
|BALANCED_4DAY_CORE|20|7|10,920|28,392|39,312|`260f7bc291e8dbfe53e68f58997414a7d89c8f15b55d59793de506fb434fac25`|

### scene级计数

|policy|K|每scene support|clear query|low-elev query|rain query|
|---|---:|---:|---:|---:|---:|
|MAXQ_ALL_UNIQUE|1|416|141,572|141,547|141,575|
|MAXQ_ALL_UNIQUE|5|2,080|139,908|139,883|139,911|
|MAXQ_ALL_UNIQUE|10|4,160|137,828|137,803|137,831|
|MAXQ_ALL_UNIQUE|20|8,320|133,668|133,643|133,671|
|BALANCED_4DAY_CORE|1|182|9,464|9,464|9,464|
|BALANCED_4DAY_CORE|5|910|9,464|9,464|9,464|
|BALANCED_4DAY_CORE|10|1,820|9,464|9,464|9,464|
|BALANCED_4DAY_CORE|20|3,640|9,464|9,464|9,464|

## 本地回读与一致性

- 精确拉取12个计划文件，总计564,514,594字节；未拉取`canonical.sqlite`、release或其他run内容。
- `summary.json`、`class_selection.json`及8份manifest均以UTF-8成功解析。
- 8份manifest均满足`schema=cvs.phase2.canonical_split_manifest.v1`、`protocol_schema=p2_min_v1`和`profile_id=SRC5_MAXP2`。
- 每份manifest的声明`row/support/query/eligible`计数与实际row严格一致；物理ID在单份manifest内唯一，support/query ID互斥。
- 所有query row均不含`tx_id`字段；检测到的query truth字段数为0。
- support总数均满足`26×eligible receiver×3 scenes×K`。
- `conflicts.csv`仅含表头，conflict row为0；因此不存在需要从split排除的conflict ID。

## profile表示边界

`configs/phase2_canonical_union_profiles_v1.json`的v1 schema只承载receiver tier候选、6个old TX和22个new TX候选，不承载audit-derived selection或policy/K资格缓存。把derived字段写入JSON会超出当前schema消费面；删除`13-7`会改变排序输入与capsule语义；重排候选则会破坏现有精确profile测试而不增加运行时约束。为保持8份manifest可复现，Task 9保持profile语义和字节不变，把完整排序、`Y_new5/10/20`及eligible-by-policy/K冻结在本文件和审计报告中。该表示边界是`NONBLOCKING`，不是新的发布gate。

## 声明边界

本次审计没有构建或验证固定received IQ，没有执行Phase2 builder/validator的一次性权限检查，没有生成prediction，也没有连接truth。最高可声明状态是canonical inventory与deterministic split enumeration已核对；不得写成`VALIDATED_ONCE`、训练完成、性能结果、`ARTIFACTS_COMPLETE`或`ANALYZED`。
