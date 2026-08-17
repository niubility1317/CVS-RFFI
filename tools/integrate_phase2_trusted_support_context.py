from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree

from normalize_phase2_report_typography import normalize_typography


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}
DOCUMENT_XML = "word/document.xml"
HEADING2_STYLE = "4"
BODY_TEXT_STYLE = "19"
RED = "FF0000"


def qn(local: str) -> str:
    return f"{{{W_NS}}}{local}"


def paragraph_text(paragraph: etree._Element) -> str:
    return "".join(paragraph.xpath(".//w:t/text()", namespaces=NS))


def make_run(text: str, *, bold: bool = False, color: str | None = None) -> etree._Element:
    run = etree.Element(qn("r"))
    if bold or color:
        properties = etree.SubElement(run, qn("rPr"))
        if bold:
            etree.SubElement(properties, qn("b"))
            etree.SubElement(properties, qn("bCs"))
        if color:
            color_element = etree.SubElement(properties, qn("color"))
            color_element.set(qn("val"), color)
    text_element = etree.SubElement(run, qn("t"))
    if text[:1].isspace() or text[-1:].isspace():
        text_element.set(f"{{{XML_NS}}}space", "preserve")
    text_element.text = text
    return run


def make_paragraph(
    style_id: str,
    runs: list[tuple[str, bool, str | None]],
) -> etree._Element:
    paragraph = etree.Element(qn("p"))
    properties = etree.SubElement(paragraph, qn("pPr"))
    style = etree.SubElement(properties, qn("pStyle"))
    style.set(qn("val"), style_id)
    for text, bold, color in runs:
        paragraph.append(make_run(text, bold=bold, color=color))
    return paragraph


def heading(text: str) -> etree._Element:
    return make_paragraph(HEADING2_STYLE, [(text, False, None)])


def body(text: str) -> etree._Element:
    return make_paragraph(BODY_TEXT_STYLE, [(text, False, None)])


def labeled_body(label: str, text: str) -> etree._Element:
    return make_paragraph(
        BODY_TEXT_STYLE,
        [(label, True, None), (text, False, None)],
    )


def emphasis(text: str) -> etree._Element:
    return make_paragraph(BODY_TEXT_STYLE, [(text, True, RED)])


INSERTED_PARAGRAPHS = [
    heading("1.1研究动机：为什么在卫星上做RFFI"),
    body(
        "射频指纹识别（Radio-Frequency Fingerprint Identification，RFFI）与网络认证回答的是两个不同问题。"
        "网络认证确认终端是否持有合法凭据或获得运营授权；RFFI判断当前波形携带的硬件失真是否与已登记的物理射频（Radio Frequency，RF）发射链一致。"
        "同一逻辑身份可能因功率放大器（Power Amplifier，PA）、振荡器或RF模组维修而对应新的物理发射链；同一物理发射链也可能因账号或业务关系变化而绑定新的逻辑身份。"
        "因此，CVS把RFFI定位为物理一致性辅助证据，不替代密码学认证，也不把一次分类结果直接视为最终身份确权。"
    ),
    body(
        "非地面网络（Non-Terrestrial Network，NTN）中的同一终端会被不同卫星或不同接收载荷连续观测。"
        "接收同相/正交（In-phase/Quadrature，I/Q）信号不仅包含发射机硬件特征，还混合传播信道以及目标接收机的低噪声放大、滤波、混频、自动增益控制（Automatic Gain Control，AGC）、"
        "模数转换器（Analog-to-Digital Converter，ADC）和采样时钟误差。"
        "地面训练即使降低了平均域敏感性，也无法穷尽一颗未见目标卫星接收前端的实际响应；这使“地面获得跨域先验、目标接收机完成局部专门化”成为独立于普通闭集分类的研究问题。"
    ),
    body(
        "星上计算的合理目标不是重训完整主干网络。Phase1继续在地面完成大规模训练、模型选择和治理；Phase2只在目标接收机侧提取特征，并利用少量支持集样本"
        "（support，指允许建立或更新模型状态的带标签目标域样本）更新prototype、adapter或校准状态。"
        "当原始I/Q持续下传受带宽限制、决策必须在下一次地面链路前完成、馈电链路间歇可用，或系统需要减少原始波形集中存储时，receiver-local轻量适配具有明确工程价值。"
        "若地面链路充足，则应保留地面辅助适配作为基线。本文假设目标平台是具有数字处理和星上计算能力的再生式载荷或专用RF感知载荷，不把仅执行模拟转发的透明载荷默认为可运行Phase2。"
    ),
    emphasis(
        "项目重点不是把普通设备分类器搬到卫星，而是处理未见目标接收前端造成的残余域偏移，并在物理RF身份库持续扩展时保持旧类与新类的统一识别。"
    ),
    heading("1.2CVS具体场景：地面训练、星上部署与受控注册"),
    body(
        "CVS的主场景是受认证NTN终端在数字或再生式卫星接收载荷上的受控注册。"
        "运营网络负责逻辑身份、授权和审计，目标卫星负责采集经过自身接收链的I/Q并执行轻量域校准或新类注册；日常业务信号只用于冻结模型后的识别。"
        "整个生命周期分为以下五个环节。"
    ),
    labeled_body(
        "环节1（Phase1地面建模）：",
        "使用多个source receiver中的少量发射机标签和大量无标签样本学习跨接收机先验，随后封存不可变deployment bundle。Phase1不读取未来目标卫星数据。",
    ),
    labeled_body(
        "环节2（Stage2-A零标签参考）：",
        "模型部署到target receiver后，可用无标签目标域观测进行零标签诊断或参考，但不得据此声明少样本旧类适应或新类注册成功。",
    ),
    labeled_body(
        "环节3（Stage2-B目标接收域校准）：",
        "在卫星commissioning、新接收链启用或计划校准窗口内，已登记旧终端先通过网络或运营流程确认身份，再发送每类K个独立物理burst。目标卫星只用这些target-old support校准旧类在新接收域中的残余偏移。",
    ),
    labeled_body(
        "环节4（Stage2-C新物理RF链注册）：",
        "新终端或维修后更换RF模组的终端完成外部身份核验和注册授权后，在单独的受控窗口重新发送K个独立burst。目标卫星将其作为新类support，并让旧类与已注册新类在同一输出空间竞争。",
    ),
    labeled_body(
        "环节5（日常识别与异常处置）：",
        "正常业务burst属于query（即模型冻结后用于独立评价或识别、不得更新predictor的观测）。无法由注册库可靠解释的信号只能标记为unknown或defer；它需要外部调查和确权，不能自动转成Phase2 support。",
    ),
    heading("1.3Phase2可信标签与K-shot support如何获取"),
    body(
        "本文所称可信标签（trusted label，指由模型外部的网络认证、运营记录、硬件维护记录和注册授权共同确认，并绑定到具体物理RF链、接收事件、目标receiver和有效时间窗的类别标签）不是算法从无标签信号中自行预测出的语义身份。"
        "Phase2发生在运营方主动创建的受控采集事件中，而不是从普通业务query中寻找“碰巧有标签”的样本。"
    ),
    emphasis(
        "可信support形成链：逻辑身份与设备记录核验→签发受控采集窗口→K个独立物理burst经过目标卫星接收链→质量与事件完整性检查→Stage2-B旧类校准或Stage2-C新类注册。"
    ),
    labeled_body(
        "步骤1（身份与权限核验）：",
        "运营方首先确认逻辑身份、设备所有权或任务身份，并核对当前物理RF链的设备序列号、安装记录和维护状态。网络认证只建立逻辑信任；物理RF链类别还需要受控采集完成绑定。",
    ),
    labeled_body(
        "步骤2（签发registration episode）：",
        "注册事件由运营方限定时间、频率、波束、波形或挑战序列，并写入一次性事件标识。该窗口是CVS提出的研究型RFF enrollment扩展，不应描述为现有NTN标准已经规定的RFFI流程。",
    ),
    labeled_body(
        "步骤3（采集K个独立物理事件）：",
        "终端在窗口内产生K次相互独立的物理发射burst，每次只形成一份固定target-domain接收I/Q。同一长burst的切片、FFT、均衡或增强view仍然只计一个shot，support与query的物理事件必须互斥。",
    ),
    labeled_body(
        "步骤4（完整性与质量检查）：",
        "系统记录support event ID、receiver ID、时间窗、信号质量、设备上下文和数据hash，排除重放、窗口外信号、重复物理样本和质量不合格样本。高价值场景还可要求两个时间窗口或两个receiver交叉确认。",
    ),
    labeled_body(
        "步骤5（按身份生命周期使用）：",
        "历史已登记物理RF链产生的support用于Stage2-B；经外部流程批准进入身份库的新RF链产生的fresh support用于Stage2-C。若维修后逻辑账号不变但PA、振荡器或RF模组已更换，应把它作为新的物理RF链重新注册。",
    ),
    body("Phase2可信support可由以下受控业务来源产生，可信度取决于标签与具体物理RF链的绑定强度，而不取决于协议字段本身是否存在。"),
    labeled_body(
        "主场景——受认证NTN终端注册：",
        "终端完成网络身份认证和运营授权后进入专用registration window，按调度发送K个独立burst。该来源同时适合旧类校准和新终端注册。",
    ),
    labeled_body(
        "高可信备选——TT&C或已登记地面站计划上行：",
        "任务控制方在指定通信pass中核对地面站、功放和调制链维护记录，再安排受控registration waveform。该场景物理RF链清晰，但类别规模通常较小。",
    ),
    labeled_body(
        "维护场景——RF模组更换后的受控复测：",
        "维修记录提供逻辑身份与硬件变化证据，复测窗口重新采集target-domain burst，用于建立新的物理RF链类别并设置有效期。",
    ),
    labeled_body(
        "外部调查场景——监管或第三方确权：",
        "频谱监测、定位或现场调查只能先形成候选身份。只有外部流程完成物理设备确认和注册授权后，系统重新采集独立fresh support，才可交给Stage2-C；历史unknown query保持为不可变检测证据。",
    ),
    labeled_body(
        "不充分来源——协议自报ID或模型伪标签：",
        "协议字段可用于候选匹配，模型置信度可用于defer或调查排序，但二者都不能单独生成可信新类标签，也不能绕过运营授权。",
    ),
    emphasis(
        "边界：可信标签不是AI自己产生的标签；身份认证成功不等于RF样本天然可信；unknown query不得事后改成support。只有外部确权与注册授权完成后重新采集的独立target-domain样本，才进入正式Phase2更新。"
    ),
]


HEADING_RENAMES = {
    "1.1RFFI任务的三轴谱系": "1.4RFFI任务的三轴谱系",
    "1.2少样本学习的定义": "1.5少样本学习的定义",
    "1.3域适应及Stage2-B定位": "1.6域适应及Stage2-B定位",
    "1.4类增量、FSCIL与新类注册": "1.7类增量、FSCIL与新类注册",
    "1.5与CVS-RFFI Phase2阶段的对应": "1.8与CVS-RFFI Phase2阶段的对应",
    "1.6样本角色与成功条件": "1.9样本角色与成功条件",
}


def rename_heading(paragraph: etree._Element, new_text: str) -> None:
    text_nodes = paragraph.xpath(".//w:t", namespaces=NS)
    if not text_nodes:
        raise RuntimeError("heading has no text node")
    text_nodes[0].text = new_text
    for node in text_nodes[1:]:
        node.text = ""


def patch_document_xml(xml: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(xml, parser)
    paragraphs = root.xpath(".//w:body/w:p", namespaces=NS)
    by_text = {paragraph_text(paragraph): paragraph for paragraph in paragraphs}

    missing = [text for text in HEADING_RENAMES if text not in by_text]
    if missing:
        raise RuntimeError(f"missing expected headings: {missing}")

    target = by_text["1.1RFFI任务的三轴谱系"]
    for new_paragraph in INSERTED_PARAGRAPHS:
        target.addprevious(etree.fromstring(etree.tostring(new_paragraph)))

    for old_text, new_text in HEADING_RENAMES.items():
        rename_heading(by_text[old_text], new_text)

    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def integrate_context(source: Path | str, output: Path | str) -> None:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    if source_path == output_path:
        raise ValueError("source and output must differ")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tempdir:
        staged_path = Path(tempdir) / "context_staged.docx"
        with ZipFile(source_path, "r") as source_archive, ZipFile(
            staged_path, "w", compression=ZIP_DEFLATED
        ) as output_archive:
            for info in source_archive.infolist():
                data = source_archive.read(info.filename)
                if info.filename == DOCUMENT_XML:
                    data = patch_document_xml(data)
                output_archive.writestr(info, data)
        normalize_typography(staged_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add the Phase2 trusted-support provenance and satellite scenario sections"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    integrate_context(args.source, args.output)


if __name__ == "__main__":
    main()
