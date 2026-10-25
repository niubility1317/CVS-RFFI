"""Build an offline D92-BranchLocalRidge-v1 lesson; reuse the local book renderer."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / "pamf_dura_interactive"


def bundle(export: Path | None = None) -> None:
    spec = importlib.util.spec_from_file_location("offline_book_vendor", BASE / "build.py")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    read = lambda p: p.read_text(encoding="utf-8")
    document = read(BASE / "index.template.html")
    document = document.replace("PAMF × DURA · 从一段 IQ 到设备身份", "D92-BranchLocalRidge-v1 · 从 Support 到新类注册")
    document = document.replace("P<span>×</span>D", "D92").replace("PAMF & DURA", "BranchLocalRidge")
    document = document.replace("从复数 IQ、卷积和梯度开始，逐步掌握 PAMF 与 DURA 的真实结构、公式推导和交互数值例子。", "从向量与矩阵开始，逐步推导 D92-BranchLocalRidge-v1 的 Phase2 特征、局部核、中心化、闭式拟合与独立推理。")
    document = document.replace("/* PAMF_CONTENT */", "/* GEOMETRY_CONTENT */").replace("/* DURA_CONTENT */", "/* RIDGE_CONTENT */")
    application = read(BASE / "app.js").replace("pamf-dura-course-20261025-v1", "d92-local-ridge-course-20261025-v1").replace("· PAMF × DURA", "· D92 BranchLocalRidge")
    assets = {
        "KATEX_CSS": helper.inline_fonts(read(helper.VENDOR / "katex.min.css")),
        "KATEX_JS": read(helper.VENDOR / "katex.min.js"),
        "KATEX_AUTO_JS": read(helper.VENDOR / "auto-render.min.js"),
        "COURSE_CSS": read(BASE / "style.css") + "\n" + read(BASE / "visuals.css"),
        "FOUNDATION_CONTENT": read(ROOT / "foundation-content.js"),
        "GEOMETRY_CONTENT": read(ROOT / "geometry-content.js"),
        "RIDGE_CONTENT": read(ROOT / "ridge-content.js"),
        "REFERENCE_CONTENT": read(ROOT / "reference-content.js"),
        "LABS_JS": read(BASE / "labs.js") + "\n" + read(ROOT / "labs.js"),
        "VISUALS_JS": read(BASE / "visuals.js") + "\n" + read(ROOT / "visuals.js"),
        "APP_JS": application,
    }
    for key, content in assets.items():
        if key.endswith("JS"):
            content = re.sub(r"//# sourceMappingURL=[^\r\n]*", "", content)
        if key not in ("KATEX_CSS", "COURSE_CSS") and "</script" in content.lower():
            raise ValueError("Unexpected script terminator in " + key)
        token = "/* " + key + " */"
        if document.count(token) != 1:
            raise ValueError("Missing unique template slot: " + key)
        document = document.replace(token, content)
    if re.search(r"<(?:script|link|iframe)[^>]+(?:src|href)=[\"']https?://", document):
        raise ValueError("Remote resource found in offline lesson")
    target = ROOT / "dist" / "index.html"
    target.parent.mkdir(exist_ok=True)
    target.write_text(document, encoding="utf-8", newline="\n")
    outputs = [str(target)]
    if export:
        export.parent.mkdir(parents=True, exist_ok=True)
        export.write_text(document, encoding="utf-8", newline="\n")
        if export.read_bytes() != target.read_bytes():
            raise ValueError("Export readback mismatch")
        outputs.append(str(export))
    print(json.dumps({"status": "BUILT", "bytes": target.stat().st_size, "outputs": outputs}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path)
    bundle(parser.parse_args().export)
