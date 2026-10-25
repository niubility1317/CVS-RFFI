"""Bundle this source-grounded tutorial into one offline HTML file (stdlib only)."""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import re
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parent
VERSION = "0.16.22"
VENDOR = ROOT / "vendor" / "katex"


def fetch_vendor() -> None:
    url = f"https://registry.npmjs.org/katex/-/katex-{VERSION}.tgz"
    with urllib.request.urlopen(url, timeout=45) as response:
        archive = response.read()
    selected = {
        "package/dist/katex.min.js": "katex.min.js",
        "package/dist/katex.min.css": "katex.min.css",
        "package/dist/contrib/auto-render.min.js": "auto-render.min.js",
        "package/LICENSE": "LICENSE",
    }
    count = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as source:
        for item in source.getmembers():
            relative = selected.get(item.name)
            if item.name.startswith("package/dist/fonts/") and item.name.endswith(".woff2"):
                relative = "fonts/" + Path(item.name).name
            if relative is None or not item.isfile():
                continue
            destination = VENDOR / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            stream = source.extractfile(item)
            if stream is None:
                raise ValueError("Missing package member: " + item.name)
            destination.write_bytes(stream.read())
            count += 1
    (VENDOR / "PROVENANCE.txt").write_text(
        f"KaTeX {VERSION}\nSource: {url}\n"
        "Official browser guidance: https://katex.org/docs/browser.html\n"
        "MIT license reproduced in LICENSE. WOFF2 assets retain their original notices.\n"
        "The tutorial embeds these files locally; it makes no CDN requests.\n",
        encoding="utf-8",
    )
    if not (VENDOR / "katex.min.js").is_file() or count < 20:
        raise ValueError("Incomplete KaTeX browser distribution")
    print(json.dumps({"vendor_version": VERSION, "files": count}))


def inline_fonts(css: str) -> str:
    css = re.sub(r',url\([^)]*\.(?:woff|ttf)\)\s*format\("[^"]+"\)', "", css)

    def replace(match: re.Match[str]) -> str:
        filename = match.group(1)
        content = (VENDOR / "fonts" / filename).read_bytes()
        encoded = base64.b64encode(content).decode("ascii")
        return "url(data:font/woff2;base64," + encoded + ")"

    css = re.sub(r"url\(fonts/([^)]+\.woff2)\)", replace, css)
    if "url(fonts/" in css:
        raise ValueError("A formula font still requires an external file")
    return css


def bundle(export: Path | None) -> None:
    document = (ROOT / "index.template.html").read_text(encoding="utf-8")
    assets = {
        "KATEX_CSS": inline_fonts((VENDOR / "katex.min.css").read_text(encoding="utf-8")),
        "COURSE_CSS": (ROOT / "style.css").read_text(encoding="utf-8") + "\n" + (ROOT / "visuals.css").read_text(encoding="utf-8"),
        "KATEX_JS": (VENDOR / "katex.min.js").read_text(encoding="utf-8"),
        "KATEX_AUTO_JS": (VENDOR / "auto-render.min.js").read_text(encoding="utf-8"),
        "FOUNDATION_CONTENT": (ROOT / "foundation-content.js").read_text(encoding="utf-8"),
        "PAMF_CONTENT": (ROOT / "pamf-content.js").read_text(encoding="utf-8"),
        "DURA_CONTENT": (ROOT / "dura-content.js").read_text(encoding="utf-8"),
        "REFERENCE_CONTENT": (ROOT / "reference-content.js").read_text(encoding="utf-8"),
        "LABS_JS": (ROOT / "labs.js").read_text(encoding="utf-8"),
        "VISUALS_JS": (ROOT / "visuals.js").read_text(encoding="utf-8"),
        "APP_JS": (ROOT / "app.js").read_text(encoding="utf-8"),
    }
    for marker, content in assets.items():
        if marker.endswith("JS"):
            content = re.sub(r"//# sourceMappingURL=[^\r\n]*", "", content)
        if marker not in ("KATEX_CSS", "COURSE_CSS") and "</script" in content.lower():
            raise ValueError("Unexpected script terminator in " + marker)
        token = "/* " + marker + " */"
        if document.count(token) != 1:
            raise ValueError("Unique replacement slot required: " + marker)
        document = document.replace(token, content)
    if re.search(r"<(?:script|link|iframe)[^>]+(?:src|href)=[\"']https?://", document):
        raise ValueError("Offline document must not request remote scripts or styles")
    target = ROOT / "dist" / "index.html"
    target.parent.mkdir(exist_ok=True)
    target.write_text(document, encoding="utf-8", newline="\n")
    outputs = [str(target)]
    if export is not None:
        export.parent.mkdir(parents=True, exist_ok=True)
        export.write_text(document, encoding="utf-8", newline="\n")
        if export.read_bytes() != target.read_bytes():
            raise ValueError("Export readback differs from the bundled HTML")
        outputs.append(str(export))
    print(json.dumps({"status": "BUILT", "bytes": target.stat().st_size, "outputs": outputs}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-vendor", action="store_true")
    parser.add_argument("--vendor-only", action="store_true")
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    if args.fetch_vendor:
        fetch_vendor()
    if not args.vendor_only:
        bundle(args.export)
