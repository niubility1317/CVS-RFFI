from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT = Path(r"E:\type10-7\analysis\fewshot_cvs_training_strategy_20260610.png")
W, H = 2200, 1260


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    path = r"C:\Windows\Fonts\msyhbd.ttc" if bold else r"C:\Windows\Fonts\msyh.ttc"
    return ImageFont.truetype(path, size)


img = Image.new("RGB", (W, H), "#F7FAFC")
d = ImageDraw.Draw(img)

F_TITLE = get_font(52, True)
F_SUB = get_font(26)
F_STEP = get_font(29, True)
F_BODY = get_font(23)
F_SMALL = get_font(20)
F_TAG = get_font(22, True)

NAVY = "#17324D"
TEAL = "#0F766E"
BLUE = "#2563EB"
ORANGE = "#D97706"
GREEN = "#15803D"
RED = "#B91C1C"
GRAY = "#475569"
LINE = "#CBD5E1"


def rounded_rect(x0, y0, x1, y1, fill, outline=None, width=2, r=24):
    d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill, outline=outline, width=width)


def wrap(text: str, max_width: int, fnt: ImageFont.FreeTypeFont):
    out = []
    for para in text.split("\n"):
        cur = ""
        for ch in para:
            test = cur + ch
            if d.textlength(test, font=fnt) <= max_width or not cur:
                cur = test
            else:
                out.append(cur)
                cur = ch
        if cur:
            out.append(cur)
    return out


def draw_wrapped(text, xy, width, fnt, fill=GRAY, line_spacing=8):
    x, y = xy
    for ln in wrap(text, width, fnt):
        d.text((x, y), ln, font=fnt, fill=fill)
        bbox = d.textbbox((0, 0), ln, font=fnt)
        y += (bbox[3] - bbox[1]) + line_spacing
    return y


def center_text(text, box, fnt, fill=NAVY):
    x0, y0, x1, y1 = box
    bbox = d.textbbox((0, 0), text, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((x0 + (x1 - x0 - tw) / 2, y0 + (y1 - y0 - th) / 2 - 2), text, font=fnt, fill=fill)


def arrow(x0, y0, x1, y1, color="#64748B", width=4):
    import math

    d.line([x0, y0, x1, y1], fill=color, width=width)
    ang = math.atan2(y1 - y0, x1 - x0)
    L = 18
    for a in (math.pi * 0.84, -math.pi * 0.84):
        x = x1 + L * math.cos(ang + a)
        y = y1 + L * math.sin(ang + a)
        d.line([x1, y1, x, y], fill=color, width=width)


rounded_rect(70, 45, 2130, 155, "#EEF6FF", outline="#D7E8FA", r=28)
d.text((100, 64), "少样本 CVS 训练策略：先稳身份，再轻量去域", font=F_TITLE, fill=NAVY)
d.text((104, 125), "适用于 K5/K10/K20 等低样本 RFFI / CV-SincNet 场景", font=F_SUB, fill=GRAY)

cards = [
    ("1", BLUE, "少样本抽样", "固定验证/测试协议\nK-shot 或 per-combo cap\n多 seed 记录方差"),
    ("2", TEAL, "主干预训练", "先让 TX 分类站稳\n优先保留 identity 信息\n不要过早强正则"),
    ("3", ORANGE, "弱去域约束", "zid 保持 TX 指纹\nzdom 吸收 receiver/day\nGRL、domain loss 轻量"),
    ("4", GREEN, "轻量稳健项", "K 足够后再开\nprototype / SupCon\nconsistency / norm guard"),
    ("5", RED, "选择 checkpoint", "看 best 而非 final\noverall + strict UDU\nworst-rx / rollback 审核"),
]

x0, y0, cw, ch, gap = 80, 235, 360, 225, 55
for idx, (num, color, title, body) in enumerate(cards):
    x = x0 + idx * (cw + gap)
    rounded_rect(x, y0, x + cw, y0 + ch, "#FFFFFF", outline=LINE, r=24)
    rounded_rect(x + 20, y0 + 20, x + 78, y0 + 78, color, r=16)
    center_text(num, (x + 20, y0 + 20, x + 78, y0 + 78), F_TAG, fill="white")
    d.text((x + 96, y0 + 27), title, font=F_STEP, fill=NAVY)
    draw_wrapped(body, (x + 28, y0 + 98), cw - 56, F_BODY)
    if idx < len(cards) - 1:
        arrow(x + cw + 12, y0 + 112, x + cw + gap - 13, y0 + 112)

rounded_rect(160, 550, 2040, 820, "#FFFFFF", outline=LINE, r=30)
d.text((210, 582), "CVS 的核心分工", font=F_STEP, fill=NAVY)
d.text((210, 624), "低样本下不要让域约束压掉 TX 身份边界；训练顺序比堆损失更重要。", font=F_BODY, fill=GRAY)

rounded_rect(250, 705, 925, 780, "#EAF7F5", outline="#B9E4DC", r=20)
d.text((285, 726), "zid：身份特征", font=F_STEP, fill=TEAL)
d.text((530, 733), "保留 TX fingerprint，用于主分类", font=F_BODY, fill=GRAY)

rounded_rect(1250, 705, 1930, 780, "#FFF7E8", outline="#F1D4A4", r=20)
d.text((1285, 726), "zdom：域特征", font=F_STEP, fill=ORANGE)
d.text((1515, 733), "吸收 receiver / day / channel nuisance", font=F_BODY, fill=GRAY)

arrow(925, 742, 1250, 742, color="#94A3B8", width=4)
d.text((1030, 700), "orth / GRL", font=F_SMALL, fill=GRAY)
d.text((1036, 728), "轻量解耦", font=F_SMALL, fill=GRAY)

bottom_y = 885
boxes = [
    (90, BLUE, "低 K：K5/K10", "TX CE + 弱 domain/GRL + norm guard。\nprototype、Fishr、强 satellite 先关或延后。"),
    (820, TEAL, "中 K：K20/K30", "逐步恢复 GroupCE、prototype、consistency。\n监控 train acc 是否被正则压低。"),
    (1550, GREEN, "高 K / ratio=0.1", "回到完整 CVS/CEN51 正则栈。\n用 strict UDU、worst-rx 和 rollback 共同选模型。"),
]
for x, color, title, body in boxes:
    rounded_rect(x, bottom_y, x + 600, bottom_y + 205, "#FFFFFF", outline=LINE, r=24)
    d.text((x + 35, bottom_y + 34), title, font=F_STEP, fill=color)
    draw_wrapped(body, (x + 35, bottom_y + 88), 530, F_BODY)

rounded_rect(70, 1180, 2130, 1224, "#ECFDF5", outline="#BBF7D0", r=16)
center_text(
    "一句话：少样本 CVS 先学稳 TX identity，再用弱域约束提升跨 receiver/day 泛化，最后按 best checkpoint 而不是 final epoch 做选择。",
    (90, 1180, 2110, 1224),
    F_SMALL,
    fill="#166534",
)

img.save(OUT)
print(OUT)
