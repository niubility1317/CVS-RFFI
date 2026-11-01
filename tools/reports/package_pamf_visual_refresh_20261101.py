"""Package the white-background PAMF figure catalog and illustration drafts."""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw


def atlas(figures, source, destination):
    document = fitz.open()
    toc = []
    for index, figure in enumerate(figures, 1):
        with fitz.open(source / (figure['id'] + '.pdf')) as src:
            rect = src[0].rect
            page = document.new_page(width=rect.width, height=rect.height + 24)
            page.show_pdf_page(fitz.Rect(0, 0, rect.width, rect.height), src, 0)
            caption = f"图 {index} | {figure.get('short_title', figure['title'])}"
            page.insert_text((12, rect.height + 16), caption, fontname='china-s',
                             fontsize=8, color=(0.32, 0.37, 0.41))
            toc.append([1, f"{index}. {figure.get('short_title', figure['title'])}", index])
        if index % 25 == 0:
            print('ATLAS', destination.name, index, flush=True)
    document.set_toc(toc)
    document.set_metadata({'title': 'PAMF figure atlas', 'author': 'PAMF research'})
    document.save(destination, garbage=3, deflate=True)
    document.close()
    with fitz.open(destination) as check:
        if len(check) != len(figures):
            raise RuntimeError(f'Wrong page count: {destination}')


def contact_sheets(figures, png_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    cell_w, cell_h, per_sheet = 460, 320, 25
    for first in range(0, len(figures), per_sheet):
        canvas = Image.new('RGB', (5 * cell_w, 5 * cell_h), 'white')
        draw = ImageDraw.Draw(canvas)
        for offset, fig in enumerate(figures[first:first + per_sheet]):
            with Image.open(png_dir / (fig['id'] + '.png')) as source:
                im = source.convert('RGB')
                im.thumbnail((cell_w - 12, cell_h - 30), Image.Resampling.LANCZOS)
                x, y = (offset % 5) * cell_w, (offset // 5) * cell_h
                canvas.paste(im, (x + (cell_w - im.width) // 2, y + 24))
                draw.text((x + 6, y + 5), f"{first + offset + 1} | {fig['id']}", fill='#354455')
        canvas.save(output / f'contact_{first + 1:03d}.jpg', quality=91)


ART_NAMES = ['white_system', 'white_pamf_fan', 'white_pamf_ribbons',
             'white_pamf_layers', 'white_dura']


def art_atlas(art, destination):
    names = ART_NAMES
    document = fitz.open()
    for name in names:
        png = art / 'png' / (name + '.png')
        with Image.open(png) as image:
            width, height = image.size
        page = document.new_page(width=720, height=720 * height / width + 36)
        page.insert_image(fitz.Rect(0, 0, 720, 720 * height / width), filename=str(png))
        page.insert_text((12, page.rect.height - 15), name + ' | visual concept',
                         fontsize=10, color=(0.22, 0.28, 0.33))
    document.save(destination, garbage=3, deflate=True)
    document.close()
    with fitz.open(destination) as check:
        if len(check) != len(names):
            raise RuntimeError('Wrong art atlas page count')


def add_tree(archive, source, prefix):
    for path in sorted(source.rglob('*')):
        if path.is_file():
            archive.write(path, str(Path(prefix) / path.relative_to(source)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source, args.output
    paper, art = output / 'paper', output / 'illustrations'
    figures = json.loads((source / 'catalog.json').read_text(encoding='utf-8'))['figures']
    selected = [f for f in figures if f['selected']]
    if len(figures) != 155 or len(selected) != 50:
        raise RuntimeError('Unexpected figure catalog size')

    print('BUILD selected atlas', flush=True)
    atlas(selected, paper / 'pdf', output / '已选实验图表_白底优化版.pdf')
    print('BUILD full atlas', flush=True)
    atlas(figures, paper / 'pdf', output / '全部实验图表_白底优化版.pdf')
    contact_sheets(figures, paper / 'png', output / '_review')
    art_atlas(art, output / '白底航天风格插图_五版预览.pdf')

    for filename in ['全部图表说明_第三版.pdf', '全部图表说明_第三版.md',
                     '图表索引.csv', 'catalog.json', 'explanation_context.json']:
        shutil.copy2(source / filename, output / filename)
    for folder in ['context_data', 'tables']:
        shutil.copytree(source / folder, output / folder, dirs_exist_ok=True)

    note = '''# PAMF 全部实验图表：白底优化版

## 使用方式

- 《已选实验图表_白底优化版.pdf》包含 50 张已选图；《全部实验图表_白底优化版.pdf》包含完整 155 张。页序与第三版目录一致，同一实验的跨接收机和双跨图保持相邻。
- `paper/pdf`、`paper/svg`、`paper/png` 分别是每张图的 PDF、SVG 和 PNG；`paper/source_data` 为每张图使用的数据。按 `图表索引.csv` 中的文件标识查找。
- 《全部图表说明_第三版.pdf》集中保留各图数据组成、实验设置、数据来源、结果和限制，`context_data` 提供原实验上下文，`tables` 保留已有表格。图注、分组和指标口径沿用原版；优先按图 ID 对照，不能只凭页码替换历史图号。
- 实验图保持白底和原有方法颜色；本次主要调整字体、坐标轴、网格线等阅读细节，并将热图改为独立矢量色块。数据、实验条件、统计口径与系列配色沿用原目录，不新增训练或预测。
- 矢量 SVG 的文字可单独编辑。少数由编辑软件替换的字体或页面缩放可能改变排版，投稿前请按最终栏宽再次检查。

## 插图说明

《白底航天风格插图_五版预览.pdf》及插图包内的 PNG、SVG 是生成式视觉草图，展示地面发射、星地链路以及不同网络分支的视觉风格。三张 PAMF 分支图分别采用扇形、曲线汇流和分层布局；另有 DURA 和系统流程各一版。插图中的波形、曲线和特征图均为示意，不是真实测量数据。SVG 为对生成图的路径追踪，含大量细小路径，并非按模块语义分层的原生结构图。

这些图尚未逐项对齐真实算法。`white_pamf_fan`、`white_pamf_ribbons` 和 `white_pamf_layers` 没有完整表达三视图投影、固定时序重排、融合方式和分类头；图中输出圆点数量也不代表真实类别数。`white_dura` 需要重核粗恢复、作用估计和两阶段的共享输入与 CG 连接；`white_system` 中两条推理路线的关系和最终分类节点仍需明确。使用前须依据 PAMF 和 DURA 的实际代码与定义重新核对节点、连接、文字及输入输出，不能直接作为方法结构图定稿。

## 已完成的校验

全部 155 张图的 PNG、PDF 和 SVG 文件齐全；逐图数据与第三版原数据一致；全部 SVG 未嵌入位图且含可编辑文本。随机抽查了网络比较、参数、热图、跨场景曲线和 DURA 增益图。以上检查不替代逐图学术审稿。
'''
    (output / '图表与插图使用说明.md').write_text(note, encoding='utf-8')
    (art / '插图结构与矢量化说明.md').write_text(note.split('## 插图说明', 1)[1].split('## 已完成的校验', 1)[0], encoding='utf-8')

    figure_zip = output / 'PAMF_全部155张实验图表_白底优化版.zip'
    with zipfile.ZipFile(figure_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=5, allowZip64=True) as archive:
        for filename in ['已选实验图表_白底优化版.pdf', '全部实验图表_白底优化版.pdf',
                         '全部图表说明_第三版.pdf', '全部图表说明_第三版.md',
                         '图表索引.csv', 'catalog.json', 'explanation_context.json',
                         '图表与插图使用说明.md']:
            archive.write(output / filename, filename)
        for sub in ['pdf', 'svg', 'png', 'source_data']:
            add_tree(archive, paper / sub, 'paper/' + sub)
        for sub in ['context_data', 'tables']:
            add_tree(archive, output / sub, sub)

    art_zip = output / 'PAMF_DURA_白底航天风格插图_五版.zip'
    with zipfile.ZipFile(art_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=5, allowZip64=True) as archive:
        archive.write(output / '白底航天风格插图_五版预览.pdf', '白底航天风格插图_五版预览.pdf')
        archive.write(art / '插图结构与矢量化说明.md', '插图结构与矢量化说明.md')
        for sub in ['png', 'svg', 'metadata']:
            suffix = '.json' if sub == 'metadata' else '.' + sub
            for name in ART_NAMES:
                archive.write(art / sub / (name + suffix), sub + '/' + name + suffix)
    for z in [figure_zip, art_zip]:
        with zipfile.ZipFile(z) as archive:
            bad = archive.testzip()
            if bad:
                raise RuntimeError(f'Corrupt archive member: {bad}')
            print('VERIFIED', z.name, len(archive.infolist()), z.stat().st_size, flush=True)
    print('DONE', len(figures), len(selected), flush=True)


if __name__ == '__main__':
    main()
