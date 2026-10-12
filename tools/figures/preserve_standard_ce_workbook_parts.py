"""Apply the artifact-exported figure-3 image while retaining all unrelated XLSX parts.

The artifact importer normalizes workbook styles on export. This package assembly
uses only its newly authored image and extent, preserving the original formulas,
styles, charts, relationships and all other figures byte for byte.
"""
import argparse
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET
import zipfile

XDR='http://schemas.openxmlformats.org/drawingml/2006/spreadsheetDrawing'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'

def image_part(archive,anchor):
    rid=anchor.find('.//{'+A+'}blip').attrib['{'+R+'}embed']
    relationships=ET.fromstring(archive.read('xl/drawings/_rels/drawing2.xml.rels'))
    target=next(e.attrib['Target'] for e in relationships if e.attrib['Id']==rid)
    # Resolve the declared package part, including root-relative OOXML targets.
    parts=[]
    declared=PurePosixPath(target.lstrip('/')) if target.startswith('/') else PurePosixPath('xl/drawings')/target
    for part in declared.parts:
        if part=='..':parts.pop()
        elif part!='.':parts.append(part)
    name='/'.join(parts)
    if not name.startswith('xl/media/') or not name.endswith('.png'):raise ValueError('Unexpected image part')
    return name

def apply(original,artifact_export,output):
    if output.exists():raise FileExistsError(output)
    drawing='xl/drawings/drawing2.xml'
    with zipfile.ZipFile(original) as old,zipfile.ZipFile(artifact_export) as authored:
        oldbytes=old.read(drawing);root=ET.fromstring(oldbytes)
        newroot=ET.fromstring(authored.read(drawing))
        if len(root)!=6 or len(newroot)!=6:raise ValueError('Expected six paper images')
        anchor,newanchor=root[2],newroot[2]
        if ET.tostring(anchor.find('{'+XDR+'}from'))!=ET.tostring(newanchor.find('{'+XDR+'}from')):
            raise ValueError('Figure origin changed')
        extent=anchor.find('{'+XDR+'}ext');newextent=newanchor.find('{'+XDR+'}ext')
        if extent.attrib['cx']!=newextent.attrib['cx']:raise ValueError('Figure width changed')
        before=extent.attrib['cy'].encode('ascii');after=newextent.attrib['cy'].encode('ascii')
        if oldbytes.count(before)!=1:raise ValueError('Ambiguous original figure height')
        changes={drawing:oldbytes.replace(before,after),
                 image_part(old,anchor):authored.read(image_part(authored,newanchor))}
        with zipfile.ZipFile(output,'x') as saved:
            for info in old.infolist():saved.writestr(info,changes.get(info.filename,old.read(info.filename)))
    with zipfile.ZipFile(original) as old,zipfile.ZipFile(output) as saved:
        if set(old.namelist())!=set(saved.namelist()) or saved.testzip() is not None:raise ValueError('Package mismatch')
        changed=[n for n in old.namelist() if old.read(n)!=saved.read(n)]
        if set(changed)!=set(changes):raise ValueError('Unrelated workbook part changed')
    print('VERIFIED: only the artifact-generated figure-3 image and its height changed; all other workbook parts preserved.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original',type=Path);parser.add_argument('artifact_export',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();apply(args.original,args.artifact_export,args.output)
