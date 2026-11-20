"""Remove the unused left margin from Figure 1 without changing chart data.

After plot_figures.py, run:
  python code/apply_figure1_crop_v1.py --figures-dir <directory>
The display uses a 6.284% left crop; all data labels remain within the canvas.
"""
import argparse
from pathlib import Path
from PIL import Image
from lxml import etree

p = argparse.ArgumentParser()
p.add_argument('--figures-dir', type=Path, required=True)
a = p.parse_args()
root = a.figures_dir
fraction = 6284 / 100000
stem = 'Figure1_Procedure_profiles'
for ext in ['png','tiff']:
    path = root / f'{stem}.{ext}'
    if path.exists():
        im = Image.open(path).convert('RGB')
        cut = round(im.width * fraction)
        im = im.crop((cut, 0, im.width, im.height))
        opts = {'dpi':(300,300)}
        if ext == 'tiff': opts['compression'] = 'tiff_lzw'
        else: opts['optimize'] = True
        im.save(path, **opts)
svg = root / f'{stem}.svg'
if svg.exists():
    doc = etree.parse(str(svg))
    e = doc.getroot()
    x, y, w, h = [float(v) for v in e.attrib['viewBox'].split()]
    e.set('viewBox',f'{x+w*fraction:.5f} {y:g} {w*(1-fraction):.5f} {h:g}')
    e.set('width', f'{w*(1-fraction):.5f}pt')
    for c in list(e):
        if etree.QName(c).localname == 'metadata':e.remove(c)
    doc.write(str(svg),encoding='UTF-8',xml_declaration=True)
print('Figure 1 cropped for presentation, numeric data unchanged')
