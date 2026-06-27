#!/usr/bin/env bash
# Compile a TikZ .tex (article class, no standalone needed) -> PDF -> cropped PNG.
# Usage: build.sh name   (expects name.tex in this dir)
set -e
cd "$(dirname "$0")"
NAME="$1"
PY="/userdata/leejr/UnderlyingEventStudy/.venv/bin/python"
pdflatex -interaction=nonstopmode -halt-on-error "${NAME}.tex" >/tmp/${NAME}.log 2>&1 || { tail -25 /tmp/${NAME}.log; exit 1; }
pdftoppm -png -r 170 "${NAME}.pdf" "${NAME}_raw" >/dev/null 2>&1
RAW=$(ls ${NAME}_raw*.png | head -1)
"$PY" - "$RAW" "${NAME}.png" <<'PY'
import sys
from PIL import Image, ImageChops
src, dst = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGB")
bg = Image.new("RGB", im.size, (255, 255, 255))
diff = ImageChops.difference(im, bg)
bbox = diff.getbbox()
if bbox:
    pad = 12
    bbox = (max(0, bbox[0]-pad), max(0, bbox[1]-pad), min(im.size[0], bbox[2]+pad), min(im.size[1], bbox[3]+pad))
    im = im.crop(bbox)
im.save(dst)
print("wrote", dst, im.size)
PY
rm -f ${NAME}_raw*.png ${NAME}.aux ${NAME}.log
echo "done ${NAME}.png"
