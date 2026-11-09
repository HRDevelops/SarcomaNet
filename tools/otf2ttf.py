"""
otf2ttf.py — convert CFF-based OpenType fonts (e.g. URW Nimbus Sans) to TrueType outlines.

matplotlib embeds CFF .otf fonts with a mismatched font type in PDFs (poppler warns
"Mismatch between font type and embedded font file"); TrueType outlines embed cleanly.
Adapted from the fontTools otf2ttf snippet.

    python tools/otf2ttf.py <out_dir> font1.otf [font2.otf ...]
"""
import sys
from pathlib import Path

from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable


def otf_to_ttf(font, max_err=1.0):
    order = font.getGlyphOrder()
    font["loca"] = newTable("loca")
    font["glyf"] = glyf = newTable("glyf")
    glyf.glyphOrder = order
    glyf.glyphs = {}
    gs = font.getGlyphSet()
    for name in order:
        pen = TTGlyphPen(gs)
        gs[name].draw(Cu2QuPen(pen, max_err, reverse_direction=True))
        glyf[name] = pen.glyph()
    font["maxp"] = maxp = newTable("maxp")
    maxp.tableVersion = 0x00010000
    for k in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
              "maxStackElements", "maxSizeOfInstructions", "maxComponentElements"):
        setattr(maxp, k, 1 if k == "maxZones" else 0)
    post = font["post"]
    post.formatType = 2.0
    post.extraNames = []
    post.mapping = {}
    post.glyphOrder = order
    font["head"].glyphDataFormat = 0
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    font.sfntVersion = "\000\001\000\000"


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for f in sys.argv[2:]:
        dst = out / (Path(f).stem + ".ttf")
        if dst.exists():
            print("exists, skipped:", dst); continue
        font = TTFont(f)
        otf_to_ttf(font)
        font.save(dst)
        print("wrote", dst)


if __name__ == "__main__":
    main()
