"""EasyCheck card geometry. All drawing/reading coordinates use a 1500x700 reference raster.
Physical dimensions: 150 x 70 mm; orange header 20 mm, left/right/bottom borders 2 mm.
"""
from __future__ import annotations
import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1500, 700
HEADER = 200
BORDER = 20
ORANGE_RGB = (235, 123, 24)
MARKER_X, MARKER_Y, MARKER_SIZE = 43, 25, 132
BUBBLES = [(782 + 171*i, 120) for i in range(4)]
BUBBLE_R = 53
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ROOT = Path(__file__).resolve().parent
FONT_REG = next((p for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/TTF/DejaVuSans.ttf'] if Path(p).is_file()), None)
FONT_BOLD = next((p for p in ['/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf'] if Path(p).is_file()), None)


def font(size: int, bold: bool = False):
    path = FONT_BOLD if bold else FONT_REG
    return ImageFont.truetype(path, size) if path else ImageFont.load_default()


def render_card(student_id: int, name: str | None = None, choice: str | None = None,
                written: str | None = None) -> Image.Image:
    if not 1 <= student_id <= 50:
        raise ValueError('Student ID must be from 1 to 50.')
    im = Image.new('RGB', (W, H), ORANGE_RGB)
    d = ImageDraw.Draw(im)
    d.rectangle((BORDER, HEADER, W-BORDER-1, H-BORDER-1), fill='white')

    # White quiet zone around the actual ArUco code.
    d.rectangle((30, 12, 188, 169), fill='white')
    marker = cv2.aruco.generateImageMarker(DICT, student_id - 1, MARKER_SIZE)
    im.paste(Image.fromarray(marker).convert('RGB'), (MARKER_X, MARKER_Y))
    d.text((60, 170), f'{student_id:03d}', font=font(24, True), fill='white')

    d.line((706, 20, 706, 185), fill='white', width=5)
    d.text((218, 24), 'EASYCHECK', font=font(37, True), fill='white')
    d.text((218, 79), (name or f'STUDENT {student_id:03d}')[:17], font=font(32, True), fill='white')
    d.text((218, 130), 'ID: ' + str(student_id).zfill(3), font=font(25), fill='white')

    for i, (x, y) in enumerate(BUBBLES):
        d.text((x - 13, 20), 'ABCD'[i], font=font(40, True), fill='white')
        d.ellipse((x-BUBBLE_R, y-BUBBLE_R, x+BUBBLE_R, y+BUBBLE_R),
                  fill='white', outline=(89, 43, 16), width=5)
        if choice == 'ABCD'[i]:
            # Used exclusively in synthetic self-test photos, never on printable blanks.
            for yy in range(y-42, y+43, 9):
                d.line((x-31, yy, x+31, yy+8), fill=(45, 43, 42), width=7)

    if written is not None:
        bbox = d.textbbox((0,0), written, font=font(225, True))
        tw = bbox[2] - bbox[0]
        d.text(((W-tw)//2, 275), written, font=font(225, True), fill=(50,50,50))
    return im


def pdf_bytes(count: int = 4, start: int = 1) -> bytes:
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    if count < 1 or start < 1 or start+count-1 > 50:
        raise ValueError('Can print student IDs 1-50 only.')
    buf = io.BytesIO()
    mm = 72.0 / 25.4
    a4_width, a4_height = 210*mm, 297*mm
    c = canvas.Canvas(buf, pagesize=(a4_width, a4_height))
    for i in range(count):
        if i and i % 3 == 0:
            c.showPage()
        sid = start+i
        y = a4_height-(30 + (i%3)*85 + 70)*mm
        c.drawImage(ImageReader(render_card(sid)), 30*mm, y,
                    width=150*mm, height=70*mm)
        c.setFont('Helvetica', 8)
        c.drawString(30*mm, y-4.5*mm, f'EasyCheck card #{sid:03d} - CUT along orange edge - Print at 100% scale')
    c.save()
    return buf.getvalue()
