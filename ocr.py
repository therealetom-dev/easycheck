"""Conservative handwriting OCR for the one-digit demo question.

The output is never silently treated as correct if the recognition evidence is weak.
Tesseract is bundled in deployment. Optional compact SVM (trained on real handwritten
scikit-learn digits dataset) can independently cross-check it.
"""
from __future__ import annotations
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
_MODEL = None


def _get_svm():
    global _MODEL
    if _MODEL is None and (ROOT/'assets'/'digits_svm.xml').is_file():
        _MODEL = cv2.ml.SVM_load(str(ROOT/'assets'/'digits_svm.xml'))
    return _MODEL


def _ink_roi(image: np.ndarray):
    """Returns a cropped region containing strong non-background pencil strokes."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = gray[20:-20, 35:-35] if gray.shape[0] > 60 and gray.shape[1] > 100 else gray
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # A flat-field correction improves contrast on uneven paper and faint graphite.
    background = cv2.GaussianBlur(blurred, (0, 0), 21)
    corrected = cv2.divide(blurred, np.maximum(background, 1), scale=255)
    threshold = cv2.adaptiveThreshold(corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 41, 12)
    # Reject isolated dust, not connected handwriting strokes.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(threshold, 8)
    selected = np.zeros_like(threshold)
    for i in range(1, n):
        x,y,w,h,area = stats[i]
        if area < 8 or (w < 2 and h < 3):
            continue
        selected[labels == i] = 255
    ys, xs = np.where(selected != 0)
    if len(xs) < 90:
        return None, 0.0
    x1, x2 = int(xs.min()), int(xs.max())+1
    y1, y2 = int(ys.min()), int(ys.max())+1
    pad = 28
    x1, x2 = max(0,x1-pad), min(gray.shape[1],x2+pad)
    y1, y2 = max(0,y1-pad), min(gray.shape[0],y2+pad)
    crop = corrected[y1:y2,x1:x2]
    fraction = float(np.count_nonzero(selected))/float(selected.size)
    return crop, fraction


def _svm_digit(crop: np.ndarray):
    svm = _get_svm()
    if svm is None:
        return None
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area>25 and w>2 and h>3:
            components.append((x,y,w,h,area))
    if not components:
        return None
    # A single 8x8 model is only appropriate for a single symbol.
    left=min(x for x,_,_,_,_ in components)
    top=min(y for _,y,_,_,_ in components)
    right=max(x+w for x,y,w,h,area in components)
    bottom=max(y+h for x,y,w,h,area in components)
    bw,bh=right-left,bottom-top
    if bw<5 or bh<7 or bw>bh*1.8:
        return None
    cut = mask[top:bottom,left:right]
    # Mimic the 8x8 handwritten-digit dataset's centred 6x6 glyph shape.
    target=np.zeros((32,32),np.uint8)
    scale=min(23/max(1,bw),23/max(1,bh))
    shape=(max(1,round(bw*scale)),max(1,round(bh*scale)))
    res=cv2.resize(cut,shape,interpolation=cv2.INTER_AREA)
    target[(32-shape[1])//2:(32-shape[1])//2+shape[1],(32-shape[0])//2:(32-shape[0])//2+shape[0]]=res
    small = cv2.resize(target,(8,8),interpolation=cv2.INTER_AREA).astype(np.float32)
    feat = ((small/255.0)*16.0).reshape(1,-1)
    _,pred=svm.predict(feat)
    return str(int(pred[0][0]))


def _tesseract_candidates(crop: np.ndarray):
    if not shutil.which('tesseract'):
        return []
    import pytesseract
    from PIL import Image
    # Run bounded, different page segmentation modes; printed text OCR is
    # not guaranteed to recognise handwriting, so low confidence stays 'review'.
    image = cv2.copyMakeBorder(crop, 35,35,35,35,cv2.BORDER_CONSTANT,value=255)
    h,w=image.shape
    if h<80 or w<80:
        image=cv2.resize(image,(max(80,w*2),max(80,h*2)),interpolation=cv2.INTER_CUBIC)
    images = [image]
    _,bw=cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    images.append(bw)
    choices=[]
    for array,psm in [(images[0],10),(images[1],8),(images[1],13)]:
        try:
            data=pytesseract.image_to_data(Image.fromarray(array), output_type=pytesseract.Output.DICT,
                config=f'--oem 1 --psm {psm} -c tessedit_char_whitelist=0123456789', timeout=4)
            for word,confidence in zip(data['text'],data['conf']):
                cleaned=re.sub(r'[^0-9]','',word.strip())
                if cleaned and len(cleaned)<=3:
                    try:
                        conf=float(confidence)
                    except (ValueError,TypeError):
                        conf=-1
                    choices.append((cleaned,conf))
        except (RuntimeError, OSError, Exception):
            continue
    return choices


def read_numeric(answer_image: np.ndarray):
    """Returns {value, suggested, status, confidence, method, note}.
    A value is returned only when evidence exceeds conservative thresholds.
    A suggestion is displayed for human verification, never auto-graded.
    """
    crop, density = _ink_roi(answer_image)
    if crop is None:
        return {'value':None, 'suggested':None, 'status':'review','confidence':0,
                'method':'no-ink', 'note':'No clearly legible handwriting detected. Write a large answer.'}
    votes=_tesseract_candidates(crop)
    svm_value=_svm_digit(crop)
    votes.sort(key=lambda x:x[1],reverse=True)
    chosen=None
    if votes:
        # Prefer repeat agreement to a single suspicious OCR guess.
        counts={}
        for word,conf in votes:
            counts[word]=counts.get(word,0)+1
        chosen=max(counts,key=lambda v:(counts[v], max(c for s,c in votes if s==v)))
        agreeing=[conf for s,conf in votes if s==chosen]
        if len(agreeing)>=2 and max(agreeing)>=55 and (svm_value is None or svm_value==chosen):
            return {'value':chosen,'suggested':chosen,'status':'auto','confidence':min(99,round(max(agreeing))),
                    'method':'OCR agreement','note':'Computer reading: please verify if handwriting is ambiguous.'}
        if svm_value==chosen and max(agreeing)>=45:
            return {'value':chosen,'suggested':chosen,'status':'auto','confidence':min(95,round(max(agreeing))),
                    'method':'OCR + digit model','note':'Computer reading: please verify if handwriting is ambiguous.'}
    suggestion=chosen or svm_value
    if chosen and svm_value and chosen != svm_value:
        return {'value':None,'suggested':chosen,'status':'review','confidence':0,
                'method':'conflicting OCR','note':f'Handwriting models disagree ({chosen} vs {svm_value}). Teacher review required.'}
    return {'value':None,'suggested':suggestion,'status':'review','confidence':0,
            'method':'low-confidence OCR' if suggestion else 'unreadable',
            'note':'Could not read handwriting confidently. Check the cropped answer and correct it manually.'}
