"""Four-card-ready ArUco image scanning. No network/model downloads are needed."""
from __future__ import annotations
import io
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from cards import (W, H, BORDER, HEADER, DICT, MARKER_X, MARKER_Y,
                   MARKER_SIZE, BUBBLES, BUBBLE_R, render_card)
from ocr import read_numeric

PARAM=cv2.aruco.DetectorParameters()
PARAM.minMarkerPerimeterRate=0.008  # Small marker on high-resolution phone photos.
PARAM.maxMarkerPerimeterRate=4.0
PARAM.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
PARAM.cornerRefinementWinSize=5
PARAM.adaptiveThreshWinSizeMin=3
PARAM.adaptiveThreshWinSizeMax=31
PARAM.adaptiveThreshWinSizeStep=4
DETECTOR=cv2.aruco.ArucoDetector(DICT,PARAM)


class PhotoError(ValueError):
    pass


def decode_photo(raw: bytes) -> np.ndarray:
    if not raw:
        raise PhotoError('The uploaded image is empty.')
    try:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        pil=Image.open(io.BytesIO(raw))
        # JPEG draft can reduce decode size before a 48 MP phone image is
        # materialised, important on free hosts with limited RAM.
        if max(pil.size)>3600:
            try:
                pil.draft('RGB',(3600,3600))
            except (ValueError, OSError):
                pass
        pil=ImageOps.exif_transpose(pil)
        if pil.width*pil.height > 40_000_000:
            # Avoid memory exhaustion from extremely large phone photos.
            scale=math.sqrt(10_000_000/(pil.width*pil.height))
            pil.thumbnail((int(pil.width*scale),int(pil.height*scale)))
        pil=pil.convert('RGB')
        arr=np.asarray(pil)
        if max(pil.size)>3600 or pil.width*pil.height>10_000_000:
            scale=min(3600/max(pil.size),math.sqrt(10_000_000/(pil.width*pil.height)))
            arr=cv2.resize(arr,(max(1,round(pil.width*scale)),max(1,round(pil.height*scale))),
                           interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)
    except (ValueError,OSError,TypeError) as e:
        raise PhotoError('Cannot read this photo. Please use JPEG, PNG or a supported HEIC photo.') from e


def jpeg_bytes(img: np.ndarray, quality=83, max_width=1200) -> bytes:
    h,w=img.shape[:2]
    if w>max_width:
        img=cv2.resize(img,(max_width,round(h*max_width/w)),interpolation=cv2.INTER_AREA)
    ok,out=cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,quality])
    if not ok:
        raise PhotoError('Could not encode scan preview.')
    return out.tobytes()


def _find_markers(gray: np.ndarray):
    # First pass on original image retains details in small printed markers.
    corners,ids,_ = DETECTOR.detectMarkers(gray)
    if ids is not None:
        return corners,ids
    # Second pass can rescue markers in flat, poorly lit scenes.
    enhanced=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
    return DETECTOR.detectMarkers(enhanced)[:2]


def _bubble_scores(card: np.ndarray):
    gray=cv2.cvtColor(card,cv2.COLOR_BGR2GRAY)
    # The ArUco marker is only 13.2 mm wide, so its perspective transform
    # can drift slightly at the far side of the 15 cm card. Independently
    # locate each printed bubble before measuring its interior.
    region=cv2.medianBlur(gray[37:195,650:1430],5)
    circles=cv2.HoughCircles(region,cv2.HOUGH_GRADIENT,dp=1.2,
        minDist=120,param1=105,param2=27,minRadius=34,maxRadius=77)
    located={}
    if circles is not None:
        for cx,cy,rad in circles[0]:
            x,y=float(cx+650),float(cy+37)
            distances=[abs(x-bx)+abs(y-by)*0.5 for bx,by in BUBBLES]
            idx=int(np.argmin(distances))
            if distances[idx]<95 and (idx not in located or distances[idx]<located[idx][3]):
                located[idx]=(round(x),round(y),round(rad),distances[idx])
    if len(located)<4:
        return None,0,[],"review","Could not confidently locate all four printed MCQ circles. Take a closer photo."
    scores=[]
    for i in range(4):
        x,y,rad,_=located[i]
        radius=max(10,round(rad*0.66))
        roi=gray[y-radius:y+radius+1,x-radius:x+radius+1]
        if roi.shape!=(2*radius+1,2*radius+1):
            return None,0,[],"review","One MCQ bubble is clipped by the image."
        yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]
        interior=(xx*xx+yy*yy)<(radius-2)**2
        px=roi[interior]
        background=float(np.percentile(px,85))
        cutoff=min(224.0,background-25.0)
        scores.append(round(float(np.mean(px<cutoff)),4))
    sorted_scores=sorted(scores,reverse=True)
    top_idx=int(np.argmax(scores))
    top,runner=sorted_scores[:2]
    if top>=0.043 and top-runner>=0.027 and (runner<0.065 or top>=1.9*runner):
        return 'ABCD'[top_idx],round(min(99,max(55,(top-runner)*280+55))),scores,'auto','Bubble mark detected.'
    if top<0.043:
        return None,0,scores,'review','No clearly filled bubble. Shade one white circle.'
    return None,0,scores,'review','Marks are ambiguous or multiple circles appear selected.'


def _orange_match(card: np.ndarray) -> float:
    sample=card[30:170,215:675]
    hsv=cv2.cvtColor(sample,cv2.COLOR_BGR2HSV)
    # Broad tolerance accommodates lighting/cheap printers without treating white as orange.
    good=(hsv[:,:,0]>=1)&(hsv[:,:,0]<=32)&(hsv[:,:,1]>=42)&(hsv[:,:,2]>=45)
    return float(np.mean(good))


def scan_image(img: np.ndarray, mode: str):
    if mode not in ('mcq','written'):
        raise ValueError('Unknown question type.')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    corners,ids=_find_markers(gray)
    overlay=img.copy()
    if ids is None:
        return [],overlay,{'markers_found':0,'cards_accepted':0,'notes':[
            'No valid EasyCheck marker detected. Ensure the four cards are printed from this app, in focus and fully visible.',
            'Move the camera closer, improve lighting, or photograph fewer cards at once.']}
    by_id={}
    notes=[]
    markers_found=len(ids)
    for marker_pts, raw_id in zip(corners,ids.flatten()):
        sid=int(raw_id)+1
        pts=marker_pts.reshape(4,2).astype(np.float32)
        marker_width=float(np.linalg.norm(pts[0]-pts[1]))
        if marker_width<20:
            notes.append(f'ID {sid}: marker too small in the photo ({marker_width:.0f} px). Move closer.')
            continue
        src=np.float32([[MARKER_X,MARKER_Y],
                        [MARKER_X+MARKER_SIZE,MARKER_Y],
                        [MARKER_X+MARKER_SIZE,MARKER_Y+MARKER_SIZE],
                        [MARKER_X,MARKER_Y+MARKER_SIZE]])
        hom=cv2.getPerspectiveTransform(src,pts)
        projected=cv2.perspectiveTransform(np.float32([[[0,0],[W,0],[W,H],[0,H]]]),hom)[0]
        if not np.isfinite(projected).all():
            notes.append(f'ID {sid}: could not determine card corners.')
            continue
        area=abs(cv2.contourArea(projected.astype(np.float32)))
        if area<3000:
            notes.append(f'ID {sid}: card is too small for reliable analysis.')
            continue
        # Refuse cards where most of the image lies outside the photograph (clipped borders).
        xx=projected[:,0];yy=projected[:,1]
        out_left=(xx.min()<-img.shape[1]*0.02 or xx.max()>img.shape[1]*1.02 or
                  yy.min()<-img.shape[0]*0.02 or yy.max()>img.shape[0]*1.02)
        if out_left:
            notes.append(f'ID {sid}: card edges are cut off. Keep all four edges in frame.')
            continue
        try:
            inv=np.linalg.inv(hom)
        except np.linalg.LinAlgError:
            continue
        card=cv2.warpPerspective(img,inv,(W,H),flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255))
        orange=_orange_match(card)
        if orange<0.18:
            notes.append(f'ID {sid}: marker detected, but the orange EasyCheck header does not match. Use the generated card layout.')
            continue
        x,y=map(int,projected[0])
        if mode=='mcq':
            value,confidence,scores,status,note=_bubble_scores(card)
            extra={'scores':scores}
        else:
            answer_img=card[HEADER+17:H-BORDER-15,BORDER+5:W-BORDER-5]
            ocr=read_numeric(answer_img)
            value,confidence,status,note=ocr['value'],ocr['confidence'],ocr['status'],ocr['note']
            extra={'suggested':ocr['suggested'],'method':ocr['method']}
        crop=card[HEADER+12:H-BORDER-9,BORDER+5:W-BORDER-5]
        candidate={'id':sid,'value':value,'status':status,'confidence':confidence,'note':note,
                   'orange_score':round(orange,2),'marker_pixels':round(marker_width),
                   'crop_bytes':jpeg_bytes(crop,quality=89,max_width=1200),**extra}
        # If the same marker is falsely detected twice, keep the sharper/larger instance.
        if sid not in by_id or by_id[sid]['marker_pixels']<marker_width:
            by_id[sid]=candidate
        cv2.polylines(overlay,[projected.astype(np.int32)],True,(22,180,18),max(2,round(img.shape[1]/1400)))
        label=f'ID {sid}: {value if value is not None else "REVIEW"}'
        cv2.putText(overlay,label,(max(0,x),max(25,y-10)),cv2.FONT_HERSHEY_SIMPLEX,
                    max(.65,img.shape[1]/3500),(20,150,20),max(2,round(img.shape[1]/1700)),cv2.LINE_AA)
    cards=sorted(by_id.values(),key=lambda v:v['id'])
    if not cards and not notes:
        notes=['Markers were found, but none were recognised as complete EasyCheck cards.']
    return cards,overlay,{'markers_found':markers_found,'cards_accepted':len(cards),'notes':notes}


def synthetic_photo(mode: str) -> np.ndarray:
    """Synthetic test image, passed through the same real detection and OCR code."""
    if mode not in ('written','mcq'):
        raise ValueError('Unknown mode')
    out=np.full((1580,2600,3),(199,207,215),np.uint8)
    choices=['C','A','C','B']
    answers=['4','3','4','4']
    positions=[(95,100),(1385,105),(105,855),(1390,850)]
    for i in range(4):
        card=render_card(i+1,choice=choices[i] if mode=='mcq' else None,
                         written=answers[i] if mode=='written' else None)
        rgb=np.array(card)
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        w=1120;h=520
        xpos,ypos=positions[i]
        # Small reproducible trapezoid distortions, but no cards overlap.
        dst=np.float32([[xpos+12,ypos+6],[xpos+w-15,ypos+10],
                        [xpos+w+0,ypos+h-12],[xpos+0,ypos+h+4]])
        hom=cv2.getPerspectiveTransform(np.float32([[0,0],[W,0],[W,H],[0,H]]),dst)
        warped=cv2.warpPerspective(bgr,hom,(out.shape[1],out.shape[0]),borderValue=(0,0,0))
        mask=cv2.warpPerspective(np.full((H,W),255,np.uint8),hom,
                                 (out.shape[1],out.shape[0]))
        out[mask>30]=warped[mask>30]
    return out
