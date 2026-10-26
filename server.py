"""EasyCheck MVP: a single output browser plus one paired phone scanner.

All data are temporary in memory. Run as ONE Uvicorn worker. The prototype is
intended for four anonymous numbered cards, not real student identity records.
"""
from __future__ import annotations
import io
import os
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Annotated

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from cards import pdf_bytes
from vision import PhotoError, decode_photo, jpeg_bytes, scan_image, synthetic_photo

BASE=Path(__file__).resolve().parent
app=FastAPI(title='EasyCheck MVP',docs_url=None,redoc_url=None)
app.mount('/static',StaticFiles(directory=str(BASE/'static')),name='static')
ADMIN_PIN=os.environ.get('EASYCHECK_RESET_PIN','042826')
QUESTION={
    'math': {'id':'math','title':'Math — 2 + 2 = ?', 'prompt':'2 + 2 = ?',
             'type':'written','answer':'4',
             'instruction':'Write the FINAL answer large in the white area. Use your notebook for working.'},
    'mcq': {'id':'mcq','title':'Science — Which planet is the Red Planet?',
            'prompt':'Which planet is known as the Red Planet? A: Earth · B: Venus · C: Mars · D: Jupiter',
            'type':'mcq','answer':'C',
            'instruction':'Shade ONE white circle (A, B, C or D) in the orange header.'},
}
_lock=threading.RLock()
_roles={'output':None,'scanner':None}
_pair_code=None
_active=None
_generation=0
_seq=0
_results={'math':{},'mcq':{}}
_crops={'math':{},'mcq':{}}
_last_scan=None
_last_overlay=None
_capture_id=0


class Claim(BaseModel):
    role: str
    code: str = ''

class Select(BaseModel):
    id: str

class Correction(BaseModel):
    id: int
    answer: str = ''

class Reset(BaseModel):
    pin: str = ''


def error(message: str,status=400):
    raise HTTPException(status_code=status,detail=message)


def caller(request:Request):
    auth=request.headers.get('Authorization','')
    if not auth.startswith('Bearer '):
        return None
    supplied=auth[7:]
    for role,token in _roles.items():
        if token and secrets.compare_digest(token,supplied):
            return role
    return None


def require(request:Request,role: str|None=None):
    with _lock:
        actual=caller(request)
    if actual is None:
        error('Session ended or server restarted. Please choose your role again.',401)
    if role and actual!=role:
        error(f'Only the {role} device can do that.',403)
    return actual


def grade(value,status,qid):
    if value is None or status=='review':
        return 'review'
    return 'correct' if str(value).strip()==QUESTION[qid]['answer'] else 'incorrect'


@app.middleware('http')
async def headers(request:Request,call_next):
    response=await call_next(request)
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff'
    return response


@app.get('/')
def index():
    return FileResponse(BASE/'static'/'index.html')


@app.get('/health')
def health():
    return {'status':'ok','vision':'aruco' if hasattr(cv2,'aruco') else 'missing',
            'ocr':'tesseract' if shutil.which('tesseract') else 'not installed'}


@app.get('/cards.pdf')
def cards(count:int=4,start:int=1):
    try:
        raw=pdf_bytes(count,start)
    except ValueError as e:
        error(str(e))
    return StreamingResponse(io.BytesIO(raw),media_type='application/pdf',
              headers={'Content-Disposition':f'attachment; filename="EasyCheck-{count}-cards.pdf"'})


@app.get('/api/availability')
def availability():
    with _lock:
        return {'output_busy':_roles['output'] is not None,
                'scanner_busy':_roles['scanner'] is not None}


@app.post('/api/claim')
def claim(info:Claim):
    global _pair_code,_seq
    role=info.role
    if role not in _roles:
        error('Choose output or scanner.')
    with _lock:
        if _roles[role]:
            error('That role is already occupied. Use Reset Devices or the emergency reset PIN.',409)
        if role=='scanner':
            if _roles['output'] is None:
                error('Choose the output device first.',409)
            code=info.code.strip().upper()
            if not code or not secrets.compare_digest(code,_pair_code):
                error('Wrong pairing code. Read the code displayed on the output device.',403)
        else:
            _pair_code=''.join(secrets.choice('23456789ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(6))
        token=secrets.token_urlsafe(32)
        _roles[role]=token
        _seq+=1
        return {'role':role,'token':token}


@app.get('/api/state')
def state(request:Request):
    with _lock:
        role=caller(request)
        if role is None:
            error('Session ended or server restarted. Choose your role again.',401)
        q=QUESTION.get(_active) if _active else None
        current=_results[_active] if _active else {}
        public_q={k:{name:val for name,val in item.items() if name!='answer'}
                  for k,item in QUESTION.items()}
        data={'role':role,'active':(q if role=='output' else
                                   ({k:v for k,v in q.items() if k!='answer'} if q else None)),
              'questions':public_q,'version':_seq,'paired':_roles['scanner'] is not None,
              'last_scan':_last_scan,'count':len(current)}
        if role=='output':
            data.update(code=_pair_code,
                        results=[dict(v) for _,v in sorted(current.items())],
                        summary={key:sum(v['grade']==key for v in current.values())
                                 for key in ('correct','incorrect','review')},
                        capture_id=_capture_id)
        return data


@app.post('/api/question')
def select(info:Select,request:Request):
    global _active,_seq,_last_scan,_last_overlay,_capture_id
    require(request,'output')
    if info.id not in QUESTION:
        error('Choose one of the two example questions.')
    with _lock:
        _active=info.id
        _last_scan=None
        _last_overlay=None
        _capture_id+=1
        _seq+=1
        return {'ok':True,'question':QUESTION[info.id]}


def process_capture(image,token,origin='photo'):
    global _seq,_last_scan,_last_overlay,_capture_id
    with _lock:
        actual=next((k for k,v in _roles.items() if v and secrets.compare_digest(v,token)),None)
        if actual!='scanner':
            error('Scanner session ended. Please reconnect.',401)
        qid=_active
        generation=_generation
    if not qid:
        error('Select a question on the output device first.',409)
    try:
        cards,overlay,diagnostics=scan_image(image,QUESTION[qid]['type'])
        jpeg=jpeg_bytes(overlay,quality=76,max_width=1400)
    except (cv2.error,ValueError,PhotoError) as e:
        error('Image processing failed: '+str(e),422)
    with _lock:
        actual=next((k for k,v in _roles.items() if v and secrets.compare_digest(v,token)),None)
        if actual!='scanner' or _active!=qid or generation!=_generation:
            error('Question or device session changed during scan. Rescan the current question.',409)
        accepted=[]
        for obs in cards:
            row={'id':obs['id'],'answer':obs['value'],'suggested':obs.get('suggested'),
                 'status':obs['status'],'confidence':obs['confidence'],
                 'grade':grade(obs['value'],obs['status'],qid),
                 'source':'automatic scan' if obs['status']=='auto' else 'needs teacher review',
                 'note':obs['note'],'updated':int(time.time()),
                 'method':obs.get('method','bubble analysis')}
            _results[qid][obs['id']]=row
            _crops[qid][obs['id']]=obs['crop_bytes']
            accepted.append({'id':obs['id'],'answer':row['answer'],'status':row['status'],
                             'suggested':row['suggested']})
        _capture_id+=1
        _last_overlay=jpeg
        _last_scan={'question_id':qid,'count':len(accepted),'ids':[v['id'] for v in accepted],
                    'markers_found':diagnostics['markers_found'],'notes':diagnostics['notes'],
                    'origin':origin,'at':int(time.time())}
        _seq+=1
        return {'ok':True,'cards':accepted,'diagnostics':diagnostics,
                'capture_id':_capture_id,'question_id':qid}


@app.post('/api/scan')
async def photo_scan(request:Request,photo:Annotated[UploadFile,File()]):
    require(request,'scanner')
    if not photo.filename:
        error('Choose a photo first.')
    raw=await photo.read(22*1024*1024+1)
    if len(raw)>22*1024*1024:
        error('Photo exceeds 22 MB. Use a normal JPEG photo.',413)
    try:
        image=await run_in_threadpool(decode_photo,raw)
    except PhotoError as e:
        error(str(e),422)
    token=request.headers['Authorization'][7:]
    return await run_in_threadpool(process_capture,image,token,'photo')


@app.post('/api/demo-scan')
async def demo_scan(request:Request):
    require(request,'scanner')
    with _lock:
        qid=_active
    if not qid:
        error('Select a question on the output device first.',409)
    token=request.headers['Authorization'][7:]
    image=await run_in_threadpool(synthetic_photo,QUESTION[qid]['type'])
    return await run_in_threadpool(process_capture,image,token,'synthetic test')


@app.get('/api/overlay')
def last_overlay(request:Request):
    require(request)
    with _lock:
        data=_last_overlay
    if not data:
        error('No scan image yet.',404)
    return StreamingResponse(io.BytesIO(data),media_type='image/jpeg')


@app.get('/api/crop/{student_id}')
def crop(student_id:int,request:Request):
    require(request,'output')
    with _lock:
        data=_crops.get(_active,{}).get(student_id)
    if not data:
        error('Answer image not found.',404)
    return StreamingResponse(io.BytesIO(data),media_type='image/jpeg')


@app.post('/api/correct')
def correct(info:Correction,request:Request):
    global _seq
    require(request,'output')
    with _lock:
        qid=_active
        if not qid or info.id not in _results[qid]:
            error('Student not found in the current question. Scan the card first.',404)
        text=info.answer.strip().upper()
        if qid=='mcq':
            if text not in ('','A','B','C','D'):
                error('MCQ answer must be A, B, C or D.')
        elif text and (not text.isascii() or not text.isdigit() or len(text)>3):
            error('The current written-answer prototype supports 0–999 only.')
        row=_results[qid][info.id]
        row['answer']=text or None
        row['status']='manual' if text else 'review'
        row['grade']=grade(text or None,row['status'],qid)
        row['confidence']=100 if text else 0
        row['source']='teacher correction' if text else 'needs teacher review'
        row['note']='Manually verified by teacher.' if text else 'Marked for teacher review.'
        row['updated']=int(time.time())
        _seq+=1
        return {'ok':True,'result':dict(row)}


@app.post('/api/clear-results')
def clear(request:Request):
    global _seq,_last_scan
    require(request,'output')
    with _lock:
        if not _active:
            error('Select a question first.',409)
        _results[_active].clear()
        _crops[_active].clear()
        _last_scan=None
        _seq+=1
        return {'ok':True}


@app.post('/api/release')
def release(request:Request):
    global _pair_code,_seq
    with _lock:
        role=caller(request)
        if role is None:
            error('Already disconnected.',401)
        _roles[role]=None
        if role=='output':
            _roles['scanner']=None
            _pair_code=None
        _seq+=1
        return {'ok':True}


@app.post('/api/reset')
def reset(info:Reset,request:Request):
    global _pair_code,_active,_generation,_seq,_results,_crops,_last_scan,_last_overlay,_capture_id
    with _lock:
        authorised=(caller(request)=='output')
        if not authorised and not (ADMIN_PIN and secrets.compare_digest(info.pin,ADMIN_PIN)):
            error('Only the output device or the emergency reset PIN can reset.',403)
        for role in _roles:
            _roles[role]=None
        _pair_code=None
        _active=None
        _generation+=1
        _results={'math':{},'mcq':{}}
        _crops={'math':{},'mcq':{}}
        _last_scan=None
        _last_overlay=None
        _capture_id+=1
        _seq+=1
        return {'ok':True}
