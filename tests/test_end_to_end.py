"""Run with: python -m pytest -q (no browser required)."""
from __future__ import annotations
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import server
from cards import render_card, pdf_bytes, W,H
from vision import scan_image, synthetic_photo, jpeg_bytes, decode_photo


def token(t):
    return {'Authorization':'Bearer '+t}


@pytest.fixture
def client():
    c=TestClient(server.app)
    c.post('/api/reset',json={'pin':'042826'})
    yield c
    c.post('/api/reset',json={'pin':'042826'})


def roles(c):
    o=c.post('/api/claim',json={'role':'output'})
    assert o.status_code==200,o.text
    ot=o.json()['token']
    s=c.get('/api/state',headers=token(ot)).json()
    p=c.post('/api/claim',json={'role':'scanner','code':s['code']})
    assert p.status_code==200,p.text
    return ot,p.json()['token']


def test_print_card_pdf_dimensions_and_count():
    import fitz
    raw=pdf_bytes(4)
    pdf=fitz.open(stream=raw,filetype='pdf')
    assert len(pdf)==2
    # The card is 150x70mm as designed and appears within A4 margins.
    for i in range(4):
        img=render_card(i+1)
        assert img.size==(1500,700)
    assert raw.startswith(b'%PDF-')


def test_distinct_roles_pairing_and_reset(client):
    assert client.get('/health').json()['vision']=='aruco'
    ot,st=roles(client)
    assert client.post('/api/claim',json={'role':'output'}).status_code==409
    assert client.post('/api/claim',json={'role':'scanner','code':'BAD123'}).status_code==409
    assert client.post('/api/question',headers=token(st),json={'id':'math'}).status_code==403
    assert client.post('/api/reset',json={'pin':'bad'}).status_code==403
    assert client.post('/api/reset',headers=token(ot),json={}).status_code==200
    assert client.get('/api/state',headers=token(st)).status_code==401
    assert client.get('/api/state',headers=token(ot)).status_code==401
    assert client.get('/api/availability').json()=={'output_busy':False,'scanner_busy':False}


def test_scan_mcq_synthetic_real_engine_and_live_sync(client):
    ot,st=roles(client)
    assert client.post('/api/question',headers=token(ot),json={'id':'mcq'}).status_code==200
    before=client.get('/api/state',headers=token(st)).json()
    assert before['active']['type']=='mcq'
    assert 'answer' not in before['active']
    captured=client.post('/api/demo-scan',headers=token(st))
    assert captured.status_code==200,captured.text
    assert captured.json()['diagnostics']['cards_accepted']==4
    output=client.get('/api/state',headers=token(ot)).json()
    assert output['count']==4
    assert {x['id']:x['answer'] for x in output['results']}=={1:'C',2:'A',3:'C',4:'B'}
    assert output['summary']=={'correct':2,'incorrect':2,'review':0}
    assert client.get('/api/overlay',headers=token(st)).headers['content-type']=='image/jpeg'
    assert client.get('/api/crop/1',headers=token(ot)).headers['content-type']=='image/jpeg'
    assert client.get('/api/crop/1',headers=token(st)).status_code==403
    # Teacher manually overrides a mistaken OCR result and it updates immediately.
    r=client.post('/api/correct',headers=token(ot),json={'id':2,'answer':'C'})
    assert r.status_code==200
    output=client.get('/api/state',headers=token(ot)).json()
    assert output['summary']['correct']==3
    assert output['results'][1]['source']=='teacher correction'
    assert client.post('/api/clear-results',headers=token(ot)).status_code==200
    assert client.get('/api/state',headers=token(ot)).json()['count']==0


def test_phone_upload_math_and_question_switch(client):
    ot,st=roles(client)
    client.post('/api/question',headers=token(ot),json={'id':'math'})
    frame=synthetic_photo('written')
    raw=jpeg_bytes(frame,quality=90,max_width=2800)
    r=client.post('/api/scan',headers=token(st),files={'photo':('camera.jpg',raw,'image/jpeg')})
    assert r.status_code==200,r.text
    assert r.json()['diagnostics']['cards_accepted']==4
    out=client.get('/api/state',headers=token(ot)).json()
    assert out['count']==4
    assert any(row['answer']=='4' for row in out['results'])
    assert all(row['grade'] in ['correct','incorrect','review'] for row in out['results'])
    # Changing the question changes the scanner mode without reconnecting.
    client.post('/api/question',headers=token(ot),json={'id':'mcq'})
    assert client.get('/api/state',headers=token(st)).json()['active']['type']=='mcq'
    assert client.get('/api/state',headers=token(ot)).json()['count']==0
    client.post('/api/question',headers=token(ot),json={'id':'math'})
    assert client.get('/api/state',headers=token(ot)).json()['count']==4


def test_blank_mcq_does_not_become_false_positive():
    arr=np.asarray(render_card(1))
    frame=cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)
    found,_,diag=scan_image(frame,'mcq')
    assert diag['cards_accepted']==1
    assert found[0]['value'] is None,found[0]
    assert found[0]['status']=='review'


@pytest.mark.parametrize('variant',[0,1,2,3,4])
def test_geometry_camera_variants(variant):
    original=synthetic_photo('mcq')
    if variant==0: image=original
    if variant==1: image=cv2.rotate(original,cv2.ROTATE_90_CLOCKWISE)
    if variant==2: image=cv2.rotate(original,cv2.ROTATE_180)
    if variant==3: image=cv2.convertScaleAbs(original,alpha=.63,beta=8)
    if variant==4:
        ok,data=cv2.imencode('.jpg',original,[cv2.IMWRITE_JPEG_QUALITY,60]);assert ok
        image=cv2.imdecode(data,cv2.IMREAD_COLOR)
    cards,_,info=scan_image(image,'mcq')
    assert info['cards_accepted']==4,(variant,info)
    assert {x['id']:x['value'] for x in cards}=={1:'C',2:'A',3:'C',4:'B'},(variant,[(x['id'],x['value'],x.get('scores')) for x in cards])


def test_invalid_photo_input(client):
    ot,st=roles(client)
    client.post('/api/question',headers=token(ot),json={'id':'mcq'})
    bad=client.post('/api/scan',headers=token(st),files={'photo':('garbage.jpg',b'not-an-image','image/jpeg')})
    assert bad.status_code==422
    assert 'photo' in bad.json()['detail'].lower()
    assert client.get('/api/state',headers=token(ot)).json()['count']==0
