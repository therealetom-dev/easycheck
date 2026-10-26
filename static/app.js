'use strict';
const $ = (id) => document.getElementById(id);
let credential = localStorage.getItem('easycheck_credential') || '';
let role = localStorage.getItem('easycheck_role') || '';
let version = -1;
let lastCapture = -1;
let currentQuestion = '';
let currentState = null;
let videoStream = null;
let busy = false;
let polling = false;

const escapeHTML = (s) => String(s ?? '').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const msg = (s,err=false) => { $('notice').textContent=s; $('notice').className = err?'error':'ok'; };
const auth = () => credential ? {'Authorization':'Bearer '+credential} : {};

async function requestAPI(path, {method='GET', body=null, isForm=false}={}) {
  const headers = auth();
  const options = {method, headers, cache:'no-store'};
  if (body !== null) {
    if (isForm) options.body=body;
    else { headers['Content-Type']='application/json'; options.body=JSON.stringify(body); }
  }
  let response;
  try { response=await fetch(path,options); }
  catch { throw new Error('Cannot reach server. Check internet or free-host wake-up, then retry.'); }
  const type=response.headers.get('content-type')||'';
  const data=type.includes('application/json') ? await response.json() : await response.text();
  if (!response.ok) {
    if (response.status===401 && role) disconnect('Session ended. Choose a device role again.');
    throw new Error(data.detail || data.error || 'Request failed ('+response.status+').');
  }
  return data;
}

function disconnect(text='') {
  credential='';role='';version=-1;currentQuestion='';currentState=null;
  localStorage.removeItem('easycheck_credential');localStorage.removeItem('easycheck_role');
  stopCamera();
  showPage();
  if(text) msg(text,true);
  refreshAvailability();
}

function showPage() {
  $('landing').classList.toggle('hidden',!!role);
  $('output').classList.toggle('hidden',role!=='output');
  $('scanner').classList.toggle('hidden',role!=='scanner');
}

async function refreshAvailability() {
  try {
    const x=await requestAPI('/api/availability');
    $('occupied').textContent = `Current roles: Output ${x.output_busy?'occupied':'available'} · Scanner ${x.scanner_busy?'occupied':'available'}`;
  } catch(e) { $('occupied').textContent=e.message; }
}

async function claim(target) {
  const code=target==='scanner' ? $('pair-input').value.trim().toUpperCase() : '';
  try {
    const data=await requestAPI('/api/claim',{method:'POST',body:{role:target,code}});
    credential=data.token;role=data.role;version=-1;
    localStorage.setItem('easycheck_credential',credential);
    localStorage.setItem('easycheck_role',role);
    msg('');showPage();await poll();
  } catch(e) { msg(e.message,true); }
}

async function poll() {
  if(!credential || polling) return;
  polling=true;
  try {
    const s=await requestAPI('/api/state');
    if(s.version!==version || s.active?.id!==currentQuestion) {
      version=s.version;
      currentQuestion=s.active?.id||'';
      currentState=s;
      if(role==='output') paintOutput(s);
      if(role==='scanner') paintScanner(s);
    }
  } catch(e) {
    if(role) $('notice').textContent=e.message;
  } finally { polling=false; }
}

function paintOutput(s) {
  $('pair-code').textContent=s.code||'------';
  $('paired-status').textContent=s.paired?'Scanner connected':'Waiting for scanner to connect';
  $('paired-status').className=s.paired?'ok':'muted';
  const q=s.active;
  for(const [id,button] of [['math','pick-math'],['mcq','pick-mcq']]) {
    $(button).className='card-button '+(q?.id===id?'selected':'secondary');
  }
  $('active-output-question').textContent=q?('Selected: '+q.prompt):'No question selected. Choose one above.';
  $('teacher-key').textContent=q?('Expected answer: '+q.answer+' · '+q.instruction):'';
  const sum=s.summary||{correct:0,incorrect:0,review:0};
  $('stats').textContent=q?`${s.count} cards · ${sum.correct} correct · ${sum.incorrect} incorrect · ${sum.review} review`:'Select a question first.';
  const ls=s.last_scan;
  $('last-scan-output').textContent=ls ? `Last ${ls.origin}: ${ls.count} cards (marker detections: ${ls.markers_found}). ${ls.notes.join(' ')}` : '';
  const rows=s.results||[];
  const html=rows.map(a=>{
    const shown=a.answer ?? (a.suggested ? `? ${a.suggested}`:'—');
    const gradeClass=a.grade==='correct'?'ok':a.grade==='incorrect'?'bad':'review';
    const editor=q?.type==='mcq'?
      `<select data-id="${a.id}" aria-label="Correct response for student ${a.id}">
         <option value="" ${a.answer==null?'selected':''}>Review</option>
         ${'ABCD'.split('').map(x=>`<option value="${x}" ${a.answer===x?'selected':''}>${x}</option>`).join('')}
       </select>` :
      `<input data-id="${a.id}" type="text" inputmode="numeric" maxlength="3" size="3" value="${escapeHTML(a.answer??'')}" aria-label="Correct answer for student ${a.id}">`;
    return `<tr><td>${a.id}</td><td>${escapeHTML(shown)}<br><span class="small muted">${escapeHTML(a.source)}</span></td>
      <td class="${gradeClass}"><strong>${escapeHTML(a.grade)}</strong><br><span class="small">${escapeHTML(a.note)}</span></td>
      <td>${editor}<button class="secondary" data-save="${a.id}">Save</button></td>
      <td><button class="secondary" data-crop="${a.id}">View crop</button></td></tr>`;
  }).join('');
  $('results').innerHTML=html || '<tr><td colspan="5" class="muted">No scans for this question yet.</td></tr>';
  // Image retrieval only happens on demand, rather than on every 1.5s poll.
  if(s.capture_id!==lastCapture) { lastCapture=s.capture_id; $('output-overlay').classList.add('hidden'); }
}

function paintScanner(s) {
  const q=s.active;
  $('scanner-question').textContent=q?`Current question: ${q.prompt}`:'Waiting for the output device to choose a question...';
  $('scanner-instruction').textContent=q?q.instruction:'';
  $('send-photo').disabled=!q||busy;
  $('demo-scan').disabled=!q||busy;
  $('scan-video').disabled=!q||busy;
  const last=s.last_scan;
  if(last) {
    $('scan-status').textContent=`Last scan (${last.origin}): ${last.count} card(s) found: ${last.ids.join(', ')||'none'}. ${last.notes.join(' ')}`;
  }
}

async function choose(id) {
  try { await requestAPI('/api/question',{method:'POST',body:{id}}); await poll(); }
  catch(e) { msg(e.message,true); }
}

async function getImage(url,img) {
  const r=await fetch(url,{headers:auth(),cache:'no-store'});
  if(!r.ok) throw Error('Image is no longer available.');
  const blob=await r.blob();
  const obj=URL.createObjectURL(blob);
  const old=img.dataset.objectUrl;
  img.src=obj;img.dataset.objectUrl=obj;
  img.classList.remove('hidden');
  if(old) URL.revokeObjectURL(old);
}

async function preparePhoto(file) {
  // Reduce 48 MP phone photos BEFORE upload to protect free-host memory.
  // The four-card MVP still gets ample detail at 3600px longest edge.
  if(typeof createImageBitmap !== 'function') return file;
  try {
    const bitmap=await createImageBitmap(file);
    const w=bitmap.width,h=bitmap.height;
    const factor=Math.min(1,3600/Math.max(w,h),Math.sqrt(10_000_000/(w*h)));
    if(factor>=.999) { bitmap.close?.();return file; }
    const c=document.createElement('canvas');
    c.width=Math.max(1,Math.round(w*factor));c.height=Math.max(1,Math.round(h*factor));
    c.getContext('2d').drawImage(bitmap,0,0,c.width,c.height);
    bitmap.close?.();
    const resized=await new Promise(resolve=>c.toBlob(resolve,'image/jpeg',.94));
    return resized||file;
  }catch{return file;}
}

async function sendScan(useSynthetic=false,blob=null) {
  if(busy)return;
  busy=true;
  $('send-photo').disabled=true; $('demo-scan').disabled=true; $('scan-video').disabled=true;
  const status=$('scan-status');status.textContent='Processing image... Do not close this page.';
  try {
    let data;
    if(useSynthetic) data=await requestAPI('/api/demo-scan',{method:'POST'});
    else {
      let f=blob||$('file').files[0];
      if(!f) throw Error('Take a photo or choose an image first.');
      status.textContent='Preparing full-quality image for upload...';
      const prepared=await preparePhoto(f);
      const body=new FormData();body.append('photo',prepared,prepared!==f?'resized-camera.jpg':(blob?'camera.jpg':f.name));
      status.textContent='Uploading and reading cards...';
      data=await requestAPI('/api/scan',{method:'POST',body,isForm:true});
    }
    const d=data.diagnostics;
    const lines=[`Scan finished: ${d.cards_accepted} card(s) recognised, ${d.markers_found} markers detected.`];
    if(data.cards.length) lines.push('IDs: '+data.cards.map(a=>`${a.id}: ${a.answer??(a.suggested?'review ('+a.suggested+'?)':'review')}`).join(' · '));
    if(d.notes.length) lines.push(...d.notes);
    if(d.cards_accepted===0) lines.push('Try a closer, sharper photo. The ArUco code and the entire card must be visible.');
    status.textContent=lines.join('\n');
    status.className=d.cards_accepted?'ok status':'error status';
    await getImage('/api/overlay',$('scan-overlay'));
    await poll();
  } catch(e) { status.textContent=e.message;status.className='error status'; }
  finally { busy=false; $('send-photo').disabled=!currentQuestion; $('demo-scan').disabled=!currentQuestion; $('scan-video').disabled=!currentQuestion; }
}

async function startCamera() {
  try {
    if(!navigator.mediaDevices?.getUserMedia) throw Error('Live camera needs HTTPS (or localhost). You can still take a photo using the file input above.');
    stopCamera();
    const stream=await navigator.mediaDevices.getUserMedia({audio:false,video:{facingMode:{ideal:'environment'},width:{ideal:3840},height:{ideal:2160}}});
    videoStream=stream;
    $('video').srcObject=stream;
    $('video').classList.remove('hidden');
    $('scan-video').classList.remove('hidden');
    await $('video').play();
    $('scan-status').textContent=`Live camera ready: ${$('video').videoWidth}×${$('video').videoHeight}.`;
  } catch(e) { $('scan-status').textContent=e.message; }
}

function stopCamera() {
  if(videoStream) videoStream.getTracks().forEach(t=>t.stop());
  videoStream=null;
  $('video').srcObject=null;
  $('video').classList.add('hidden');$('scan-video').classList.add('hidden');
}

async function captureVideo() {
  const v=$('video');
  if(!videoStream || !v.videoWidth) { $('scan-status').textContent='Camera not ready.';return; }
  const c=document.createElement('canvas');
  c.width=v.videoWidth;c.height=v.videoHeight;
  c.getContext('2d').drawImage(v,0,0);
  const blob=await new Promise(resolve=>c.toBlob(resolve,'image/jpeg',.94));
  if(blob)await sendScan(false,blob);
}

async function correct(id) {
  const editor=$('results').querySelector(`[data-id="${id}"]`);
  try {
    await requestAPI('/api/correct',{method:'POST',body:{id,answer:editor?.value||''}});
    await poll();msg('Teacher correction saved.');
  }catch(e){msg(e.message,true);}
}

async function viewCrop(id) {
  try {
    await getImage(`/api/crop/${id}`,$('crop-img'));
    $('crop-title').textContent='Card '+id+' — written answer image';
    $('crop-dialog').showModal();
  } catch(e) { msg(e.message,true); }
}

async function resetAll(isEmergency=false) {
  if(!confirm('Log out BOTH devices and ERASE every recorded answer?'))return;
  try {
    const pin=isEmergency?$('emergency-pin').value:'';
    await requestAPI('/api/reset',{method:'POST',body:{pin}});
    disconnect('Both device roles reset; all answers cleared.');
  } catch(e) {msg(e.message,true);}
}

async function leave() {
  try {await requestAPI('/api/release',{method:'POST'});} catch(e) { /* already disconnected */ }
  disconnect('Device role released.');
}

$('join-output').onclick=()=>claim('output');
$('join-scanner').onclick=()=>claim('scanner');
$('pick-math').onclick=()=>choose('math');
$('pick-mcq').onclick=()=>choose('mcq');
$('send-photo').onclick=()=>sendScan(false);
$('demo-scan').onclick=()=>sendScan(true);
$('start-camera').onclick=startCamera;
$('stop-camera').onclick=stopCamera;
$('scan-video').onclick=captureVideo;
$('clear-results').onclick=async()=>{if(confirm('Erase responses for the selected question?')){try{await requestAPI('/api/clear-results',{method:'POST'});await poll();}catch(e){msg(e.message,true);}}};
$('reset-output').onclick=()=>resetAll(false);
$('emergency-reset').onclick=()=>resetAll(true);
$('leave-output').onclick=leave;
$('leave-scanner').onclick=leave;
$('show-overlay').onclick=async()=>{ try { await getImage('/api/overlay',$('output-overlay-img'));$('output-overlay').classList.remove('hidden'); } catch(e) {msg(e.message,true);} };
$('results').addEventListener('click',e=>{
  const save=e.target.closest('[data-save]');const crop=e.target.closest('[data-crop]');
  if(save)correct(Number(save.dataset.save));
  if(crop)viewCrop(Number(crop.dataset.crop));
});
$('close-crop').onclick=()=>$('crop-dialog').close();
$('pair-input').addEventListener('keydown',e=>{if(e.key==='Enter')claim('scanner')});

async function init() {
  showPage();
  if(credential)await poll();
  else await refreshAvailability();
  setInterval(()=>{ if(credential)poll(); else refreshAvailability(); }, 1600);
}
init();
