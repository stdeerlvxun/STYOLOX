'use strict';
(function(global){
  function imageRect(view){
    const scale=Math.min(view.width/view.imageWidth,view.height/view.imageHeight);
    const width=view.imageWidth*scale,height=view.imageHeight*scale;
    return {x:(view.width-width)/2,y:(view.height-height)/2,width,height};
  }
  function imagePoint(x,y,view){
    const radians=view.rotation*Math.PI/180,dx=x-view.width/2-view.panX,dy=y-view.height/2-view.panY;
    const px=(Math.cos(radians)*dx+Math.sin(radians)*dy)/view.zoom+view.width/2;
    const py=(-Math.sin(radians)*dx+Math.cos(radians)*dy)/view.zoom+view.height/2;
    const rect=imageRect(view),nx=(px-rect.x)/rect.width,ny=(py-rect.y)/rect.height;
    return {x:Math.max(0,Math.min(1,nx)),y:Math.max(0,Math.min(1,ny)),inside:nx>=0&&nx<=1&&ny>=0&&ny<=1};
  }
  function screenPoint(x,y,view){
    const rect=imageRect(view),px=rect.x+x*rect.width-view.width/2,py=rect.y+y*rect.height-view.height/2;
    const radians=view.rotation*Math.PI/180;
    return {x:view.width/2+view.panX+view.zoom*(Math.cos(radians)*px-Math.sin(radians)*py),
      y:view.height/2+view.panY+view.zoom*(Math.sin(radians)*px+Math.cos(radians)*py)};
  }
  const geometry={imageRect,imagePoint,screenPoint};global.AnnotationGeometry=geometry;
  if(typeof document==='undefined'){if(typeof module!=='undefined')module.exports=geometry;return;}
  const A={id:null,active:false,loading:false,source:null,version:0,boxes:[],outcome:'boxed',labelFingerprint:'',
    review:null,dirty:false,revision:0,tail:Promise.resolve(),timer:null,undo:[],selected:'',draw:false,drag:null,
    error:'',saveError:'',readOnly:false,loadTicket:0,saving:0,proposal:null,proposalId:null,rejecting:false};
  const kinds={problem_miss:'漏判',extra_candidate:'额外候选',normal_overcall:'正常图报问题',false_unrelated:'相关图误判无关',missed_unrelated:'无关图未识别'};
  const el=id=>document.getElementById(id);
  function view(){return {width:el('photoStage').clientWidth,height:el('photoStage').clientHeight,
    imageWidth:el('photo').naturalWidth,imageHeight:el('photo').naturalHeight,zoom:S.zoom,rotation:S.rotation,panX:S.panX,panY:S.panY};}
  function ready(allowNavigation=false){return A.id===S.current?.id&&A.source&&!A.loading&&!A.readOnly&&!A.error&&(allowNavigation||!S.navigating);}
  function svgNode(name,attributes){const node=document.createElementNS('http://www.w3.org/2000/svg',name);for(const [k,v] of Object.entries(attributes))node.setAttribute(k,v);return node;}
  function proposalCurrent(){return A.proposal&&A.proposal.source_sha256===A.source?.sha256&&A.proposal.label_fingerprint===A.labelFingerprint;}
  function proposalAdopted(){return A.proposal&&A.boxes.some(box=>box.proposal_id===A.proposal.proposal_id);}
  function suggestionBoxes(){return proposalCurrent()&&A.proposal.status==='pending'&&!proposalAdopted()?A.proposal.boxes:[];}
  function renderOverlay(){
    const svg=el('annotationOverlay'),v=view();svg.replaceChildren();
    const hidden=!A.active||!S.current||A.id!==S.current.id||el('photo').hidden||!v.imageWidth||(!A.boxes.length&&!A.drag&&!suggestionBoxes().length);
    // SVGElement does not reflect HTMLElement.hidden; remove the actual attribute.
    svg.toggleAttribute('hidden',hidden);
    if(hidden)return;
    svg.setAttribute('viewBox','0 0 '+v.width+' '+v.height);svg.style.transform=el('photo').style.transform;
    const rect=imageRect(v),boxes=A.boxes.map(box=>A.drag?.original?.id===box.id?{...box,...A.drag.box}:box);
    if(A.drag?.kind==='draw')boxes.push({...A.drag.box,id:'preview',problem:el('boxProblem').value});
    boxes.push(...suggestionBoxes().map(box=>({...box,suggestion:true})));
    for(const [index,box] of boxes.entries()){
      const x=rect.x+box.x*rect.width,y=rect.y+box.y*rect.height,selected=box.id===A.selected||box.id==='preview';
      const color=selected?'#ff453a':box.suggestion?'#d58aff':'#00e6a0';
      svg.append(svgNode('rect',{x,y,width:box.w*rect.width,height:box.h*rect.height,fill:'none',
        stroke:'#17212e','stroke-opacity':.85,'stroke-width':selected?6:5,
        'stroke-dasharray':box.suggestion?'6 4':'none','vector-effect':'non-scaling-stroke'}));
      svg.append(svgNode('rect',{x,y,width:box.w*rect.width,height:box.h*rect.height,fill:color,
        'fill-opacity':selected?.22:.12,stroke:color,'stroke-width':selected?3.5:2.5,
        'stroke-dasharray':box.suggestion?'6 4':'none','vector-effect':'non-scaling-stroke'}));
      const text=svgNode('text',{x:x+3,y:Math.max(14,y-5),fill:'#ffffff',stroke:'#17212e','stroke-width':3,
        'paint-order':'stroke','font-size':12/S.zoom,'font-family':'Microsoft YaHei, sans-serif'});
      text.setAttribute('stroke-width',3/S.zoom);
      text.textContent=(box.suggestion?'机器建议':(index+1))+' · '+box.problem;svg.append(text);
      if(selected&&!box.suggestion&&box.id!=='preview'){
        const size=9/v.zoom;for(const [cx,cy]of [[x,y],[x+box.w*rect.width,y],[x,y+box.h*rect.height],[x+box.w*rect.width,y+box.h*rect.height]])
          svg.append(svgNode('rect',{x:cx-size/2,y:cy-size/2,width:size,height:size,fill:color,stroke:'#ffffff','stroke-width':1.5/v.zoom}));
      }
    }
  }
  function render(){
    el('annotationBar').hidden=!A.active;el('toggleAnnotations').setAttribute('aria-pressed',String(A.active));
    el('photoStage').classList.toggle('annotating',A.active&&A.draw);
    el('photoStage').classList.toggle('box-editing',A.active&&!A.draw&&!!A.selected);
    const chosen=el('boxProblem').value;el('boxProblem').replaceChildren();
    for(const name of S.tags.defect){const option=document.createElement('option');option.value=name;option.textContent=name;el('boxProblem').append(option);}
    el('boxProblem').value=S.tags.defect.includes(chosen)?chosen:(A.review?.missing_problems?.[0]||S.current?.label.defects[0]||S.tags.defect[0]||'');
    el('boxSelect').replaceChildren();const empty=document.createElement('option');empty.value='';empty.textContent='选择已有框（'+A.boxes.length+'）';el('boxSelect').append(empty);
    for(const [i,box]of A.boxes.entries()){const option=document.createElement('option');option.value=box.id;option.textContent=(i+1)+' · '+box.problem;el('boxSelect').append(option);}
    el('boxSelect').value=A.selected;el('reviewOutcome').value=A.outcome;
    const blocked=!ready()||S.busy||el('photo').hidden;
    for(const id of ['boxProblem','drawBox','boxSelect','reviewOutcome','confirmAnnotation'])el(id).disabled=blocked;
    el('deleteBox').disabled=blocked||!A.selected;el('undoBox').disabled=blocked||!A.undo.length;
    el('relabelBox').disabled=blocked||!A.selected;
    el('reloadAnnotation').disabled=A.loading||A.saving>0||S.busy;
    el('drawBox').setAttribute('aria-pressed',String(A.draw));el('drawBox').textContent=A.draw?'框选中 · 拖动原图':'开始框选';
    el('autoBoxProposal').hidden=!A.proposal;
    const adopted=proposalAdopted(),current=proposalCurrent();
    el('adoptAutoBoxes').disabled=blocked||A.rejecting||!current||A.proposal?.status!=='pending'||adopted||!A.proposal?.boxes.length;
    el('adoptAutoBoxes').textContent=adopted?'建议框已加入补标':'采用建议框';
    el('rejectAutoBoxes').hidden=!!adopted;
    el('rejectAutoBoxes').disabled=blocked||A.rejecting||!current||A.proposal?.status!=='pending'||adopted;
    el('autoBoxSummary').textContent=!A.proposal?'':!current?'建议对应的原图或分类已变化，请重新生成建议。':adopted?'建议已加入当前补标，已有人工框保留。请调整后点击“保存补标并下一张”。':A.proposal.status==='accepted'?'这批建议已完成人工位置确认。':A.proposal.status==='rejected'?'这批机器建议已拒绝，人工分类和补标保留。':'机器初定位 · '+A.proposal.boxes.length+' 个建议框，尚未人工确认。'+(A.proposal.note||'');
    el('annotationStatus').textContent=A.error||A.readOnly?'读取有冲突，原补标保留':A.saveError?'保存未成功，可修改后重试':A.loading?'正在读取补标…':A.saving?'正在保存补标…':A.dirty?'补标待暂存':A.version?'位置补标已保存 · v'+A.version:'尚未保存位置补标';
    const label=S.current?.label,review=A.review;
    let text='当前标签：'+(label?.defects.join('、')||label?.image_state||'尚未分类')+'。补标单独保存。';
    const stateNames={normal:'无可见破损',related:'板件相关图',unrelated:'无关图',retain:'保留人工分类'};
    if(review){text+=' '+review.kinds.map(k=>kinds[k]||k).join(' / ')+'；评估时标签：'+(review.true_problems?.join('、')||stateNames[review.truth_state]||'见原分类')+'；评估模型候选：'+(review.candidate_problems?.join('、')||stateNames[review.predicted_state]||'无');}
    if(A.error||A.saveError)text=A.error||A.saveError;if(A.readOnly)text='原图已经变化，已保留原补标。请先核对源文件，再重新读取。';
    el('errorReviewSummary').textContent=text;el('errorReviewSummary').title=text;renderOverlay();
  }
  function change(action){
    if(!ready())return;A.undo.push({boxes:structuredClone(A.boxes),outcome:A.outcome,proposalId:A.proposalId});if(A.undo.length>30)A.undo.shift();
    action();A.saveError='';A.dirty=true;A.revision++;render();clearTimeout(A.timer);A.timer=setTimeout(()=>save().catch(fail),500);
  }
  async function save(status='draft'){
    clearTimeout(A.timer);if(!A.id||(!A.dirty&&status==='draft'))return A.tail;
    if(!ready(true))throw new Error(A.error||'补标尚未读取，或原图已变化，请重新读取');
    const id=A.id,revision=A.revision,boxes=structuredClone(A.boxes),outcome=A.outcome,source=A.source.sha256,fp=A.labelFingerprint,proposalId=A.proposalId;
    const task=A.tail.catch(()=>{}).then(async()=>{
      A.saving++;render();
      try{const result=await api('/api/annotations',{id,expected_version:A.version,source_sha256:source,
          label_fingerprint:fp,boxes,outcome,status,review_run:A.review?.run_id||null,...(proposalId?{proposal_id:proposalId}:{})});
        if(A.id===id){A.version=result.version;A.saveError='';if(A.revision===revision)A.dirty=false;render();}
        return result;
      }finally{A.saving--;render();}
    });A.tail=task;
    try{return await task;}catch(error){if(A.id===id){A.saveError=error.message;A.dirty=true;render();}throw error;}
  }
  async function loadForImage(item){
    const ticket=++A.loadTicket;A.id=item.id;A.source=null;A.boxes=[];A.review=null;A.undo=[];A.selected='';A.draw=false;A.drag=null;A.dirty=false;A.error='';A.saveError='';A.readOnly=false;A.version=0;A.tail=Promise.resolve();A.revision=0;A.proposal=null;A.proposalId=null;A.rejecting=false;
    const mode=el('statusFilter').value;
    if(mode.startsWith('error_review')||mode==='annotation_pending'||mode==='annotated'||mode==='auto_boxes')A.active=true;
    if(!A.active&&!item.has_annotations){A.loading=false;render();return;}
    A.loading=true;render();
    try{const result=await api('/api/annotations?id='+item.id);if(ticket!==A.loadTicket||item.id!==S.current?.id)return;
      A.source=result.source;A.boxes=result.document.boxes;A.outcome=result.document.outcome;A.version=result.version;
      A.labelFingerprint=result.label_fingerprint;A.review=result.review;A.readOnly=result.source_changed;A.tail=Promise.resolve();
      A.proposal=result.proposal||null;A.proposalId=result.document.proposal_id||A.boxes.find(box=>box.proposal_id)?.proposal_id||null;
    }catch(error){if(ticket===A.loadTicket)A.error='补标读取失败：'+error.message;}
    finally{if(ticket===A.loadTicket){A.loading=false;render();}}
  }
  async function confirm(){
    if(!ready()||S.busy||S.navigating)return;
    if(A.outcome==='boxed'&&!A.boxes.length)throw new Error('尚未框选问题位置；只复核分类请使用右侧“保存分类并完成复核”，或先选择其他复核结果。');
    S.busy=true;renderTags();render();
    try{await saveDraft();await save('confirmed');await refreshState();await reload({index:S.index});toast('补标已确认，原分类保持不变');}
    finally{S.busy=false;renderTags();render();}
  }
  function undo(){if(!ready()||!A.undo.length)return;const previous=A.undo.pop();A.boxes=previous.boxes;A.outcome=previous.outcome;A.proposalId=previous.proposalId;A.selected='';A.dirty=true;A.revision++;render();clearTimeout(A.timer);A.timer=setTimeout(()=>save().catch(fail),500);}
  el('toggleAnnotations').addEventListener('click',()=>{(async()=>{await save();A.active=!A.active;if(A.active&&S.current&&!A.source)await loadForImage(S.current);else render();})().catch(fail);});
  el('drawBox').addEventListener('click',()=>{A.draw=!A.draw;A.selected='';render();});
  el('boxSelect').addEventListener('change',()=>{A.selected=el('boxSelect').value;A.draw=false;render();});
  el('deleteBox').addEventListener('click',()=>change(()=>{A.boxes=A.boxes.filter(b=>b.id!==A.selected);A.selected='';}));
  el('relabelBox').addEventListener('click',()=>change(()=>{const box=A.boxes.find(b=>b.id===A.selected);if(box)box.problem=el('boxProblem').value;}));
  el('undoBox').addEventListener('click',undo);
  el('adoptAutoBoxes').addEventListener('click',()=>{
    if(!ready()||S.busy||!proposalCurrent()||A.proposal.status!=='pending'||proposalAdopted())return;
    change(()=>{A.proposalId=A.proposal.proposal_id;for(const box of A.proposal.boxes)A.boxes.push({...structuredClone(box),id:clientId(),origin:'model',proposal_id:A.proposalId});A.selected=A.boxes.at(-1)?.id||'';A.draw=false;A.outcome='boxed';});
    toast('建议框已加入补标草稿；可拖动框和四角调整，原分类保持不变');
  });
  el('rejectAutoBoxes').addEventListener('click',()=>{(async()=>{
    if(!ready()||S.busy||!proposalCurrent()||A.proposal.status!=='pending'||proposalAdopted()||A.rejecting)return;
    const id=A.id,proposalId=A.proposal.proposal_id;A.rejecting=true;render();
    try{await save();await api('/api/annotation-proposals/reject',{id,proposal_id:proposalId});if(A.id===id){A.proposal.status='rejected';render();}await refreshState();if(el('statusFilter').value==='auto_boxes')await reload({index:S.index});toast('已拒绝机器建议，原分类和人工补标保持不变');}
    finally{if(A.id===id){A.rejecting=false;render();}}
  })().catch(fail);});
  el('reviewOutcome').addEventListener('change',()=>{const value=el('reviewOutcome').value;change(()=>{A.outcome=value;});});
  el('confirmAnnotation').addEventListener('click',()=>confirm().catch(fail));
  el('reloadAnnotation').addEventListener('click',()=>{if(A.dirty&&!global.confirm('重新读取将放弃当前未保存的改框。是否继续？'))return;if(S.current)loadForImage(S.current).catch(fail);});
  const stage=el('photoStage');
  function pointer(event){const rect=stage.getBoundingClientRect();return imagePoint(event.clientX-rect.left,event.clientY-rect.top,view());}
  function corners(box){return [{x:box.x,y:box.y},{x:box.x+box.w,y:box.y},{x:box.x,y:box.y+box.h},{x:box.x+box.w,y:box.y+box.h}];}
  function cornerHit(event,box){const rect=stage.getBoundingClientRect(),v=view();return corners(box).findIndex(c=>{const p=screenPoint(c.x,c.y,v);return Math.hypot(p.x-event.clientX+rect.left,p.y-event.clientY+rect.top)<=11;});}
  function consume(event){event.preventDefault();event.stopImmediatePropagation();}
  stage.addEventListener('pointerdown',event=>{
    if(!A.active||!ready()||S.busy||el('photo').hidden||event.target.closest('button')||(event.button!==undefined&&event.button!==0))return;
    if(A.drag){consume(event);return;}
    const point=pointer(event),selected=A.boxes.find(box=>box.id===A.selected),corner=selected&&!A.draw?cornerHit(event,selected):-1;
    if(A.draw&&point.inside)A.drag={kind:'draw',start:point,box:{x:point.x,y:point.y,w:0,h:0}};
    else if(corner>=0)A.drag={kind:'resize',start:point,original:structuredClone(selected),corner,box:structuredClone(selected)};
    else if(!A.draw&&point.inside){
      const box=[...A.boxes].reverse().find(b=>point.x>=b.x&&point.x<=b.x+b.w&&point.y>=b.y&&point.y<=b.y+b.h);
      if(box){A.selected=box.id;A.drag={kind:'move',start:point,original:structuredClone(box),box:structuredClone(box)};}
    }
    if(!A.drag)return;
    consume(event);A.drag.pointerId=event.pointerId;stage.setPointerCapture(event.pointerId);render();
  },true);
  stage.addEventListener('pointermove',event=>{
    if(!A.drag||A.drag.pointerId!==event.pointerId)return;consume(event);const p=pointer(event),s=A.drag.start,original=A.drag.original;
    if(A.drag.kind==='move')A.drag.box={...original,x:Math.max(0,Math.min(1-original.w,original.x+p.x-s.x)),y:Math.max(0,Math.min(1-original.h,original.y+p.y-s.y))};
    else{const anchor=A.drag.kind==='resize'?corners(original)[3-A.drag.corner]:s;
      A.drag.box={x:Math.min(anchor.x,p.x),y:Math.min(anchor.y,p.y),w:Math.abs(anchor.x-p.x),h:Math.abs(anchor.y-p.y)};}
    renderOverlay();
  },true);
  stage.addEventListener('pointerup',event=>{
    if(!A.drag||A.drag.pointerId!==event.pointerId)return;consume(event);const drag=A.drag,box=drag.box;A.drag=null;stage.releasePointerCapture?.(event.pointerId);
    const rect=imageRect(view());if(box.w*rect.width*S.zoom<4||box.h*rect.height*S.zoom<4){renderOverlay();toast('框太小，请放大图片后重新框选');return;}
    if(drag.kind!=='draw'&&['x','y','w','h'].every(key=>Math.abs(box[key]-drag.original[key])<1e-8)){render();return;}
    change(()=>{if(drag.kind==='draw'){const id=clientId();A.boxes.push({...box,id,problem:el('boxProblem').value,origin:'human'});A.selected=id;}
      else{const existing=A.boxes.find(b=>b.id===drag.original.id);Object.assign(existing,{x:box.x,y:box.y,w:box.w,h:box.h});}
      A.outcome='boxed';});
  },true);
  function cancelDrag(){const pointerId=A.drag?.pointerId;A.drag=null;if(pointerId!==undefined&&stage.hasPointerCapture?.(pointerId))stage.releasePointerCapture(pointerId);renderOverlay();}
  stage.addEventListener('pointercancel',event=>{if(A.drag?.pointerId===event.pointerId)cancelDrag();},true);
  stage.addEventListener('lostpointercapture',event=>{if(A.drag?.pointerId===event.pointerId)cancelDrag();},true);
  window.addEventListener('blur',cancelDrag);
  el('photo').addEventListener('load',render);window.addEventListener('resize',renderOverlay);
  document.addEventListener('keydown',event=>{
    if(!A.active||event.target.matches('input,textarea,select'))return;
    if(event.ctrlKey&&event.key.toLowerCase()==='z'){event.preventDefault();event.stopImmediatePropagation();undo();}
    else if(event.key==='Enter'&&(event.target.closest('#annotationBar')||stage.contains(event.target))){
      event.preventDefault();event.stopImmediatePropagation();confirm().catch(fail);
    }else if(event.key==='Escape'){A.draw=false;cancelDrag();render();}
  },true);
  window.addEventListener('beforeunload',event=>{if(A.dirty){event.preventDefault();event.returnValue='';}});
  global.AnnotationUI={saveDraft:()=>save(),loadForImage,renderOverlay,refresh:()=>{if(A.active)render();},
    empty(){A.id=null;A.source=null;A.boxes=[];A.review=null;A.proposal=null;A.proposalId=null;A.loading=false;render();}};
})(globalThis);
