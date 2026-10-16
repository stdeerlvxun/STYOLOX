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
  function overlapRatio(a,b){const width=Math.max(0,Math.min(a.x+a.w,b.x+b.w)-Math.max(a.x,b.x)),height=Math.max(0,Math.min(a.y+a.h,b.y+b.h)-Math.max(a.y,b.y));return width*height/Math.min(a.w*a.h,b.w*b.h);}
  const geometry={imageRect,imagePoint,screenPoint,overlapRatio};global.AnnotationGeometry=geometry;
  if(typeof document==='undefined'){if(typeof module!=='undefined')module.exports=geometry;return;}
  const A={id:null,active:false,loading:false,source:null,version:0,boxes:[],outcome:'boxed',labelFingerprint:'',
    review:null,dirty:false,revision:0,tail:Promise.resolve(),timer:null,undo:[],selected:'',draw:false,drag:null,
    error:'',saveError:'',readOnly:false,loadTicket:0,saving:0,proposal:null,proposalId:null,proposalFingerprint:null,rejecting:false,
    negativeRegions:[],documentStatus:null,feedback:null,popup:null,lastProblem:'',quickFlow:false};
  const kinds={problem_miss:'漏判',extra_candidate:'额外候选',normal_overcall:'正常图报问题',false_unrelated:'相关图误判无关',missed_unrelated:'无关图未识别'};
  const el=id=>document.getElementById(id);
  function view(){return {width:el('photoStage').clientWidth,height:el('photoStage').clientHeight,
    imageWidth:el('photo').naturalWidth,imageHeight:el('photo').naturalHeight,zoom:S.zoom,rotation:S.rotation,panX:S.panX,panY:S.panY};}
  function ready(allowNavigation=false){return A.id===S.current?.id&&A.source&&!A.loading&&!A.readOnly&&!A.error&&(allowNavigation||!S.navigating);}
  function svgNode(name,attributes){const node=document.createElementNS('http://www.w3.org/2000/svg',name);for(const [k,v] of Object.entries(attributes))node.setAttribute(k,v);return node;}
  function proposalCurrent(){return A.proposal&&A.proposal.source_sha256===A.source?.sha256&&A.proposal.label_fingerprint===A.labelFingerprint;}
  function proposalAdopted(){return A.proposal&&(A.proposalId===A.proposal.proposal_id||A.boxes.some(box=>box.proposal_id===A.proposal.proposal_id));}
  function suggestionBoxes(){return proposalCurrent()&&A.proposal.status==='pending'&&!proposalAdopted()?A.proposal.boxes.map(box=>({...box,id:'suggestion:'+box.id,source_box_id:box.id,suggestion:true})):[];}
  function displayBoxes(){return [...A.boxes,...suggestionBoxes()];}
  function selectedBox(){return displayBoxes().find(box=>box.id===A.selected);}
  function problemOrder(){return [...new Set([...(S.tags.defect||[]),...(S.current?.label?.defects||[])])];}
  function defaultProblem(){const existing=S.current?.label?.defects||[];return [selectedBox()?.problem,existing.includes(S.responsibilityProblem)?S.responsibilityProblem:'',existing.includes(A.lastProblem)?A.lastProblem:'',...existing,A.lastProblem,el('boxProblem').value,...problemOrder()].find(name=>problemOrder().includes(name))||'';}
  function standardResponsibilities(problem,box){
    if(!problem)return [];
    if(box&&Object.hasOwn(box,'responsibilities'))return [...box.responsibilities];
    const label=S.current?.label,mapping=label?.problem_responsibilities;
    if(mapping&&Object.hasOwn(mapping,problem))return [...mapping[problem]];
    if(S.rules?.[problem])return [S.rules[problem]];
    return [...(label?.defects?.length===1&&label.defects[0]===problem?label.responsibilities||[]:[])];
  }
  function manualResponsibility(problem,box){
    const automatic=S.rules?.[problem],roles=standardResponsibilities(problem,box);
    return box?.responsibility_override===true||!!S.current?.label?.responsibility_overrides?.includes(problem)||!automatic||roles.length!==1||roles[0]!==automatic;
  }
  function responsibilityFields(problem){return {responsibilities:standardResponsibilities(problem),...(manualResponsibility(problem)?{responsibility_override:true}:{})};}
  function updatesForBoxes(boxes){
    const mapping={};for(const box of boxes)if(box.problem&&Object.hasOwn(box,'responsibilities'))mapping[box.problem]=[...new Set([...(mapping[box.problem]||[]),...box.responsibilities])];
    const defects=Object.keys(mapping),overrides=[...new Set(boxes.filter(box=>box.responsibility_override===true&&Object.hasOwn(mapping,box.problem)).map(box=>box.problem))];
    return defects.length?{defects,responsibilities:[...new Set(Object.values(mapping).flat())],problem_responsibilities:mapping,...(overrides.length?{responsibility_overrides:overrides}:{})}:null;
  }
  function conflictingRegions(box){return box?A.negativeRegions.filter(region=>overlapRatio(box,region)>=.5):[];}
  function firstConflict(boxes=displayBoxes()){for(const box of boxes){const regions=conflictingRegions(box);if(regions.length)return {box,regions};}return null;}
  function regionName(region){return region.reason==='process_hole'?'工艺孔':region.reason==='normal_structure'?'正常结构':'已删除误报区域';}
  function conflictMessage(box,regions){return '“'+(box.problem||'待选问题点')+'”与 '+regions.length+' 处'+[...new Set(regions.map(regionName))].join(' / ')+'重叠，请选择保留一项。';}
  function showConflict(conflict){
    if(!conflict)return;const {box,regions}=conflict;A.active=true;A.quickFlow=true;A.draw=false;A.selected=box.id;
    if(A.popup?.boxId!==box.id){const rect=stage.getBoundingClientRect(),point=screenPoint(box.x+box.w,box.y+box.h,view());openPopup(box,rect.left+point.x,rect.top+point.y);}
    A.popup.error=conflictMessage(box,regions);render();
  }
  function requireNoConflicts(boxes=displayBoxes()){const conflict=firstConflict(boxes);if(conflict){showConflict(conflict);throw new Error(conflictMessage(conflict.box,conflict.regions));}}
  function resolveConflict(keepProblem){
    if(!ready()||S.busy)return;const box=popupBox(),regions=conflictingRegions(box);if(!box||!regions.length)return;
    const ids=new Set(regions.map(region=>region.id));closePopup();
    change(()=>{
      if(keepProblem){A.negativeRegions=A.negativeRegions.filter(region=>!ids.has(region.id));A.selected=box.id;}
      else{if(box.suggestion)adoptSuggestions();A.boxes=A.boxes.filter(current=>current.id!==box.id&&!(box.suggestion&&current.source_box_id===box.source_box_id));A.selected='';}
      if(!A.boxes.length&&A.negativeRegions.length&&!suggestionBoxes().length)A.outcome='needs_correction';
    });
    const next=firstConflict();if(next)showConflict(next);else if(keepProblem){const rect=stage.getBoundingClientRect();openPopup(selectedBox(),rect.left+20,rect.top+20);}
    toast(keepProblem?'已保留问题，撤销 '+regions.length+' 处冲突区域；其他框保留，可撤销':'已保留记录区域，移除当前问题框；其他框保留，可撤销');
  }
  function annotationSnapshot(){return {boxes:structuredClone(A.boxes),negativeRegions:structuredClone(A.negativeRegions),outcome:A.outcome,proposalId:A.proposalId,selected:A.selected};}
  function closePopup(){A.popup=null;el('boxEditor').hidden=true;}
  function popupBox(){return displayBoxes().find(box=>box.id===A.popup?.boxId);}
  function cancelPopup(){
    const popup=A.popup;if(!popup)return;
    closePopup();
    if(popup.changed){Object.assign(A,structuredClone(popup.before));A.undo.length=Math.min(A.undo.length,popup.undoLength);A.dirty=true;A.revision++;A.saveError='';clearTimeout(A.timer);A.timer=setTimeout(()=>save().catch(fail),500);}
    render();
  }
  function popupPosition(){
    if(!A.popup)return;const editor=el('boxEditor'),width=editor.offsetWidth||540,height=editor.offsetHeight||440;
    const screenWidth=window.innerWidth||document.documentElement?.clientWidth||1280,screenHeight=window.innerHeight||document.documentElement?.clientHeight||720;
    const stageRect=stage.getBoundingClientRect(),withinPhoto=stage.clientHeight>=height+16;
    const top=withinPhoto?Math.max(8,stageRect.top+8):8;
    const bottom=withinPhoto?Math.min(screenHeight-8,stageRect.top+stage.clientHeight-8):screenHeight-8;
    const left=stage.clientWidth>=width+16?Math.max(8,stageRect.left+8):8;
    let x=A.popup.x+12,y=A.popup.y+12;
    if(x+width>screenWidth-8)x=A.popup.x-width-12;
    editor.style.left=Math.max(left,Math.min(x,screenWidth-width-8))+'px';
    editor.style.top=Math.max(top,Math.min(y,bottom-height))+'px';
  }
  function editPopup(action){
    if(!A.popup)return;A.popup.changed=true;A.popup.error='';
    change(()=>{const box=A.boxes.find(b=>b.id===A.selected);if(box){A.popup.boxId=box.id;action(box);}},true);
  }
  function renderPopup(){
    const editor=el('boxEditor'),box=popupBox();editor.hidden=!box;if(!box){A.popup=null;return;}
    const automatic=S.rules?.[box.problem],roles=standardResponsibilities(box.problem,box),conflicts=conflictingRegions(box);
    if(el('boxEditorTitle'))el('boxEditorTitle').textContent=box.suggestion?'纠正机器框':'标注问题位置';
    el('boxEditorProblems').replaceChildren();
    for(const name of problemOrder()){const button=document.createElement('button');button.type='button';button.textContent=name;button.setAttribute('aria-pressed',String(name===box.problem));button.classList.toggle('existing-problem',(S.current?.label?.defects||[]).includes(name));button.addEventListener('click',()=>{if(name===box.problem)return;editPopup(current=>{
      current.problem=name;delete current.responsibility_override;Object.assign(current,responsibilityFields(name));A.lastProblem=name;
    });});el('boxEditorProblems').append(button);}
    el('boxEditorResponsibilitySection').hidden=false;
    el('boxEditorHint').textContent=box.problem?'空格确认此框并继续标下一处 · Enter 保存整张并下一张':'待选问题点：先选择问题，再确认责任；机器仅提供位置建议';
    if(el('boxEditorAutomatic')){el('boxEditorAutomatic').hidden=!automatic;el('boxEditorAutomatic').textContent=automatic?'默认：'+automatic:'';}
    if(el('boxEditorRestoreDefault')){const restore=el('boxEditorRestoreDefault');restore.hidden=!automatic;restore.disabled=S.busy||roles.length===1&&roles[0]===automatic;restore.textContent=restore.disabled&&!S.busy?'已用默认':'恢复默认';}
    if(el('boxEditorResponsibilityHint'))el('boxEditorResponsibilityHint').textContent=!box.problem?'选择问题点后再确认责任。':manualResponsibility(box.problem,box)?'已按人工选择，可多选；不确定选待核。':'点击其他责任即可纠正默认责任。';
    if(el('boxEditorError')){const message=conflicts.length?conflictMessage(box,conflicts):A.popup?.error||'';el('boxEditorError').textContent=message;el('boxEditorError').hidden=!message;}
    if(el('boxEditorConflictActions'))el('boxEditorConflictActions').hidden=!conflicts.length;
    if(el('boxEditorKeepProblem')){el('boxEditorKeepProblem').disabled=S.busy;el('boxEditorKeepProblem').title='保留这个问题框，撤销它重叠的 '+conflicts.length+' 处记录区域；其他框与分类保留，可撤销';}
    if(el('boxEditorKeepNormal')){el('boxEditorKeepNormal').disabled=S.busy;el('boxEditorKeepNormal').title='保留这些记录区域，仅移除这个问题框；其他框与分类保留，可撤销';el('boxEditorKeepNormal').textContent=conflicts.every(region=>region.reason==='false_positive')?'保留误报判定':'保留正常区域';}
    el('boxEditorResponsibilities').replaceChildren();
    const available=[...new Set([...(S.tags.responsibility||[]),'待核'])],priority=['运输','安装','其他','待核'],later=['生产','原材料'];
    const roleOrder=[...priority.filter(name=>available.includes(name)),...available.filter(name=>!priority.includes(name)&&!later.includes(name)),...available.filter(name=>later.includes(name))];
    for(const name of roleOrder.filter(name=>name!==automatic)){const button=document.createElement('button');button.type='button';button.textContent=name;button.disabled=S.busy||!box.problem;button.setAttribute('aria-pressed',String(roles.includes(name)));button.addEventListener('click',()=>{if(!box.problem)return;editPopup(current=>{
      const currentRoles=standardResponsibilities(current.problem,current),defaultOnly=automatic&&currentRoles.length===1&&currentRoles[0]===automatic;let values=defaultOnly?[]:currentRoles;values=values.includes(name)?values.filter(v=>v!==name):name==='待核'?['待核']:[...values.filter(v=>v!=='待核'),name];current.responsibilities=values;current.responsibility_override=true;
    });});el('boxEditorResponsibilities').append(button);}
    // Keep the action reachable so missing responsibility can produce a useful inline error.
    el('boxEditorDone').disabled=S.busy;popupPosition();
  }
  function openPopup(box,x,y,before=null){if(!box)return;A.popup={boxId:box.id,x,y,before:before||annotationSnapshot(),undoLength:A.undo.length-(before?1:0),changed:!!before};renderPopup();}
  function confirmCurrentBox(){
    const box=popupBox();if(!box)return true;
    requireNoConflicts([box]);
    if(!box.problem){A.popup.error='请先为当前框选择问题点，机器尚未判断问题类型';renderPopup();throw new Error(A.popup.error);}
    const roles=standardResponsibilities(box.problem,box);
    if(!box.problem||!roles.length){A.popup.error='请先为“'+(box.problem||'当前问题')+'”选择责任，拿不准可选待核';renderPopup();throw new Error(A.popup.error);}
    if(box.suggestion||!Object.hasOwn(box,'responsibilities')||JSON.stringify(box.responsibilities)!==JSON.stringify(roles))editPopup(current=>{current.responsibilities=roles;if(manualResponsibility(current.problem,current))current.responsibility_override=true;});
    A.lastProblem=box.problem;closePopup();return true;
  }
  function continueBox(){
    if(!ready()||S.busy)return false;
    confirmCurrentBox();A.active=true;A.quickFlow=true;A.draw=true;A.selected='';render();toast('继续框选下一处；Enter 保存整张图片');return true;
  }
  function adoptSuggestions(){
    const suggestions=suggestionBoxes();if(!suggestions.length)return;
    A.proposalId=A.proposal.proposal_id;
    A.proposalFingerprint=A.proposal.content_fingerprint||null;
    for(const box of suggestions){const id=A.boxes.some(b=>b.id===box.source_box_id)?clientId():box.source_box_id;
      const {suggestion,...clean}=box;A.boxes.push({...clean,id,origin:'model',proposal_id:A.proposalId,...(box.problem&&!Object.hasOwn(box,'responsibilities')?responsibilityFields(box.problem):{})});
      if(A.selected===box.id)A.selected=id;}
    A.outcome='boxed';
  }
  async function refreshFeedback(){
    try{A.feedback=await api('/api/annotation-feedback/state');render();}catch{/* A feedback counter must not block reviewing images. */}
  }
  function overlayLabel(svg,x,y,text,color='#17212e'){
    const size=12/S.zoom,padding=4/S.zoom,height=20/S.zoom;
    const width=Array.from(text).reduce((sum,char)=>sum+(/[\u0000-\u00ff]/.test(char)?.62:1)*size,0)+padding*2;
    const top=Math.max(0,y-height-3/S.zoom);
    svg.append(svgNode('rect',{x,y:top,width,height,rx:3/S.zoom,fill:color,'fill-opacity':1,stroke:'#ffffff','stroke-width':.7/S.zoom}));
    const label=svgNode('text',{x:x+padding,y:top+height/2,fill:'#ffffff','font-size':size,'font-family':'Microsoft YaHei, sans-serif','dominant-baseline':'central'});label.textContent=text;svg.append(label);
  }
  function renderOverlay(){
    const svg=el('annotationOverlay'),v=view();svg.replaceChildren();
    const hidden=!A.active||!S.current||A.id!==S.current.id||el('photo').hidden||!v.imageWidth||(!A.boxes.length&&!A.drag&&!suggestionBoxes().length&&!A.negativeRegions.length);
    // SVGElement does not reflect HTMLElement.hidden; remove the actual attribute.
    svg.toggleAttribute('hidden',hidden);
    if(hidden)return;
    svg.setAttribute('viewBox','0 0 '+v.width+' '+v.height);svg.style.transform=el('photo').style.transform;
    const rect=imageRect(v),boxes=displayBoxes().map(box=>A.drag?.original?.id===box.id?{...box,...A.drag.box}:box);
    if(A.drag?.kind==='draw')boxes.push({...A.drag.box,id:'preview',problem:el('boxProblem').value});
    for(const region of A.negativeRegions){const x=rect.x+region.x*rect.width,y=rect.y+region.y*rect.height,conflict=displayBoxes().some(box=>overlapRatio(box,region)>=.5);
      svg.append(svgNode('rect',{x,y,width:region.w*rect.width,height:region.h*rect.height,fill:conflict?'#ffbd69':'none','fill-opacity':conflict?.32:0,stroke:'#ffbd69','stroke-width':conflict?4:2,'stroke-dasharray':conflict?'8 4':'2 5','vector-effect':'non-scaling-stroke'}));
      overlayLabel(svg,x,y,(conflict?'冲突 · ':'')+(region.reason==='process_hole'?'工艺孔 · 非破损':region.reason==='normal_structure'?'正常结构 · 非破损':'已删除误报区域'));}
    for(const [index,box] of boxes.entries()){
      const x=rect.x+box.x*rect.width,y=rect.y+box.y*rect.height,selected=box.id===A.selected||box.id==='preview';
      const color=selected?'#ff453a':box.suggestion?'#d58aff':'#00e6a0';
      svg.append(svgNode('rect',{x,y,width:box.w*rect.width,height:box.h*rect.height,fill:'none',
        stroke:'#17212e','stroke-opacity':.85,'stroke-width':selected?6:5,
        'stroke-dasharray':box.suggestion?'6 4':'none','vector-effect':'non-scaling-stroke'}));
      svg.append(svgNode('rect',{x,y,width:box.w*rect.width,height:box.h*rect.height,fill:color,
        'fill-opacity':selected?.22:.12,stroke:color,'stroke-width':selected?3.5:2.5,
        'stroke-dasharray':box.suggestion?'6 4':'none','vector-effect':'non-scaling-stroke'}));
      const roles=standardResponsibilities(box.problem,box);overlayLabel(svg,x,y,(box.suggestion?'机器建议':(index+1))+' · '+(box.problem||'待选问题点')+(roles.length?' · '+roles.join('、'):''));
      if(selected&&box.id!=='preview'){
        const size=9/v.zoom;for(const [cx,cy]of [[x,y],[x+box.w*rect.width,y],[x,y+box.h*rect.height],[x+box.w*rect.width,y+box.h*rect.height]])
          svg.append(svgNode('rect',{x:cx-size/2,y:cy-size/2,width:size,height:size,fill:color,stroke:'#ffffff','stroke-width':1.5/v.zoom}));
      }
    }
  }
  function render(){
    el('annotationBar').hidden=!S.current||!A.active||A.quickFlow;el('toggleAnnotations').setAttribute('aria-pressed',String(A.active));
    el('photoStage').classList.toggle('annotating',A.active&&A.draw);
    el('photoStage').classList.toggle('box-editing',A.active&&!A.draw&&!!A.selected);
    const chosen=el('boxProblem').value;el('boxProblem').replaceChildren();
    for(const name of S.tags.defect){const option=document.createElement('option');option.value=name;option.textContent=name;el('boxProblem').append(option);}
    el('boxProblem').value=S.tags.defect.includes(chosen)?chosen:(A.review?.missing_problems?.[0]||S.current?.label.defects[0]||S.tags.defect[0]||'');
    el('boxSelect').replaceChildren();const empty=document.createElement('option');empty.value='';empty.textContent='点击图片选框（'+displayBoxes().length+'）';el('boxSelect').append(empty);
    for(const [i,box]of displayBoxes().entries()){const option=document.createElement('option');option.value=box.id;option.textContent=(box.suggestion?'建议':i+1)+' · '+(box.problem||'待选问题点');el('boxSelect').append(option);}
    el('boxSelect').value=A.selected;el('reviewOutcome').value=A.outcome;
    const blocked=!ready()||S.busy||el('photo').hidden;
    for(const id of ['boxProblem','drawBox','boxSelect','reviewOutcome','confirmAnnotation'])el(id).disabled=blocked;
    el('deleteBox').disabled=blocked||!A.selected;el('undoBox').disabled=blocked||!A.undo.length;
    el('relabelBox').disabled=blocked||!A.selected;
    el('editBoxDetails').disabled=blocked||!A.selected;
    const machineSelected=selectedBox()?.origin==='model';
    el('processHoleBox').disabled=blocked||!machineSelected;el('normalStructureBox').disabled=blocked||!machineSelected;
    el('reloadAnnotation').disabled=A.loading||A.saving>0||S.busy;
    el('drawBox').setAttribute('aria-pressed',String(A.draw));el('drawBox').textContent=A.draw?'框选中 · 拖动原图':'开始框选';
    el('autoBoxProposal').hidden=!A.proposal;
    const adopted=proposalAdopted(),current=proposalCurrent();
    el('adoptAutoBoxes').disabled=blocked||A.rejecting||!current||A.proposal?.status!=='pending'||adopted||!A.proposal?.boxes.length;
    el('adoptAutoBoxes').textContent=adopted?'建议已加入草稿':'保留全部建议';
    el('rejectAutoBoxes').disabled=blocked||A.rejecting||!current||A.proposal?.status!=='pending'||!displayBoxes().some(box=>box.origin==='model');
    el('autoBoxSummary').textContent=!A.proposal?'':!current?'建议对应的原图或分类已变化，请重新生成建议。':A.documentStatus==='confirmed'&&adopted?'这批建议已完成人工审核。':adopted?'修改已加入草稿；Enter 确认并下一张，确认前不进入训练。':A.proposal.status==='accepted'?'这批建议已完成人工位置确认。':A.proposal.status==='rejected'?'这批机器建议已拒绝，人工分类和补标保留。':'机器建议 · '+A.proposal.boxes.length+' 框；'+(A.proposal.boxes.some(box=>!box.problem)?'点击待选框选择问题与责任，再按 Enter 确认。':'直接点击、拖动或改类，Enter 确认。');
    el('annotationStatus').textContent=A.error||A.readOnly?'读取有冲突，原补标保留':A.saveError?'保存未成功，可修改后重试':A.loading?'正在读取补标…':A.saving?'正在保存补标…':A.dirty?'改动待暂存':A.documentStatus==='confirmed'?'人工已确认 · v'+A.version:A.version?'草稿已暂存 · 尚未确认':'尚未保存位置补标';
    el('annotationFeedbackStatus').textContent=Number.isFinite(A.feedback?.confirmed_corrections)?A.feedback.confirmed_corrections+' 份纠错已确认 · 用于批次训练':'纠错在确认后纳入后续批次训练';
    el('annotationStatus').title=A.negativeRegions.length?'已记录 '+A.negativeRegions.length+' 处局部误报；不改变整图分类。':'';
    const label=S.current?.label,review=A.review;
    let text='当前标签：'+(label?.defects.join('、')||label?.image_state||'尚未分类')+'。补标单独保存。';
    const stateNames={normal:'无可见破损',related:'板件相关图',unrelated:'无关图',retain:'保留人工分类'};
    if(review){text+=' '+review.kinds.map(k=>kinds[k]||k).join(' / ')+'；评估时标签：'+(review.true_problems?.join('、')||stateNames[review.truth_state]||'见原分类')+'；评估模型候选：'+(review.candidate_problems?.join('、')||stateNames[review.predicted_state]||'无');}
    if(A.error||A.saveError)text=A.error||A.saveError;if(A.readOnly)text='原图已经变化，已保留原补标。请先核对源文件，再重新读取。';
    el('errorReviewSummary').textContent=text;el('errorReviewSummary').title=text;renderOverlay();renderPopup();
  }
  function change(action,adopt=false,prospectiveSuggestions=null){
    if(!ready()||S.busy)return false;
    if(adopt){const conflict=firstConflict(prospectiveSuggestions||suggestionBoxes());if(conflict){showConflict(conflict);toast('机器框与已有纠错区域冲突，请先选择保留一项');return false;}}
    A.undo.push({boxes:structuredClone(A.boxes),negativeRegions:structuredClone(A.negativeRegions),outcome:A.outcome,proposalId:A.proposalId,selected:A.selected});if(A.undo.length>30)A.undo.shift();
    if(adopt)adoptSuggestions();action();A.saveError='';A.dirty=true;A.revision++;render();clearTimeout(A.timer);A.timer=setTimeout(()=>save().catch(fail),500);return true;
  }
  async function save(status='draft'){
    clearTimeout(A.timer);if(!A.id||(!A.dirty&&status==='draft'))return A.tail;
    if(!ready(true))throw new Error(A.error||'补标尚未读取，或原图已变化，请重新读取');
    const id=A.id,revision=A.revision,boxes=structuredClone(A.boxes),negativeRegions=structuredClone(A.negativeRegions),outcome=A.outcome,source=A.source.sha256,fp=A.labelFingerprint,proposalId=A.proposalId,proposalFingerprint=A.proposalFingerprint,updates=updatesForBoxes(boxes);
    const task=A.tail.catch(()=>{}).then(async()=>{
      A.saving++;render();
      try{const result=await api('/api/annotations',{id,expected_version:A.version,source_sha256:source,
          label_fingerprint:fp,boxes,negative_regions:negativeRegions,outcome,status,review_run:A.review?.run_id||null,...(proposalId?{proposal_id:proposalId,...(proposalFingerprint?{expected_proposal_fingerprint:proposalFingerprint}:{})}:{}),...(updates?{classification_updates:updates,...(status==='confirmed'?{expected_classification_draft:S.current?.draft_fingerprint||null}:{})}:{})});
        if(A.id===id){A.version=result.version;A.documentStatus=status;A.saveError='';if(result.label_fingerprint)A.labelFingerprint=result.label_fingerprint;if(result.label&&S.current?.id===id){S.current.label=result.label;if(status==='confirmed'&&updates){S.current.has_draft=false;S.current.draft_fingerprint=null;}}if(A.revision===revision)A.dirty=false;render();}
        return result;
      }finally{A.saving--;render();}
    });A.tail=task;
    try{return await task;}catch(error){if(A.id===id){A.saveError=error.message;A.dirty=true;render();}throw error;}
  }
  async function loadForImage(item){
    const ticket=++A.loadTicket;closePopup();A.id=item.id;A.source=null;A.boxes=[];A.negativeRegions=[];A.documentStatus=null;A.review=null;A.undo=[];A.selected='';A.draw=false;A.drag=null;A.dirty=false;A.error='';A.saveError='';A.readOnly=false;A.version=0;A.tail=Promise.resolve();A.revision=0;A.proposal=null;A.proposalId=null;A.proposalFingerprint=null;A.rejecting=false;
    const mode=el('statusFilter').value;
    if(mode.startsWith('error_review')||mode==='annotation_pending'||mode==='annotated'||mode==='auto_boxes'){A.active=true;A.quickFlow=false;}
    // Only the current image is prefetched, so its first right-button drag is ready without opening annotation controls.
    A.loading=true;render();
    try{const result=await api('/api/annotations?id='+item.id);if(ticket!==A.loadTicket||item.id!==S.current?.id)return;
      A.source=result.source;A.boxes=result.document.boxes;A.outcome=result.document.outcome;A.version=result.version;
      A.negativeRegions=result.document.negative_regions||[];A.documentStatus=result.document.status||null;
      A.labelFingerprint=result.label_fingerprint;A.review=result.review;A.readOnly=result.source_changed;A.tail=Promise.resolve();
      A.proposal=result.proposal||null;A.proposalId=result.document.proposal_id||A.boxes.find(box=>box.proposal_id)?.proposal_id||null;
      A.proposalFingerprint=result.proposal?.content_fingerprint||result.document.proposal_content_fingerprint||null;
    }catch(error){if(ticket===A.loadTicket)A.error='补标读取失败：'+error.message;}
    finally{if(ticket===A.loadTicket){A.loading=false;render();refreshFeedback();}}
  }
  async function confirm(){
    if(!ready()||S.busy||S.navigating)return;
    if(A.drag)throw new Error('请先松开鼠标完成当前框选，再保存整张图片');
    requireNoConflicts(A.outcome==='boxed'?displayBoxes():A.boxes);
    confirmCurrentBox();
    const unassigned=(A.outcome==='boxed'?displayBoxes():A.boxes).find(box=>!box.problem);
    if(unassigned){A.active=true;A.quickFlow=true;A.draw=false;A.selected=unassigned.id;openPopup(unassigned,stage.getBoundingClientRect().left+20,stage.getBoundingClientRect().top+20);A.popup.error='请先为当前框选择问题点，机器尚未判断问题类型';render();throw new Error(A.popup.error);}
    if(A.outcome==='boxed'&&suggestionBoxes().length)change(()=>{},true);
    if(A.outcome==='boxed'&&A.boxes.some(box=>box.origin==='model'&&box.problem&&!Object.hasOwn(box,'responsibilities')))change(()=>{for(const box of A.boxes)if(box.origin==='model'&&box.problem&&!Object.hasOwn(box,'responsibilities'))Object.assign(box,responsibilityFields(box.problem));});
    if(A.outcome==='boxed'&&!A.boxes.length&&A.negativeRegions.length)A.outcome='needs_correction';
    if(A.outcome==='boxed'&&!A.boxes.length)throw new Error('尚未框选问题位置；只复核分类请使用底部“图片状态 / 分类”，或先选择其他复核结果。');
    const incomplete=A.boxes.find(box=>Object.hasOwn(box,'responsibilities')&&!box.responsibilities.length);
    if(incomplete){A.selected=incomplete.id;openPopup(incomplete,stage.getBoundingClientRect().left+20,stage.getBoundingClientRect().top+20);throw new Error('请先为“'+incomplete.problem+'”选择责任，拿不准可选待核');}
    const completedId=A.id,completedIndex=S.index;S.busy=true;renderTags();render();
    try{closePopup();await saveDraft();await save('confirmed');await refreshState();await refreshFeedback();
      const remaining=await api('/api/images?'+params({offset:completedIndex,limit:1}));
      const next=remaining.items[0]?.id===completedId?completedIndex+1:completedIndex;
      await reload({index:Math.min(next,Math.max(0,remaining.total-1))});toast('审核已保存；已有分类保留，新增问题与责任已追加');}
    finally{S.busy=false;renderTags();render();}
  }
  function undo(){if(!ready()||S.busy||!A.undo.length)return;closePopup();const previous=A.undo.pop();A.boxes=previous.boxes;A.negativeRegions=previous.negativeRegions;A.outcome=previous.outcome;A.proposalId=previous.proposalId;A.selected=previous.selected;A.dirty=true;A.revision++;render();clearTimeout(A.timer);A.timer=setTimeout(()=>save().catch(fail),500);}
  el('toggleAnnotations').addEventListener('click',()=>{(async()=>{await save();if(A.quickFlow){A.quickFlow=false;A.active=true;}else A.active=!A.active;if(A.active&&S.current&&!A.source)await loadForImage(S.current);else render();})().catch(fail);});
  el('drawBox').addEventListener('click',()=>{A.draw=!A.draw;A.selected='';render();});
  el('editBoxDetails').addEventListener('click',()=>{const box=selectedBox();if(!box)return;const rect=stage.getBoundingClientRect();openPopup(box,rect.left+20,rect.top+20);});
  el('boxEditorKeepProblem')?.addEventListener('click',()=>resolveConflict(true));
  el('boxEditorKeepNormal')?.addEventListener('click',()=>resolveConflict(false));
  el('boxEditorRestoreDefault')?.addEventListener('click',()=>{const box=popupBox(),automatic=S.rules?.[box?.problem];if(automatic)editPopup(current=>{current.responsibilities=[automatic];current.responsibility_override=true;});});
  el('boxEditorDone').addEventListener('click',()=>{try{continueBox();}catch(error){fail(error);}});
  el('boxEditorClose').addEventListener('click',()=>{try{confirmCurrentBox();A.draw=false;render();}catch(error){fail(error);}});
  el('boxSelect').addEventListener('change',()=>{A.selected=el('boxSelect').value;A.draw=false;const box=selectedBox();if(box)el('boxProblem').value=box.problem;render();});
  function markNegative(box,reason){
    A.negativeRegions.push({id:clientId(),x:box.x,y:box.y,w:box.w,h:box.h,reason,...(box.source_box_id||box.origin==='model'?{source_box_id:box.source_box_id||box.id}:{})});
  }
  function removeSelected(reason='false_positive'){
    const target=selectedBox();if(!target)return;let duplicate=false;
    const changed=change(()=>{const box=A.boxes.find(b=>b.id===A.selected);if(!box)return;
      duplicate=reason==='false_positive'&&box.origin==='model'&&A.boxes.some(other=>other.id!==box.id&&overlapRatio(box,other)>=.5);
      if(box.origin==='model'&&!duplicate)markNegative(box,reason);
      A.boxes=A.boxes.filter(b=>b.id!==A.selected);A.selected='';
      if(!A.boxes.length&&A.negativeRegions.length)A.outcome='needs_correction';},true,target.suggestion?suggestionBoxes().filter(box=>box.id!==target.id):null);
    if(changed&&duplicate)toast('已移除重复机器框，保留问题区域');
  }
  el('deleteBox').addEventListener('click',()=>removeSelected());
  el('processHoleBox').addEventListener('click',()=>{if(selectedBox()?.origin==='model')removeSelected('process_hole');});
  el('normalStructureBox').addEventListener('click',()=>{if(selectedBox()?.origin==='model')removeSelected('normal_structure');});
  function relabelSelected(){const chosen=el('boxProblem').value,selected=selectedBox();if(!selected||selected.problem===chosen)return;
    change(()=>{const box=A.boxes.find(b=>b.id===A.selected);if(box){box.problem=chosen;delete box.responsibility_override;Object.assign(box,responsibilityFields(chosen));A.lastProblem=chosen;}},true);
    if(!S.rules?.[chosen]){const rect=stage.getBoundingClientRect();openPopup(selectedBox(),rect.left+20,rect.top+20);}}
  el('relabelBox').addEventListener('click',relabelSelected);
  el('boxProblem').addEventListener('change',relabelSelected);
  el('undoBox').addEventListener('click',undo);
  el('adoptAutoBoxes').addEventListener('click',()=>{
    if(!ready()||S.busy||!proposalCurrent()||A.proposal.status!=='pending'||proposalAdopted())return;
    if(!change(()=>{A.selected=A.boxes.at(-1)?.id||'';A.draw=false;},true))return;
    toast('建议已加入草稿，可直接调整；按 Enter 确认并下一张');
  });
  el('rejectAutoBoxes').addEventListener('click',()=>{
    if(!ready()||S.busy||!proposalCurrent()||A.proposal.status!=='pending')return;
    change(()=>{const removed=A.boxes.filter(box=>box.origin==='model'&&box.proposal_id===A.proposalId);
      const kept=A.boxes.filter(box=>!removed.includes(box));for(const box of removed)if(!kept.some(other=>overlapRatio(box,other)>=.5))markNegative(box,'false_positive');A.boxes=kept;
      A.selected='';if(!A.boxes.length&&A.negativeRegions.length)A.outcome='needs_correction';},true,[]);
    toast('已移除本批机器框，已有人工框保留；可撤销，按 Enter 确认纠错');
  });
  el('reviewOutcome').addEventListener('change',()=>{const value=el('reviewOutcome').value;change(()=>{A.outcome=value;});});
  el('confirmAnnotation').addEventListener('click',()=>confirm().catch(fail));
  el('reloadAnnotation').addEventListener('click',()=>{if(A.dirty&&!global.confirm('重新读取将放弃当前未保存的改框。是否继续？'))return;if(S.current)loadForImage(S.current).catch(fail);});
  const stage=el('photoStage');
  function pointer(event){const rect=stage.getBoundingClientRect();return imagePoint(event.clientX-rect.left,event.clientY-rect.top,view());}
  function corners(box){return [{x:box.x,y:box.y},{x:box.x+box.w,y:box.y},{x:box.x,y:box.y+box.h},{x:box.x+box.w,y:box.y+box.h}];}
  function cornerHit(event,box){const rect=stage.getBoundingClientRect(),v=view();return corners(box).findIndex(c=>{const p=screenPoint(c.x,c.y,v);return Math.hypot(p.x-event.clientX+rect.left,p.y-event.clientY+rect.top)<=11;});}
  function consume(event){event.preventDefault();event.stopImmediatePropagation();}
  stage.addEventListener('contextmenu',event=>{if(!el('photo').hidden&&!event.target.closest('#boxEditor'))event.preventDefault();});
  stage.addEventListener('pointerdown',event=>{
    const right=event.button===2;
    if((!A.active&&!right)||!ready()||S.busy||el('photo').hidden||event.target.closest('button')||event.target.closest('#boxEditor')||(event.button!==undefined&&event.button!==0&&!right))return;
    if(A.drag){consume(event);return;}
    if(A.popup){if(conflictingRegions(popupBox()).length)closePopup();else try{confirmCurrentBox();}catch(error){consume(event);fail(error);return;}}
    if(right){if(!A.active)A.quickFlow=true;A.active=true;A.draw=false;}
    const point=pointer(event),selected=selectedBox(),corner=selected&&!A.draw?cornerHit(event,selected):-1;
    if((A.draw||right)&&point.inside)A.drag={kind:'draw',start:point,box:{x:point.x,y:point.y,w:0,h:0},popup:true,problem:defaultProblem(),before:annotationSnapshot()};
    else if(corner>=0)A.drag={kind:'resize',start:point,original:structuredClone(selected),corner,box:structuredClone(selected)};
    else if(!A.draw&&point.inside){
      const box=displayBoxes().reverse().find(b=>point.x>=b.x&&point.x<=b.x+b.w&&point.y>=b.y&&point.y<=b.y+b.h);
      if(box){A.selected=box.id;el('boxProblem').value=box.problem;A.drag={kind:'move',start:point,original:structuredClone(box),box:structuredClone(box)};}
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
    if(drag.kind!=='draw'&&['x','y','w','h'].every(key=>Math.abs(box[key]-drag.original[key])<1e-8)){openPopup(selectedBox(),event.clientX,event.clientY);render();return;}
    const changed=change(()=>{if(drag.kind==='draw'){const id=clientId(),problem=drag.popup?drag.problem:el('boxProblem').value;A.boxes.push({...box,id,problem,origin:'human',...(drag.popup?responsibilityFields(problem):{})});A.selected=id;A.lastProblem=problem;}
      else{const existing=A.boxes.find(b=>b.id===drag.original.id||drag.original.suggestion&&b.source_box_id===drag.original.source_box_id);if(existing){Object.assign(existing,{x:box.x,y:box.y,w:box.w,h:box.h});A.selected=existing.id;}}
      A.outcome='boxed';},true,drag.original?.suggestion?suggestionBoxes().map(candidate=>candidate.id===drag.original.id?{...candidate,...box}:candidate):null);
    if(!changed)return;
    if(drag.popup){A.draw=false;openPopup(selectedBox(),event.clientX,event.clientY,drag.before);}
  },true);
  function cancelDrag(){const pointerId=A.drag?.pointerId;A.drag=null;if(pointerId!==undefined&&stage.hasPointerCapture?.(pointerId))stage.releasePointerCapture(pointerId);renderOverlay();}
  stage.addEventListener('pointercancel',event=>{if(A.drag?.pointerId===event.pointerId)cancelDrag();},true);
  stage.addEventListener('lostpointercapture',event=>{if(A.drag?.pointerId===event.pointerId)cancelDrag();},true);
  window.addEventListener('blur',cancelDrag);
  el('photo').addEventListener('load',render);window.addEventListener('resize',()=>{renderOverlay();popupPosition();});
  document.addEventListener('keydown',event=>{
    if(!A.active||event.target.matches('input,textarea,select,[contenteditable=true]')||event.target.closest('#labelPanel'))return;
    if(event.key==='Escape'&&el('labelPanel')&&!el('labelPanel').hidden)return;
    if(A.popup&&!event.ctrlKey&&!event.altKey&&(event.code?.startsWith('Digit')||event.code?.startsWith('Numpad'))){event.preventDefault();event.stopImmediatePropagation();return;}
    if(event.ctrlKey&&event.key.toLowerCase()==='z'){event.preventDefault();event.stopImmediatePropagation();undo();}
    else if(event.key==='Delete'&&A.selected){event.preventDefault();event.stopImmediatePropagation();removeSelected();}
    else if((event.key===' '||event.code==='Space')&&(A.popup||A.draw)){event.preventDefault();event.stopImmediatePropagation();if(!event.repeat)try{continueBox();}catch(error){fail(error);}}
    else if(event.key==='Enter'&&(A.popup||!event.target.closest('button')||['confirmAnnotation','workspaceCommit'].includes(event.target.closest('button').id))){
      event.preventDefault();event.stopImmediatePropagation();(global.submitWorkspace?global.submitWorkspace():confirm()).catch(fail);
    }else if(event.key==='Escape'){event.preventDefault();event.stopImmediatePropagation();cancelPopup();A.draw=false;cancelDrag();render();}
  },true);
  window.addEventListener('beforeunload',event=>{if(A.dirty){event.preventDefault();event.returnValue='';}});
  global.AnnotationUI={submitImage:confirm,continueBox,hasWork:()=>A.id===S.current?.id&&(A.boxes.length>0||A.negativeRegions.length>0||suggestionBoxes().length>0||!!A.popup||!!A.drag),isReady:()=>ready(),isLoading:()=>!!S.current&&(A.id!==S.current.id||A.loading||!!A.error||A.readOnly||!A.source),hideEditor:closePopup,saveDraft:()=>save(),loadForImage,renderOverlay,refresh:()=>{if(A.active)render();},
    empty(){closePopup();A.id=null;A.source=null;A.boxes=[];A.negativeRegions=[];A.documentStatus=null;A.review=null;A.proposal=null;A.proposalId=null;A.proposalFingerprint=null;A.loading=false;render();}};
})(globalThis);
