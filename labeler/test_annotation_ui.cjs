const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
class Element{
  constructor(){this.children=[];this.events={};this.style={};this.value='';this.hidden=false;this.attributes={};this.classList={toggle(){}};}
  replaceChildren(){this.children=[];}append(child){this.children.push(child);child.parent=this;}
  setAttribute(key,value){this.attributes[key]=String(value);this[key]=value;}
  toggleAttribute(key,force){if(force)this.attributes[key]='';else delete this.attributes[key];return force;}
  hasAttribute(key){return Object.hasOwn(this.attributes,key);}
  addEventListener(name,handler){(this.events[name]??=[]).push(handler);}
  closest(){return null;}contains(child){return child===this;}setPointerCapture(){}
  getBoundingClientRect(){return {left:0,top:0};}
  async fire(name,event={}){for(const handler of this.events[name]||[])await handler({target:this,preventDefault(){},stopImmediatePropagation(){},...event});}
}
(async()=>{
  const elements={},el=id=>elements[id]??=(new Element());
  el('photoStage').clientWidth=800;el('photoStage').clientHeight=600;
  el('photo').naturalWidth=1600;el('photo').naturalHeight=900;el('statusFilter').value='error_review';
  el('annotationOverlay').setAttribute('hidden','');
  const label={defects:['崩边'],responsibilities:['生产'],image_state:'defect',note:'原备注'};
  const writes=[],failures=[],rejections=[];let version=0,navigations=0,failNext=false,clientIds=0,proposal=null,documentBoxes=[],documentNegatives=[],documentOutcome='boxed',documentStatus=null,documentProposalId=null;const keys=[];
  const context=vm.createContext({structuredClone,Promise,setTimeout(){return 1;},clearTimeout(){},
    document:{getElementById:el,createElement:()=>new Element(),createElementNS:()=>new Element(),addEventListener(type,handler){if(type==='keydown')keys.push(handler);},body:new Element()},
    S:{current:{id:7,label:structuredClone(label)},tags:{defect:['崩边','划伤']},busy:false,navigating:false,zoom:1,rotation:0,panX:0,panY:0,index:0},
    clientId:()=>'new-box-'+(++clientIds),fail:e=>failures.push(e),toast(){},renderTags(){},
    async refreshState(){},async reload(){navigations++;},
    async api(path,body){
      if(path==='/api/annotation-feedback/state')return {confirmed_corrections:6};
      if(!body)return {source:{sha256:'source',width:1600,height:900},document:{boxes:structuredClone(documentBoxes),negative_regions:structuredClone(documentNegatives),outcome:documentOutcome,status:documentStatus,proposal_id:documentProposalId},version,proposal:structuredClone(proposal),
        label_fingerprint:'human-label',review:{run_id:'frozen-review',kinds:['problem_miss'],missing_problems:['崩边']}};
      if(path==='/api/annotation-proposals/reject'){rejections.push(structuredClone(body));return {status:'rejected'};}
      assert.equal(path,'/api/annotations');assert.equal(body.expected_version,version);writes.push(structuredClone(body));
      if(failNext){failNext=false;throw new Error('模拟断网');}
      version++;return {version,labels_unchanged:true};
    },window:{addEventListener(){}},confirm:()=>true});
  context.saveDraft=()=>vm.runInContext('AnnotationUI.saveDraft()',context);
  vm.runInContext(fs.readFileSync('web/annotations.js','utf8'),context);
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  assert.equal(el('annotationStatus').textContent,'尚未保存位置补标');
  assert.equal(el('annotationOverlay').hasAttribute('hidden'),true,'Empty SVG overlay must remain hidden');
  const body=context.document.body;body.matches=()=>false;
  let intercepted=false;
  for(const handler of keys)handler({target:body,key:'Enter',ctrlKey:false,preventDefault(){intercepted=true;},stopImmediatePropagation(){}});
  assert.equal(intercepted,true,'Annotation mode must own Enter even from the page body');
  await new Promise(resolve=>setImmediate(resolve));assert.match(failures.pop().message,/尚未框选/);
  await el('toggleAnnotations').fire('click');await new Promise(resolve=>setImmediate(resolve));intercepted=false;
  for(const handler of keys)handler({target:body,key:'Enter',ctrlKey:false,preventDefault(){intercepted=true;},stopImmediatePropagation(){}});
  assert.equal(intercepted,false,'Classification mode must retain the existing Enter flow');
  await el('toggleAnnotations').fire('click');await new Promise(resolve=>setImmediate(resolve));
  await el('confirmAnnotation').fire('click');
  assert.equal(writes.length,0);assert.equal(el('reviewOutcome').disabled,false);
  assert.match(failures.pop().message,/尚未框选/);
  await el('drawBox').fire('click');
  const geometry=context.AnnotationGeometry,view={width:800,height:600,imageWidth:1600,imageHeight:900,zoom:1,rotation:0,panX:0,panY:0};
  const start=geometry.screenPoint(.1,.2,view),end=geometry.screenPoint(.4,.6,view);
  await el('photoStage').fire('pointerdown',{clientX:start.x,clientY:start.y,pointerId:1});
  await el('photoStage').fire('pointermove',{clientX:end.x,clientY:end.y,pointerId:1});
  assert.equal(el('annotationOverlay').hasAttribute('hidden'),false,'Drawing must remove the SVG hidden content attribute');
  assert.ok(el('annotationOverlay').children.some(node=>node.fill==='#ff453a'&&node['fill-opacity']===.22&&node['stroke-width']===3.5),'Active drawing needs a red border and visible translucent fill');
  await el('photoStage').fire('pointerup',{clientX:end.x,clientY:end.y,pointerId:1});
  failNext=true;
  await assert.rejects(vm.runInContext('AnnotationUI.saveDraft()',context),/模拟断网/);
  assert.equal(el('reviewOutcome').disabled,false);
  assert.equal(el('drawBox').disabled,false);
  await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.equal(writes[0].status,'draft');assert.equal(writes[0].boxes.length,1);
  assert.ok(Math.abs(writes[0].boxes[0].w-.3)<1e-10);assert.equal(writes[0].review_run,'frozen-review');
  await el('confirmAnnotation').fire('click');
  assert.equal(writes.at(-1).status,'confirmed');assert.equal(navigations,1);assert.equal(failures.length,0);
  assert.deepEqual(context.S.current.label,label);assert.equal(writes.filter(w=>w.status==='confirmed').length,1);
  console.log('Actual annotation UI: draw, normalized draft, confirmation, navigation and original-label preservation passed.');
  console.log('Save failure retry, empty-box validation and classification Enter routing passed.');
  proposal={proposal_id:'proposal-1',model_id:'weak-localizer',method:'patch-score',status:'pending',
    source_sha256:'source',label_fingerprint:'human-label',note:'待人工审核',
    boxes:[{id:'machine-box',problem:'崩边',x:.4,y:.3,w:.2,h:.25,score:.7,origin:'model'}]};
  documentBoxes=[{id:'human-box',problem:'划伤',x:.05,y:.05,w:.1,h:.1,origin:'human'}];
  el('statusFilter').value='auto_boxes';
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  const beforeProposal=writes.length;
  assert.equal(el('annotationBar').hidden,false);assert.equal(el('autoBoxProposal').hidden,false);
  assert.equal(el('adoptAutoBoxes').disabled,false);assert.equal(writes.length,beforeProposal,'Showing model proposals must not write a human annotation');
  assert.ok(el('annotationOverlay').children.some(node=>node['stroke-dasharray']==='6 4'));
  assert.ok(el('annotationOverlay').children.some(node=>node.stroke==='#d58aff'&&node['stroke-dasharray']==='6 4'),'Machine suggestions must be purple dashed boxes');
  assert.ok(el('annotationOverlay').children.some(node=>node.stroke==='#00e6a0'),'Other human boxes must be green');
  await el('adoptAutoBoxes').fire('click');
  assert.equal(writes.length,beforeProposal,'Adopting a proposal must remain dirty until the debounce or navigation flush');
  context.S.navigating=true;
  await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.equal(writes.length,beforeProposal+1,'Navigation must flush the current dirty proposal before the 500 ms autosave');
  assert.equal(writes.at(-1).id,context.S.current.id);assert.deepEqual(context.S.current.label,label);
  context.S.navigating=false;
  assert.equal(writes.at(-1).status,'draft');assert.equal(writes.at(-1).proposal_id,'proposal-1');
  assert.equal(writes.at(-1).boxes.length,2);assert.deepEqual(writes.at(-1).boxes[0],documentBoxes[0]);
  assert.equal(writes.at(-1).boxes[1].origin,'model');assert.equal(writes.at(-1).boxes[1].proposal_id,'proposal-1');
  assert.equal(el('adoptAutoBoxes').disabled,true);
  await el('adoptAutoBoxes').fire('click');assert.equal(el('boxSelect').children.length,3,'Repeated adoption must not duplicate boxes');
  context.S.rotation=90;context.S.zoom=1.7;context.S.panX=23;context.S.panY=-14;
  const transformed={...view,rotation:90,zoom:1.7,panX:23,panY:-14};
  async function gesture(start,end,pointerId=2){
    await el('photoStage').fire('pointerdown',{clientX:start.x,clientY:start.y,pointerId});
    await el('photoStage').fire('pointermove',{clientX:end.x,clientY:end.y,pointerId});
    await el('photoStage').fire('pointerup',{clientX:end.x,clientY:end.y,pointerId});
  }
  await gesture(geometry.screenPoint(.5,.425,transformed),geometry.screenPoint(.55,.475,transformed));
  const beforeMoveFlush=writes.length;
  context.S.navigating=true;
  await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.equal(writes.length,beforeMoveFlush+1,'Navigation must flush a moved model box before autosave');
  assert.equal(writes.at(-1).id,context.S.current.id);assert.equal(writes.at(-1).status,'draft');
  assert.deepEqual(context.S.current.label,label);context.S.navigating=false;
  let moved=writes.at(-1).boxes[1];assert.ok(Math.abs(moved.x-.45)<1e-10);assert.ok(Math.abs(moved.y-.35)<1e-10);
  assert.deepEqual(writes.at(-1).boxes[0],documentBoxes[0],'Moving a model box must preserve the human box');
  await gesture(geometry.screenPoint(moved.x+moved.w,moved.y+moved.h,transformed),geometry.screenPoint(.72,.7,transformed));
  await vm.runInContext('AnnotationUI.saveDraft()',context);
  const resized=writes.at(-1).boxes[1];assert.ok(Math.abs(resized.w-.27)<1e-10);assert.ok(Math.abs(resized.h-.35)<1e-10);
  await el('undoBox').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.deepEqual(writes.at(-1).boxes[1],moved,'Undo must restore the prior model box with provenance');
  const cancelStart=geometry.screenPoint(.55,.475,transformed),cancelEnd=geometry.screenPoint(.8,.8,transformed);
  await el('photoStage').fire('pointerdown',{clientX:cancelStart.x,clientY:cancelStart.y,pointerId:3});
  await el('photoStage').fire('pointermove',{clientX:cancelEnd.x,clientY:cancelEnd.y,pointerId:99});
  await el('photoStage').fire('pointercancel',{pointerId:3});
  await vm.runInContext('AnnotationUI.saveDraft()',context);assert.deepEqual(writes.at(-1).boxes[1],moved);
  await el('confirmAnnotation').fire('click');assert.equal(writes.at(-1).status,'confirmed');assert.equal(writes.at(-1).proposal_id,'proposal-1');
  assert.deepEqual(context.S.current.label,label);
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  const beforeReject=writes.length;
  await el('rejectAutoBoxes').fire('click');await new Promise(resolve=>setImmediate(resolve));
  assert.equal(rejections.length,0,'Whole-batch corrections stay undoable until explicit confirmation');
  assert.equal(writes.length,beforeReject,'Rejecting a suggestion must not immediately change human annotations');
  await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.deepEqual(writes.at(-1).boxes,documentBoxes,'Rejecting all model candidates must preserve human boxes');
  assert.equal(writes.at(-1).negative_regions.length,1);assert.equal(writes.at(-1).negative_regions[0].reason,'false_positive');
  assert.deepEqual(context.S.current.label,label);assert.equal(el('adoptAutoBoxes').disabled,true);
  proposal.label_fingerprint='stale-label';
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  assert.equal(el('adoptAutoBoxes').disabled,true);assert.match(el('autoBoxSummary').textContent,/已变化/);
  proposal.label_fingerprint='human-label';documentBoxes=[];
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  const beforeDirect=writes.length,point=geometry.screenPoint(.5,.425,transformed);
  await el('photoStage').fire('pointerdown',{clientX:point.x,clientY:point.y,pointerId:4});
  await el('photoStage').fire('pointerup',{clientX:point.x,clientY:point.y,pointerId:4});
  assert.equal(writes.length,beforeDirect,'Selecting a machine suggestion must not save or accept it');
  assert.equal(el('boxSelect').value,'suggestion:machine-box');assert.equal(el('processHoleBox').disabled,false);
  await gesture(point,geometry.screenPoint(.55,.475,transformed),5);
  await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.equal(writes.at(-1).proposal_id,'proposal-1');assert.equal(writes.at(-1).boxes.length,1);
  assert.equal(writes.at(-1).boxes[0].source_box_id,'machine-box');assert.ok(Math.abs(writes.at(-1).boxes[0].x-.45)<1e-10);
  assert.equal(writes.at(-1).negative_regions.length,0,'Moving a candidate must not mark its entire old ROI as normal');
  assert.match(el('annotationStatus').textContent,/草稿已暂存/);
  await el('processHoleBox').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  const holeDraft=structuredClone(writes.at(-1));
  assert.equal(holeDraft.boxes.length,0);assert.equal(holeDraft.negative_regions[0].reason,'process_hole');
  assert.equal(holeDraft.negative_regions[0].source_box_id,'machine-box');assert.equal(holeDraft.outcome,'needs_correction');
  assert.equal(holeDraft.proposal_id,'proposal-1');assert.equal(el('boxSelect').children.length,1,'Deleting the last machine candidate must not resurrect its preview');
  assert.ok(el('annotationOverlay').children.some(node=>node.textContent==='工艺孔 · 非破损'));
  await el('undoBox').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.equal(writes.at(-1).boxes.length,1);assert.equal(writes.at(-1).negative_regions.length,0,'Undo must restore both positive and negative regions');
  for(const handler of keys)handler({target:body,key:'Delete',ctrlKey:false,preventDefault(){},stopImmediatePropagation(){}});
  await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.equal(writes.at(-1).boxes.length,0);assert.equal(writes.at(-1).negative_regions[0].reason,'false_positive');
  await el('undoBox').fire('click');await el('processHoleBox').fire('click');
  for(const handler of keys)handler({target:body,key:'Enter',ctrlKey:false,preventDefault(){},stopImmediatePropagation(){}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(writes.at(-1).status,'confirmed');assert.equal(writes.at(-1).outcome,'needs_correction');
  assert.equal(writes.at(-1).negative_regions[0].reason,'process_hole');assert.deepEqual(context.S.current.label,label);
  assert.match(el('annotationStatus').textContent,/人工已确认/);
  documentBoxes=[];documentNegatives=holeDraft.negative_regions;documentOutcome='needs_correction';documentStatus='draft';documentProposalId='proposal-1';
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  assert.equal(el('boxSelect').children.length,1,'Reloading a negative-only draft must not revive removed candidates');
  assert.match(el('annotationStatus').textContent,/草稿/);
  documentNegatives=[];documentOutcome='boxed';documentStatus=null;documentProposalId=null;
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  await el('photoStage').fire('pointerdown',{clientX:point.x,clientY:point.y,pointerId:6});
  await el('photoStage').fire('pointerup',{clientX:point.x,clientY:point.y,pointerId:6});
  el('boxProblem').value='划伤';await el('boxProblem').fire('change');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.equal(writes.at(-1).boxes[0].problem,'划伤','Changing the type selector directly reclassifies a selected suggestion');
  assert.equal(writes.at(-1).boxes[0].origin,'model');assert.equal(writes.at(-1).negative_regions.length,0);
  await el('normalStructureBox').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.equal(writes.at(-1).negative_regions[0].reason,'normal_structure');
  documentBoxes=[];await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  const beforeAccept=writes.length;
  for(const handler of keys)handler({target:body,key:'Enter',ctrlKey:false,preventDefault(){},stopImmediatePropagation(){}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.ok(writes.length>beforeAccept);assert.equal(writes.at(-1).status,'confirmed');assert.equal(writes.at(-1).boxes[0].id,'machine-box','Enter can confirm unchanged visible model boxes without a separate adoption click');
  assert.deepEqual(context.S.current.label,label);
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  el('reviewOutcome').value='unclear';await el('reviewOutcome').fire('change');
  await el('confirmAnnotation').fire('click');
  assert.equal(writes.at(-1).outcome,'unclear');assert.equal(writes.at(-1).boxes.length,0,'Unclear images must not silently adopt machine boxes as positive training annotations');
  assert.equal(failures.length,0);
  console.log('Automatic-box UI: explicit adoption, preserved human boxes, transformed move/resize, undo/cancel, provenance, rejection and stale guards passed.');
  console.log('Dirty proposal adoption and move are saved during navigation before debounce; confirmed save and original classification remain intact.');
  console.log('Direct suggestion correction, local process-hole/normal feedback, Delete/Enter shortcuts, negative-only draft reload, undo, and explicit batch-training status passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
