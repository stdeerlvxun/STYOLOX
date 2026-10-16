const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
class Element{
  constructor(){this.children=[];this.events={};this.style={};this.value='';this.hidden=false;this.classList={toggle(){}};}
  replaceChildren(){this.children=[];}append(child){this.children.push(child);child.parent=this;}
  setAttribute(key,value){this[key]=value;}addEventListener(name,handler){(this.events[name]??=[]).push(handler);}
  closest(){return null;}contains(child){return child===this;}setPointerCapture(){}
  getBoundingClientRect(){return {left:0,top:0};}
  async fire(name,event={}){for(const handler of this.events[name]||[])await handler({target:this,preventDefault(){},stopImmediatePropagation(){},...event});}
}
(async()=>{
  const elements={},el=id=>elements[id]??=(new Element());
  el('photoStage').clientWidth=800;el('photoStage').clientHeight=600;
  el('photo').naturalWidth=1600;el('photo').naturalHeight=900;el('statusFilter').value='error_review';
  const label={defects:['崩边'],responsibilities:['生产'],image_state:'defect',note:'原备注'};
  const writes=[],failures=[],rejections=[];let version=0,navigations=0,failNext=false,clientIds=0,proposal=null,documentBoxes=[];const keys=[];
  const context=vm.createContext({structuredClone,Promise,setTimeout(){return 1;},clearTimeout(){},
    document:{getElementById:el,createElement:()=>new Element(),createElementNS:()=>new Element(),addEventListener(type,handler){if(type==='keydown')keys.push(handler);},body:new Element()},
    S:{current:{id:7,label:structuredClone(label)},tags:{defect:['崩边','划伤']},busy:false,navigating:false,zoom:1,rotation:0,panX:0,panY:0,index:0},
    clientId:()=>'new-box-'+(++clientIds),fail:e=>failures.push(e),toast(){},renderTags(){},
    async refreshState(){},async reload(){navigations++;},
    async api(path,body){
      if(!body)return {source:{sha256:'source',width:1600,height:900},document:{boxes:structuredClone(documentBoxes),outcome:'boxed'},version,proposal:structuredClone(proposal),
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
  const body=context.document.body;body.matches=()=>false;
  let intercepted=false;
  for(const handler of keys)handler({target:body,key:'Enter',ctrlKey:false,preventDefault(){intercepted=true;},stopImmediatePropagation(){}});
  assert.equal(intercepted,false,'Body Enter must reach the classification save, not an empty-box confirmation');
  await el('confirmAnnotation').fire('click');
  assert.equal(writes.length,0);assert.equal(el('reviewOutcome').disabled,false);
  assert.match(failures.pop().message,/尚未框选/);
  await el('drawBox').fire('click');
  const geometry=context.AnnotationGeometry,view={width:800,height:600,imageWidth:1600,imageHeight:900,zoom:1,rotation:0,panX:0,panY:0};
  const start=geometry.screenPoint(.1,.2,view),end=geometry.screenPoint(.4,.6,view);
  await el('photoStage').fire('pointerdown',{clientX:start.x,clientY:start.y,pointerId:1});
  await el('photoStage').fire('pointermove',{clientX:end.x,clientY:end.y,pointerId:1});
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
  assert.equal(el('rejectAutoBoxes').hidden,true);assert.equal(el('adoptAutoBoxes').disabled,true);
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
  assert.equal(rejections.length,1);assert.deepEqual(rejections[0],{id:7,proposal_id:'proposal-1'});
  assert.equal(writes.length,beforeReject,'Rejecting a suggestion must not change human annotations');
  assert.deepEqual(context.S.current.label,label);assert.equal(el('adoptAutoBoxes').disabled,true);
  proposal.label_fingerprint='stale-label';
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  assert.equal(el('adoptAutoBoxes').disabled,true);assert.match(el('autoBoxSummary').textContent,/已变化/);
  assert.equal(failures.length,0);
  console.log('Automatic-box UI: explicit adoption, preserved human boxes, transformed move/resize, undo/cancel, provenance, rejection and stale guards passed.');
  console.log('Dirty proposal adoption and move are saved during navigation before debounce; confirmed save and original classification remain intact.');
})().catch(error=>{console.error(error);process.exitCode=1;});
