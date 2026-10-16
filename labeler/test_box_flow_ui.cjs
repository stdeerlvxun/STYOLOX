const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
class Element{
  constructor(tag='div',id=''){this.tag=tag;this.id=id;this.children=[];this.events={};this.style={};this.value='';this.hidden=false;this.attributes={};this.classList={toggle(){}};}
  replaceChildren(){this.children=[];}append(child){this.children.push(child);child.parent=this;}
  setAttribute(key,value){this.attributes[key]=String(value);this[key]=value;}
  toggleAttribute(key,force){if(force)this.attributes[key]='';else delete this.attributes[key];return force;}
  addEventListener(name,handler){(this.events[name]??=[]).push(handler);}
  closest(selector){for(let node=this;node;node=node.parent)if(selector==='button'&&node.tag==='button'||selector==='#boxEditor'&&node.id==='boxEditor')return node;return null;}
  matches(){return false;}setPointerCapture(){}releasePointerCapture(){}
  getBoundingClientRect(){return {left:0,top:0};}
  async fire(name,event={}){for(const handler of this.events[name]||[])await handler({target:this,preventDefault(){},stopImmediatePropagation(){},...event});}
}
(async()=>{
  const elements={},el=id=>elements[id]??=new Element('div',id),keys=[],writes=[],failures=[];
  el('photoStage').clientWidth=800;el('photoStage').clientHeight=600;el('photo').naturalWidth=1600;el('photo').naturalHeight=900;
  el('statusFilter').value='all';el('boxEditor').offsetWidth=342;el('boxEditor').offsetHeight=360;
  el('labelPanel').hidden=true;
  const label={defects:['崩边','划伤'],responsibilities:['生产'],problem_responsibilities:{崩边:['生产'],划伤:[]},image_state:'defect',note:'原备注'};
  const human={id:'retained-human',problem:'崩边',x:.01,y:.01,w:.08,h:.08,origin:'human'};
  let fixtureBoxes=[structuredClone(human)],fixtureRegions=[],fixtureProposal=null,fixtureProposalId=null;
  let version=0,ids=0,navigations=0,reads=0,rejectConfirmation=false,retainedInQueue=true;const navigationIndexes=[];
  const context=vm.createContext({structuredClone,Promise,setTimeout(){return 1;},clearTimeout(){},
    document:{getElementById:el,createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),addEventListener(type,handler){if(type==='keydown')keys.push(handler);},body:new Element('body')},
    window:{innerWidth:840,innerHeight:620,addEventListener(){}},
    S:{current:{id:7,label:structuredClone(label),has_draft:true,draft_fingerprint:'visible-draft'},tags:{defect:['残胶','划伤','崩边'],responsibility:['生产','运输','原材料']},rules:{崩边:'生产',残胶:'生产'},busy:false,navigating:false,zoom:1,rotation:0,panX:0,panY:0,index:0},
    clientId:()=>'new-'+(++ids),fail:e=>failures.push(e),toast(){},renderTags(){},params:args=>new URLSearchParams(args).toString(),async refreshState(){},async reload({index}){navigations++;navigationIndexes.push(index);},
    async api(path,body){if(path==='/api/annotation-feedback/state')return {confirmed_corrections:6};
      if(path.startsWith('/api/images?'))return {items:[{id:retainedInQueue?7:8}],total:3};
      if(!body){reads++;return {source:{sha256:'source',width:1600,height:900},document:{boxes:structuredClone(fixtureBoxes),negative_regions:structuredClone(fixtureRegions),outcome:'boxed',proposal_id:fixtureProposalId},proposal:structuredClone(fixtureProposal),version,label_fingerprint:'old-label'};}
      assert.equal(path,'/api/annotations');assert.equal(body.expected_version,version);writes.push(structuredClone(body));
      if(body.status==='confirmed'&&rejectConfirmation){rejectConfirmation=false;throw new Error('分类草稿已经变化，请重新读取');}
      version++;return {version,label_fingerprint:body.status==='confirmed'?'confirmed-label':'old-label',...(body.status==='confirmed'?{label:structuredClone(label)}:{})};
    }});
  context.saveDraft=()=>vm.runInContext('AnnotationUI.saveDraft()',context);
  vm.runInContext(fs.readFileSync('web/annotations.js','utf8'),context);
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  assert.equal(reads,1,'Current image metadata is ready even outside annotation mode');
  assert.equal(el('annotationBar').hidden,true,'Prefetch must not open annotation controls');
  const geometry=context.AnnotationGeometry,view={width:800,height:600,imageWidth:1600,imageHeight:900,zoom:1,rotation:0,panX:0,panY:0};
  async function draw(a,b,id,mouseButton=2){const start=geometry.screenPoint(...a,view),end=geometry.screenPoint(...b,view);
    await el('photoStage').fire('pointerdown',{button:mouseButton,clientX:start.x,clientY:start.y,pointerId:id});
    await el('photoStage').fire('pointermove',{button:mouseButton,clientX:end.x,clientY:end.y,pointerId:id});
    await el('photoStage').fire('pointerup',{button:mouseButton,clientX:end.x,clientY:end.y,pointerId:id});}
  function button(id,name){return el(id).children.find(b=>b.textContent===name);}
  async function key(keyName,target=context.document.body){let intercepted=false;for(const handler of keys)handler({target,key:keyName,code:keyName===' '?'Space':keyName,ctrlKey:false,preventDefault(){intercepted=true;},stopImmediatePropagation(){}});await new Promise(resolve=>setImmediate(resolve));return intercepted;}
  async function clickPoint(x,y,id){const point=geometry.screenPoint(x,y,view);await el('photoStage').fire('pointerdown',{button:0,clientX:point.x,clientY:point.y,pointerId:id});await el('photoStage').fire('pointerup',{button:0,clientX:point.x,clientY:point.y,pointerId:id});}
  await draw([.2,.2],[.45,.5],1);
  assert.equal(el('annotationBar').hidden,true,'First quick draw must keep the canvas height stable instead of inserting a toolbar mid-drag');assert.equal(el('boxEditor').hidden,false);
  assert.deepEqual(el('boxEditorProblems').children.map(b=>b.textContent),['残胶','划伤','崩边'],'Problem order stays fixed while existing problems remain selected');
  assert.equal(button('boxEditorProblems','崩边').attributes['aria-pressed'],'true','Existing problem is selected without another click');
  assert.match(el('boxEditorAutomatic').textContent,/默认：生产/);
  assert.equal(el('boxEditorResponsibilitySection').hidden,false);assert.equal(button('boxEditorResponsibilities','生产'),undefined,'Automatic role is text below instead of a leading button');
  assert.deepEqual(el('boxEditorResponsibilities').children.map(b=>b.textContent),['运输','待核','原材料']);assert.equal(button('boxEditorResponsibilities','运输').disabled,false,'Alternative roles must remain clickable for standard problems');
  assert.ok(parseFloat(el('boxEditor').style.left)<=490&&parseFloat(el('boxEditor').style.top)<=252,'Popup stays inside viewport');
  await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.equal(writes.at(-1).status,'draft');assert.deepEqual(writes.at(-1).boxes[0],human,'Adding a box preserves existing human coordinates and metadata');
  assert.deepEqual(writes.at(-1).boxes[1].responsibilities,['生产']);
  await button('boxEditorResponsibilities','运输').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.deepEqual(writes.at(-1).boxes[1].responsibilities,['运输'],'First correction replaces the automatic production role instead of adding to it');
  assert.equal(writes.at(-1).boxes[1].problem,'崩边');assert.equal(writes.at(-1).boxes[1].responsibility_override,true);
  assert.deepEqual(writes.at(-1).classification_updates.responsibility_overrides,['崩边']);
  await button('boxEditorResponsibilities','原材料').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.deepEqual(writes.at(-1).boxes[1].responsibilities,['运输','原材料'],'Further explicit human roles support multiple selection');
  await button('boxEditorResponsibilities','待核').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);assert.deepEqual(writes.at(-1).boxes[1].responsibilities,['待核']);
  await button('boxEditorResponsibilities','待核').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.deepEqual(writes.at(-1).boxes[1].responsibilities,[],'Clearing explicit choices cannot silently restore an automatic role');
  assert.equal(await key(' '),true);assert.match(failures.pop().message,/选择责任/);assert.equal(el('boxEditorError').hidden,false);assert.match(el('boxEditorError').textContent,/选择责任/);
  await el('boxEditorRestoreDefault').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.deepEqual(writes.at(-1).boxes[1].responsibilities,['生产']);assert.equal(writes.at(-1).boxes[1].responsibility_override,true,'Restoring the default is itself an explicit update to the previous human mapping');assert.equal(el('boxEditorError').hidden,true);
  await button('boxEditorResponsibilities','运输').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);assert.deepEqual(writes.at(-1).boxes[1].responsibilities,['运输'],'The first correction after restoring default also replaces that default');
  await button('boxEditorProblems','划伤').fire('click');
  assert.equal(el('boxEditor').hidden,false);assert.equal(el('boxEditorResponsibilitySection').hidden,false);
  for(const handler of keys)handler({target:context.document.body,key:'Enter',code:'Enter',ctrlKey:false,preventDefault(){},stopImmediatePropagation(){}});
  await new Promise(resolve=>setImmediate(resolve));
  assert.match(failures.pop().message,/选择责任/);assert.equal(navigations,0);assert.equal(writes.filter(w=>w.status==='confirmed').length,0);
  await button('boxEditorResponsibilities','运输').fire('click');await button('boxEditorResponsibilities','生产').fire('click');
  await vm.runInContext('AnnotationUI.saveDraft()',context);assert.deepEqual(writes.at(-1).boxes[1].responsibilities,['运输','生产']);
  await button('boxEditorResponsibilities','待核').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);assert.deepEqual(writes.at(-1).boxes[1].responsibilities,['待核'],'待核 is exclusive');
  await button('boxEditorResponsibilities','运输').fire('click');
  await el('boxEditorDone').fire('click');assert.equal(el('boxEditor').hidden,true);
  await draw([.5,.3],[.8,.7],2);await button('boxEditorProblems','崩边').fire('click');
  assert.equal(el('boxEditor').hidden,false,'Automatic responsibility must leave the complete two-column editor open');
  await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.deepEqual(writes.at(-1).boxes.map(b=>b.id),['retained-human','new-1','new-2']);
  assert.deepEqual(writes.at(-1).boxes[2].responsibilities,['生产']);
  rejectConfirmation=true;await el('confirmAnnotation').fire('click');
  assert.equal(navigations,0,'A rejected save must not navigate away');assert.match(failures.pop().message,/草稿已经变化/);
  await el('confirmAnnotation').fire('click');
  assert.equal(navigations,1);const confirmed=writes.at(-1);assert.equal(confirmed.status,'confirmed');
  assert.equal(confirmed.expected_classification_draft,'visible-draft','Confirm binds the visible classification draft');
  assert.deepEqual(confirmed.classification_updates,{defects:['划伤','崩边'],responsibilities:['运输','生产'],problem_responsibilities:{划伤:['运输'],崩边:['生产']},responsibility_overrides:['划伤']});
  assert.equal(context.S.current.has_draft,false);assert.equal(context.S.current.draft_fingerprint,null);
  assert.equal(navigationIndexes.at(-1),1,'An image retained in the current filter advances one position');
  assert.deepEqual(confirmed.boxes[0],human);assert.equal(failures.length,0);
  retainedInQueue=false;await el('confirmAnnotation').fire('click');assert.equal(navigationIndexes.at(-1),0,'A completed image removed from pending filters must not skip its successor');
  // Space finishes one region, left-drag continues, Escape rolls back only the current edit.
  await draw([.2,.6],[.4,.8],20);await button('boxEditorProblems','划伤').fire('click');
  const beforeMissingSpace=writes.length;assert.equal(await key(' '),true);assert.match(failures.pop().message,/选择责任/);assert.equal(el('boxEditor').hidden,false);assert.equal(writes.length,beforeMissingSpace);
  await button('boxEditorResponsibilities','运输').fire('click');
  assert.equal(await key(' '),true);assert.equal(el('boxEditor').hidden,true);await vm.runInContext('AnnotationUI.saveDraft()',context);
  const acceptedBeforeCancel=structuredClone(writes.at(-1).boxes),leftStartCount=acceptedBeforeCancel.length;
  await draw([.55,.6],[.75,.8],21,0);assert.equal(el('boxEditor').hidden,false,'Space enables the next left-button drag');
  await vm.runInContext('AnnotationUI.saveDraft()',context);assert.equal(writes.at(-1).boxes.length,leftStartCount+1);
  const noteTarget=new Element('textarea','note');noteTarget.matches=selector=>selector.includes('textarea');assert.equal(await key(' ',noteTarget),false,'Space in a note remains text input');assert.equal(el('boxEditor').hidden,false);
  await key('Escape');await vm.runInContext('AnnotationUI.saveDraft()',context);assert.deepEqual(writes.at(-1).boxes,acceptedBeforeCancel,'Escape after autosave removes only the new provisional region');
  await clickPoint(.3,.7,22);assert.equal(el('boxEditor').hidden,false,'Clicking an existing region opens its editor');
  await button('boxEditorProblems','残胶').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);assert.equal(writes.at(-1).boxes.at(-1).problem,'残胶');
  assert.ok(el('annotationOverlay').children.some(node=>node.fill==='#17212e'&&node['fill-opacity']===1),'Overlay labels have solid backgrounds');
  await key('Escape');await vm.runInContext('AnnotationUI.saveDraft()',context);assert.deepEqual(writes.at(-1).boxes,acceptedBeforeCancel,'Escape restores the edited existing region instead of deleting it');
  assert.equal(vm.runInContext('AnnotationUI.hasWork()',context),true);assert.equal(vm.runInContext('AnnotationUI.isLoading()',context),false);
  assert.equal(failures.length,0);
  // Reload/default construction must honour a human problem mapping before the automatic rule.
  context.S.current.label.problem_responsibilities.崩边=['运输'];await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  await draw([.2,.2],[.45,.5],30);await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.deepEqual(writes.at(-1).boxes.at(-1).responsibilities,['运输']);assert.equal(writes.at(-1).boxes.at(-1).responsibility_override,true);
  await el('boxEditorRestoreDefault').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);assert.deepEqual(writes.at(-1).boxes.at(-1).responsibilities,['生产'],'Explicit current box override has priority over prior image mapping');
  context.S.current.label.problem_responsibilities.崩边=[];await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  await draw([.2,.2],[.45,.5],31);await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.deepEqual(writes.at(-1).boxes.at(-1).responsibilities,[],'An explicit empty image mapping is not silently replaced by the default');
  await key('Enter');assert.match(failures.pop().message,/选择责任/);assert.equal(el('boxEditor').hidden,false);
  assert.equal(failures.length,0);
  // Conflict recovery mirrors the server: intersection / smaller area >= 0.5.
  const unit={x:0,y:0,w:1,h:1};assert.equal(geometry.overlapRatio(unit,{x:.5,y:0,w:1,h:1}),.5);assert.ok(geometry.overlapRatio(unit,{x:.51,y:0,w:1,h:1})<.5);
  context.S.current.label=structuredClone(label);
  const conflictingBox={id:'current-defect',problem:'崩边',x:.35,y:.3,w:.3,h:.3,origin:'model',responsibilities:['生产']};
  const firstNegative={id:'deleted-fp-1',x:.35,y:.3,w:.3,h:.3,reason:'false_positive'},secondNegative={id:'deleted-fp-2',x:.36,y:.31,w:.28,h:.28,reason:'false_positive'},unrelatedNegative={id:'keep-normal-away',x:.8,y:.75,w:.08,h:.08,reason:'normal_structure'};
  fixtureBoxes=[structuredClone(human),structuredClone(conflictingBox)];fixtureRegions=[firstNegative,secondNegative,unrelatedNegative];
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);const conflictWrites=writes.length,conflictNavigations=navigations;
  await el('confirmAnnotation').fire('click');assert.match(failures.pop().message,/2 处已删除误报区域/);assert.equal(writes.length,conflictWrites,'Conflict is explained before any confirm API or implicit candidate adoption');assert.equal(navigations,conflictNavigations);
  assert.equal(el('boxSelect').value,conflictingBox.id);assert.equal(el('boxEditorConflictActions').hidden,false);assert.match(el('boxEditorError').textContent,/已删除误报区域/);assert.equal(el('boxEditorKeepNormal').textContent,'保留误报判定');
  assert.ok(el('annotationOverlay').children.some(node=>node.fill==='#ffbd69'&&node['fill-opacity']===.32),'Conflicting negative regions get a visible orange fill');
  await el('boxEditorKeepProblem').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.deepEqual(writes.at(-1).boxes,fixtureBoxes);assert.deepEqual(writes.at(-1).negative_regions,[unrelatedNegative],'Keep problem removes all negatives conflicting with this one problem, preserving unrelated regions');
  await el('undoBox').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);assert.deepEqual(writes.at(-1).negative_regions,fixtureRegions,'Conflict decisions are undoable including every removed ROI');
  await el('confirmAnnotation').fire('click');failures.pop();await el('boxEditorKeepNormal').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.deepEqual(writes.at(-1).boxes,[human]);assert.deepEqual(writes.at(-1).negative_regions,fixtureRegions);assert.deepEqual(context.S.current.label,label,'Removing a conflicting box never deletes image classification');
  await el('undoBox').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);assert.deepEqual(writes.at(-1).boxes,fixtureBoxes);

  // A pending model box overlapping a known-normal ROI remains pending until explicit resolution.
  const proposalBox={id:'machine-pending',problem:'崩边',x:.4,y:.3,w:.2,h:.25,origin:'model',score:.7};
  fixtureBoxes=[structuredClone(human)];fixtureRegions=[{id:'known-hole',x:.4,y:.3,w:.2,h:.25,reason:'process_hole'},unrelatedNegative];fixtureProposal={proposal_id:'conflict-proposal',model_id:'test',method:'test',status:'pending',source_sha256:'source',label_fingerprint:'old-label',boxes:[proposalBox]};
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);const pendingWrites=writes.length;
  await el('confirmAnnotation').fire('click');assert.match(failures.pop().message,/工艺孔/);assert.equal(writes.length,pendingWrites);assert.equal(el('boxSelect').value,'suggestion:machine-pending');
  await el('adoptAutoBoxes').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);assert.equal(writes.length,pendingWrites,'Keep-all does not silently adopt conflicting machine suggestions');
  await el('boxEditorKeepProblem').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);assert.deepEqual(writes.at(-1).boxes,[human],'Resolving normal feedback does not yet accept a machine candidate');assert.deepEqual(writes.at(-1).negative_regions,[unrelatedNegative]);
  await el('confirmAnnotation').fire('click');assert.equal(writes.at(-1).status,'confirmed');assert.equal(writes.at(-1).boxes.length,2);assert.deepEqual(writes.at(-1).boxes[0],human);

  // Removing a duplicate machine rectangle does not turn the remaining positive region into a negative ROI.
  const duplicateHuman={...human,id:'human-overlap',x:.4,y:.3,w:.2,h:.25};
  fixtureBoxes=[duplicateHuman];fixtureRegions=[];fixtureProposalId=null;context.S.current.label=structuredClone(label);
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  await clickPoint(.5,.425,40);await el('deleteBox').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.deepEqual(writes.at(-1).boxes,[duplicateHuman]);assert.equal(writes.at(-1).negative_regions.length,0,'Ordinary duplicate deletion records proposal removal without negative ROI');
  await el('undoBox').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);assert.deepEqual(writes.at(-1).boxes,[duplicateHuman]);
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);await el('rejectAutoBoxes').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.deepEqual(writes.at(-1).boxes,[duplicateHuman]);assert.equal(writes.at(-1).negative_regions.length,0,'Reject-all must protect overlapping human positives');
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);await clickPoint(.5,.425,41);await el('processHoleBox').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.equal(writes.at(-1).negative_regions[0].reason,'process_hole','An explicit process-hole correction is retained even when a human positive overlaps it');
  await el('confirmAnnotation').fire('click');assert.match(failures.pop().message,/工艺孔/);assert.equal(el('boxEditorConflictActions').hidden,false);
  assert.equal(failures.length,0);
  // One conflicting and one unrelated candidate: explicit repair must preserve the second candidate.
  fixtureBoxes=[structuredClone(human)];fixtureRegions=[{id:'known-hole-2',x:.4,y:.3,w:.2,h:.25,reason:'process_hole'}];
  const secondCandidate={...proposalBox,id:'machine-unrelated',x:.75,y:.1,w:.1,h:.1};fixtureProposal.boxes=[proposalBox,secondCandidate];
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  await el('confirmAnnotation').fire('click');failures.pop();await el('boxEditorKeepNormal').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.deepEqual(writes.at(-1).boxes.map(box=>box.id),['retained-human','machine-unrelated']);assert.deepEqual(writes.at(-1).negative_regions,fixtureRegions);
  await el('confirmAnnotation').fire('click');assert.equal(writes.at(-1).status,'confirmed');
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  await el('confirmAnnotation').fire('click');assert.match(failures.pop().message,/工艺孔/);assert.equal(el('boxEditorConflictActions').hidden,false);
  const moveStart=geometry.screenPoint(.5,.425,view),moveEnd=geometry.screenPoint(.25,.8,view);
  await el('photoStage').fire('pointerdown',{button:0,clientX:moveStart.x,clientY:moveStart.y,pointerId:50});
  await el('photoStage').fire('pointermove',{button:0,clientX:moveEnd.x,clientY:moveEnd.y,pointerId:50});
  await el('photoStage').fire('pointerup',{button:0,clientX:moveEnd.x,clientY:moveEnd.y,pointerId:50});await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.equal(writes.at(-1).boxes.length,3,'Moving the conflicting candidate away retains both candidates for review');assert.ok(Math.abs(writes.at(-1).boxes.find(box=>box.id==='machine-pending').x-.15)<1e-10);
  await el('confirmAnnotation').fire('click');assert.equal(writes.at(-1).status,'confirmed');
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);await clickPoint(.5,.425,51);await el('deleteBox').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.deepEqual(writes.at(-1).boxes.map(box=>box.id),['retained-human','machine-unrelated'],'Explicit Delete of the conflicting candidate can complete while preserving the unrelated one');
  await el('confirmAnnotation').fire('click');assert.equal(writes.at(-1).status,'confirmed');assert.equal(failures.length,0);
  // All-image proposals locate possible defects without fabricating a type or a responsibility.
  context.S.current.label={defects:[],responsibilities:[],problem_responsibilities:{},image_state:'',note:'未分类原备注'};
  const unclassifiedLabel=structuredClone(context.S.current.label);
  fixtureBoxes=[];fixtureRegions=[];fixtureProposalId=null;el('statusFilter').value='auto_boxes';
  fixtureProposal={proposal_id:'unclassified-proposal',model_id:'test',method:'test',status:'pending',source_sha256:'source',label_fingerprint:'old-label',boxes:[{...proposalBox,id:'unknown-1',problem:''},{...secondCandidate,id:'unknown-2',problem:''}]};
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  assert.ok(el('annotationOverlay').children.some(node=>node.textContent==='机器建议 · 待选问题点'),'Unknown model locations must be clearly labelled without inventing an issue');
  const unknownWrites=writes.length,unknownNavigations=navigations;
  await key('Enter');assert.match(failures.pop().message,/选择问题点/);assert.equal(writes.length,unknownWrites);assert.equal(navigations,unknownNavigations);
  assert.equal(el('boxSelect').value,'suggestion:unknown-1');assert.equal(el('boxEditor').hidden,false);
  assert.ok(el('boxEditorProblems').children.every(button=>button.attributes['aria-pressed']==='false'),'No problem is selected from tag order or past image');
  assert.ok(el('boxEditorResponsibilities').children.every(button=>button.disabled),'Responsibilities wait for an explicit problem choice');
  assert.equal(el('boxEditorAutomatic').hidden,true);
  await el('adoptAutoBoxes').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.equal(writes.at(-1).status,'draft');assert.ok(writes.at(-1).boxes.every(box=>box.problem===''));
  assert.equal(Object.hasOwn(writes.at(-1),'classification_updates'),false,'Unknown drafts must never create a blank classification key');
  assert.deepEqual(context.S.current.label,unclassifiedLabel);
  fixtureBoxes=structuredClone(writes.at(-1).boxes);fixtureProposalId='unclassified-proposal';
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  assert.equal(el('boxSelect').children.length,3,'Reloaded unknown drafts retain two boxes without duplicated previews');
  await key('Enter');assert.match(failures.pop().message,/选择问题点/);assert.equal(el('boxSelect').value,'unknown-1');
  await button('boxEditorProblems','崩边').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.equal(writes.at(-1).boxes[0].problem,'崩边');assert.deepEqual(writes.at(-1).boxes[0].responsibilities,['生产']);
  assert.equal(writes.at(-1).boxes[1].problem,'');assert.deepEqual(writes.at(-1).classification_updates.defects,['崩边']);
  await key('Enter');assert.match(failures.pop().message,/选择问题点/);assert.equal(el('boxSelect').value,'unknown-2');assert.equal(navigations,unknownNavigations);
  await button('boxEditorProblems','划伤').fire('click');await key('Enter');assert.match(failures.pop().message,/选择责任/);
  await button('boxEditorResponsibilities','运输').fire('click');await key('Enter');
  assert.equal(writes.at(-1).status,'confirmed');assert.deepEqual(writes.at(-1).classification_updates.problem_responsibilities,{崩边:['生产'],划伤:['运输']});
  assert.equal(navigations,unknownNavigations+1);
  // Multiple existing issues are not enough to choose a type for an unknown model location.
  context.S.current.label=structuredClone(label);fixtureBoxes=[structuredClone(human)];fixtureProposalId=null;
  fixtureProposal.boxes=[{...proposalBox,id:'multiple-unknown',problem:''}];await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  await clickPoint(.5,.425,60);assert.ok(el('boxEditorProblems').children.every(button=>button.attributes['aria-pressed']==='false'));
  await key(' ');assert.match(failures.pop().message,/选择问题点/);
  await button('boxEditorProblems','崩边').fire('click');await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.deepEqual(writes.at(-1).boxes[0],human,'Assigning a new candidate preserves every existing human box');
  assert.deepEqual(context.S.current.label,label);assert.equal(failures.length,0);
  console.log('Untyped all-image boxes: explicit type and responsibility, Enter focus/error, resumable draft, empty classification prevention and human-box preservation passed.');
  console.log('Conflict repair, move and Delete preserve other model candidates and can proceed to confirmation.');
  console.log('Pre-submit overlap detection, exact conflict highlighting, explicit recovery/undo, pending proposal protection and duplicate-deletion prevention passed.');
  console.log('Automatic responsibility correction, explicit override payload, later multi-select, empty-choice protection and restore-default passed.');
  console.log('Space continuation, missing-role interception, note-input guard, click-to-edit, opaque labels and Escape snapshot restoration passed.');
  console.log('Right-button first draw, existing-problem default, standard responsibility, shared-problem multi-select, viewport bounds, draft preservation, atomic confirmation and save-failure navigation guard passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
