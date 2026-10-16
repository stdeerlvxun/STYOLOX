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
  let version=0,ids=0,navigations=0,reads=0,rejectConfirmation=false,retainedInQueue=true;const navigationIndexes=[];
  const context=vm.createContext({structuredClone,Promise,setTimeout(){return 1;},clearTimeout(){},
    document:{getElementById:el,createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),addEventListener(type,handler){if(type==='keydown')keys.push(handler);},body:new Element('body')},
    window:{innerWidth:840,innerHeight:620,addEventListener(){}},
    S:{current:{id:7,label:structuredClone(label),has_draft:true,draft_fingerprint:'visible-draft'},tags:{defect:['残胶','划伤','崩边'],responsibility:['生产','运输','原材料']},rules:{崩边:'生产',残胶:'生产'},busy:false,navigating:false,zoom:1,rotation:0,panX:0,panY:0,index:0},
    clientId:()=>'new-'+(++ids),fail:e=>failures.push(e),toast(){},renderTags(){},params:args=>new URLSearchParams(args).toString(),async refreshState(){},async reload({index}){navigations++;navigationIndexes.push(index);},
    async api(path,body){if(path==='/api/annotation-feedback/state')return {confirmed_corrections:6};
      if(path.startsWith('/api/images?'))return {items:[{id:retainedInQueue?7:8}],total:3};
      if(!body){reads++;return {source:{sha256:'source',width:1600,height:900},document:{boxes:[structuredClone(human)],negative_regions:[],outcome:'boxed'},version,label_fingerprint:'old-label'};}
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
  assert.match(el('boxEditorAutomatic').textContent,/按标准自动：生产/);
  assert.equal(el('boxEditorResponsibilitySection').hidden,false);assert.equal(button('boxEditorResponsibilities','生产'),undefined,'Automatic role is text below instead of a leading button');
  assert.deepEqual(el('boxEditorResponsibilities').children.map(b=>b.textContent),['运输','待核','原材料']);assert.equal(button('boxEditorResponsibilities','运输').disabled,true);
  assert.ok(parseFloat(el('boxEditor').style.left)<=490&&parseFloat(el('boxEditor').style.top)<=252,'Popup stays inside viewport');
  await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.equal(writes.at(-1).status,'draft');assert.deepEqual(writes.at(-1).boxes[0],human,'Adding a box preserves existing human coordinates and metadata');
  assert.deepEqual(writes.at(-1).boxes[1].responsibilities,['生产']);
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
  assert.deepEqual(confirmed.classification_updates,{defects:['划伤','崩边'],responsibilities:['运输','生产'],problem_responsibilities:{划伤:['运输'],崩边:['生产']}});
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
  console.log('Space continuation, missing-role interception, note-input guard, click-to-edit, opaque labels and Escape snapshot restoration passed.');
  console.log('Right-button first draw, existing-problem default, standard responsibility, shared-problem multi-select, viewport bounds, draft preservation, atomic confirmation and save-failure navigation guard passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
