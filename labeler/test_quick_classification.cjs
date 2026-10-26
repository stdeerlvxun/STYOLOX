const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('web/app.js','utf8'),nodes={},handlers={},keys=[],calls=[];
const el=id=>nodes[id]??={hidden:true,dataset:{},style:{},offsetWidth:660,offsetHeight:430,setAttribute(k,v){this[k]=v;},focus(){this.focused=true;},setPointerCapture(){},releasePointerCapture(){}};
el('photo').hidden=false;
const context=vm.createContext({$:el,S:{current:{id:1,label:{defects:['崩边'],responsibilities:['生产'],problem_responsibilities:{崩边:['生产']},image_state:'defect'}},rules:{崩边:'生产'},panX:0,panY:0,labelStep:1},
 AnnotationUI:{hideEditor(){},isLoading:()=>false,hasWork:()=>true,async submitImage(){calls.push('boxes');}},
 window:{innerWidth:1280,innerHeight:720,addEventListener(){}},document:{addEventListener(name,fn){if(name==='keydown')keys.push(fn);else handlers['document:'+name]=fn;}},
 renderTags(){},edited(){calls.push('edited');},async commit(){calls.push('classification');},transform(){},on(id,event,fn){handlers[id+':'+event]=fn;},fail(error){throw error;}});
const run=code=>vm.runInContext(code,context);
run(source.slice(source.indexOf('function setLabelPanel('),source.indexOf('function buttonTag(')));
run(source.slice(source.indexOf('function syncResponsibilities('),source.indexOf('function renderResponsibilities(')));
run(source.slice(source.indexOf('let drag=null;'),source.indexOf("on('clearSelection'")));
const target={closest:()=>null,matches:()=>false};
const event=(x,y,other={})=>({clientX:x,clientY:y,pointerId:1,button:0,target,...other});
const fire=(name,e)=>handlers['photoStage:'+name](e);
(async()=>{
 fire('pointerdown',event(1000,600));fire('pointermove',event(1002,601));fire('pointerup',event(1002,601));
 assert.equal(el('labelPanel').dataset.panelView,'quick');assert.equal(el('labelPanel').hidden,false);assert.equal(el('labelPanel').focused,true);
 assert.equal(el('labelPanel').style.left,'608px');assert.equal(el('labelPanel').style.top,'159px');
 const css=fs.readFileSync('web/style.css','utf8');assert.ok(!css.includes('padding-right:calc(var(--classification-dock-width) + 24px)'));
 for(const [x,y] of [[0,0],[1279,0],[0,719],[1279,719],[640,360]]){
   run('openQuickClassification('+x+','+y+')');
   const left=parseFloat(el('labelPanel').style.left),top=parseFloat(el('labelPanel').style.top);
   assert.ok(left>=12&&left+660<=1268);assert.ok(top>=12&&top+430<=708);
 }
 const fixed=el('labelPanel').style.left;fire('pointermove',event(50,50));assert.equal(el('labelPanel').style.left,fixed,'Panel must not chase the pointer while choosing');
 run('openQuickClassification(400,300)');assert.equal(el('labelPanel').style.left,'12px','Left image half prefers the left edge when space is limited');
 run('openQuickClassification(900,300)');assert.equal(el('labelPanel').style.left,'608px','Right image half stays on the right');
 run(source.slice(source.indexOf("document.addEventListener('click',e=>{const inside=")).split('\n')[0]);
 handlers['document:click']({target:{closest:()=>null},composedPath:()=>[{id:'removed-tag-button'},{id:'labelPanel'}]});
 assert.equal(el('labelPanel').hidden,false,'Rebuilt tag buttons keep the original event path and must not close the panel');
 assert.equal(context.S.panX,0,'Small click jitter must not pan');
 run("chooseProblemResponsibility('崩边','运输')");assert.equal(JSON.stringify(context.S.current.label.problem_responsibilities.崩边),'["运输"]');
 run("chooseProblemResponsibility('崩边','安装')");assert.equal(JSON.stringify(context.S.current.label.problem_responsibilities.崩边),'["运输","安装"]');calls.length=0;
 await run('submitWorkspace(true)');assert.deepEqual(calls,['classification'],'Whole-image classification must not route into box confirmation');calls.length=0;
 run("S.current.label.problem_responsibilities.崩边=[]");await assert.rejects(run('submitWorkspace(true)'),/选择责任/);assert.equal(el('labelPanel').dataset.panelView,'quick');assert.deepEqual(calls,[]);
 run("S.current.label.problem_responsibilities.崩边=['运输'];setLabelPanel()");
 fire('pointerdown',event(200,200));fire('pointermove',event(225,220));fire('pointerup',event(225,220));assert.equal(el('labelPanel').hidden,true);assert.equal(context.S.panX,25);
 fire('pointerdown',event(200,200,{button:2}));fire('pointerup',event(200,200,{button:2}));assert.equal(el('labelPanel').hidden,true);
 fire('pointerdown',event(200,200));fire('pointercancel',event(200,200));fire('pointerup',event(200,200));assert.equal(el('labelPanel').hidden,true);
 run('openQuickClassification(300,300);setLabelPanel("details")');assert.equal(el('labelPanel').style.left,'');assert.equal(el('labelPanel').style.top,'');
 run('openQuickClassification(300,300)');
 const keyStart=source.lastIndexOf("document.addEventListener('keydown'",source.indexOf('if(e.defaultPrevented)'));
 run(source.slice(keyStart,source.indexOf("window.addEventListener('beforeunload'",keyStart)));
 let prevented=false;keys[0]({key:'Enter',target:{matches:()=>false,closest:selector=>selector==='button'?{id:'responsibility-choice'}:null},preventDefault(){prevented=true;}});
 await Promise.resolve();await Promise.resolve();assert.ok(prevented);assert.deepEqual(calls,['classification']);calls.length=0;
 context.commit=async()=>{throw Error('保存失败')};await assert.rejects(run('submitWorkspace(true)'),/保存失败/);assert.equal(context.S.current.id,1);assert.equal(el('labelPanel').hidden,false);
 console.log('Quick classification: click/pan/cancel separation, viewport bounds, editable default responsibility, multiple roles, Enter from tag buttons, independent label save, box preservation and failed-save retention passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
