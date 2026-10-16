// Test routing the fixed footer through the existing data-preserving save paths.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('web/app.js','utf8');
const nodes={},el=id=>nodes[id]??={hidden:true,dataset:{},setAttribute(k,v){this[k]=v;}};
const calls=[],ctx=vm.createContext({$:el,S:{current:null,labelStep:1,busy:false,navigating:false},
  AnnotationUI:{isLoading:()=>false,hasWork:()=>true,async submitImage(){calls.push('boxes');},hideEditor(){calls.push('hide');}},
  async commit(){calls.push('classification');},renderTags(){calls.push('render');}});
vm.runInContext(source.slice(source.indexOf('function setLabelPanel('),source.indexOf('function buttonTag(')),ctx);
const run=code=>vm.runInContext(code,ctx);
(async()=>{
  run("S.current={id:1,label:{defects:['崩边'],image_state:'defect',problem_responsibilities:{崩边:['生产']}}}");
  await run('submitWorkspace()');assert.deepEqual(calls.splice(0),['boxes']);
  run("S.current.label={defects:[],image_state:'unclear',problem_responsibilities:{}}");
  await run('submitWorkspace()');assert.deepEqual(calls.splice(0),['classification']);
  run('AnnotationUI.hasWork=()=>false');run("S.current.label={defects:[],image_state:''}");
  await assert.rejects(run('submitWorkspace()'),/选择图片状态/);
  assert.equal(el('labelPanel').hidden,false);assert.equal(el('labelPanel').dataset.panelView,'state');calls.length=0;
  run("S.current.label={defects:['划伤'],image_state:'defect',problem_responsibilities:{划伤:[]}}");
  await assert.rejects(run('submitWorkspace()'),/选择责任/);assert.equal(el('labelPanel').dataset.panelView,'details');assert.equal(ctx.S.labelStep,2);assert.ok(!calls.includes('classification'));calls.length=0;
  run("S.current.label.problem_responsibilities.划伤=['运输']");
  await run('submitWorkspace()');assert.deepEqual(calls.splice(0),['classification']);
  run('AnnotationUI.isLoading=()=>true');await assert.rejects(run('submitWorkspace()'),/正在读取/);assert.equal(calls.length,0);
  run('AnnotationUI.isLoading=()=>false;AnnotationUI.hasWork=()=>true;AnnotationUI.submitImage=async()=>{throw Error("保存冲突")};setLabelPanel()');
  await assert.rejects(run('submitWorkspace()'),/保存冲突/);assert.equal(ctx.S.current.id,1);
  assert.equal(el('labelPanel').hidden,true);assert.equal(el('imageStateToggle')['aria-expanded'],'false');
  console.log('Fixed footer: boxes, explicit states, missing responsibility, load protection and failed-save preservation passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
