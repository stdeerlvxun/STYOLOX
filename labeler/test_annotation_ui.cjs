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
  const writes=[],failures=[];let version=0,navigations=0;
  const context=vm.createContext({structuredClone,Promise,setTimeout(){return 1;},clearTimeout(){},
    document:{getElementById:el,createElement:()=>new Element(),createElementNS:()=>new Element(),addEventListener(){},body:new Element()},
    S:{current:{id:7,label:structuredClone(label)},tags:{defect:['崩边','划伤']},busy:false,navigating:false,zoom:1,rotation:0,panX:0,panY:0,index:0},
    clientId:()=>'new-box',fail:e=>failures.push(e),toast(){},renderTags(){},
    async refreshState(){},async reload(){navigations++;},
    async api(path,body){
      if(!body)return {source:{sha256:'source',width:1600,height:900},document:{boxes:[],outcome:'boxed'},version,
        label_fingerprint:'human-label',review:{run_id:'frozen-review',kinds:['problem_miss'],missing_problems:['崩边']}};
      assert.equal(path,'/api/annotations');assert.equal(body.expected_version,version);writes.push(structuredClone(body));
      version++;return {version,labels_unchanged:true};
    },window:{addEventListener(){}},confirm:()=>true});
  context.saveDraft=()=>vm.runInContext('AnnotationUI.saveDraft()',context);
  vm.runInContext(fs.readFileSync('web/annotations.js','utf8'),context);
  await vm.runInContext('AnnotationUI.loadForImage(S.current)',context);
  await el('drawBox').fire('click');
  const geometry=context.AnnotationGeometry,view={width:800,height:600,imageWidth:1600,imageHeight:900,zoom:1,rotation:0,panX:0,panY:0};
  const start=geometry.screenPoint(.1,.2,view),end=geometry.screenPoint(.4,.6,view);
  await el('photoStage').fire('pointerdown',{clientX:start.x,clientY:start.y,pointerId:1});
  await el('photoStage').fire('pointermove',{clientX:end.x,clientY:end.y,pointerId:1});
  await el('photoStage').fire('pointerup',{clientX:end.x,clientY:end.y,pointerId:1});
  await vm.runInContext('AnnotationUI.saveDraft()',context);
  assert.equal(writes[0].status,'draft');assert.equal(writes[0].boxes.length,1);
  assert.ok(Math.abs(writes[0].boxes[0].w-.3)<1e-10);assert.equal(writes[0].review_run,'frozen-review');
  await el('confirmAnnotation').fire('click');
  assert.equal(writes.at(-1).status,'confirmed');assert.equal(navigations,1);assert.equal(failures.length,0);
  assert.deepEqual(context.S.current.label,label);assert.equal(writes.filter(w=>w.status==='confirmed').length,1);
  console.log('Actual annotation UI: draw, normalized draft, confirmation, navigation and original-label preservation passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
