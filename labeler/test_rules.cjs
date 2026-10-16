'use strict';
// Exercise the actual frontend functions without opening or writing the production store.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('web/app.js','utf8');
const context=vm.createContext({S:{rules:{残胶:'生产',五金质量:'原材料',崩边:'生产'},current:null,busy:false,navigating:false,responsibilityEdit:new Set()},edited(){},toast(){}});
vm.runInContext(source.slice(source.indexOf('function syncResponsibilities('),source.indexOf('function renderResponsibilities('))+
  source.slice(source.indexOf('function toggleTag('),source.indexOf('async function saveDraft(')),context);
const run=code=>vm.runInContext(code,context);
const actual=()=>JSON.parse(run('JSON.stringify(S.current.label)'));
run("S.current={label:{defects:[],responsibilities:['待核'],image_state:'',note:'',problem_responsibilities:{}}}");
run("toggleTag('defect','残胶');toggleTag('defect','五金质量');toggleTag('defect','划伤')");
assert.deepEqual(actual().responsibilities,['生产','原材料']);
assert.deepEqual(actual().problem_responsibilities.划伤,[]);
run("toggleProblemResponsibility('划伤','运输');toggleTag('defect','残胶')");
assert.deepEqual(actual().responsibilities,['原材料','运输']);
assert.equal(Object.hasOwn(actual().problem_responsibilities,'残胶'),false);
run("toggleTag('defect','板件瑕疵');toggleProblemResponsibility('板件瑕疵','待核')");
assert.deepEqual(actual().responsibilities,['原材料','运输','待核']);
run("toggleProblemResponsibility('划伤','安装')");
assert.deepEqual(actual().problem_responsibilities.划伤,['运输','安装']);
assert.deepEqual(actual().problem_responsibilities.板件瑕疵,['待核']);
run("toggleTag('defect','五金质量');toggleTag('defect','划伤');toggleTag('defect','板件瑕疵')");
assert.deepEqual(actual().responsibilities,['待核']);
assert.deepEqual(actual().problem_responsibilities,{});
run("S.current={label:{defects:['崩边','划伤'],responsibilities:['运输'],image_state:'defect',note:''}};toggleTag('defect','残胶')");
assert.deepEqual(actual().problem_responsibilities,{崩边:['生产'],划伤:['运输'],残胶:['生产']});
console.log('Frontend responsibility selection checks passed.');
// Model suggestions are a separate preview. Only an explicit adoption appends
// problems; existing labels, drafts, notes, and image-state choices are protected.
vm.runInContext(source.slice(source.indexOf('function canRecommend('),source.indexOf('function renderRecommendations('))+
  source.slice(source.indexOf('function adoptRecommendations('),source.indexOf('async function requestRecommendations(')),context);
run("S.recommendationsEnabled=true;S.recommendation={candidates:[{name:'残胶'},{name:'划伤'}]};S.current={classified:false,has_draft:false,label:{defects:['崩边'],responsibilities:['安装'],image_state:'defect',note:'原有备注',problem_responsibilities:{崩边:['安装']}}}");
run("adoptRecommendations(['残胶','划伤','不存在的建议'])");
assert.deepEqual(actual().defects,['崩边','残胶','划伤']);
assert.deepEqual(actual().problem_responsibilities.崩边,['安装']);
assert.equal(actual().note,'原有备注');
assert.equal(run('S.current.recommendation_applied'),true);
run("S.current.classified=true");const protectedLabel=actual();run("adoptRecommendations(['残胶'])");assert.deepEqual(actual(),protectedLabel);
run("S.current.classified=false;S.current.has_draft=true;S.current.label.defects=[]");run("adoptRecommendations(['残胶'])");assert.deepEqual(actual().defects,[]);
run("S.current.has_draft=false;S.current.label.image_state='normal'");run("adoptRecommendations(['残胶'])");assert.deepEqual(actual().defects,[]);assert.equal(actual().image_state,'normal');
console.log('Frontend recommendation adoption/protection checks passed.');
run("S.recommendation={candidates:[{name:'崩边'}],image_state_suggestion:{state:'unrelated'}};S.current={classified:false,has_draft:false,label:{defects:[],responsibilities:['待核'],image_state:'',note:'保留备注',problem_responsibilities:{}}}");
run('adoptUnrelatedRecommendation()');
assert.equal(actual().image_state,'unrelated');assert.equal(actual().note,'保留备注');
assert.deepEqual(actual().defects,[]);assert.equal(run('S.current.recommendation_applied'),true);
run("S.current.label.image_state='defect';S.current.label.defects=['崩边']");const existingProblem=actual();
run('adoptUnrelatedRecommendation()');assert.deepEqual(actual(),existingProblem);
run("S.current.label.defects=[];S.current.label.image_state='unclear'");run('adoptUnrelatedRecommendation()');assert.equal(actual().image_state,'unclear');
run("S.current.label.image_state='';S.current.has_draft=true");run('adoptUnrelatedRecommendation()');assert.equal(actual().image_state,'');
run("S.current.has_draft=false;S.current.classified=true");run('adoptUnrelatedRecommendation()');assert.equal(actual().image_state,'');
console.log('Unrelated-state explicit adoption and no-overwrite checks passed.');
run("S.current={classified:false,has_draft:false,label:{defects:['划伤','板件瑕疵'],responsibilities:[],image_state:'defect',note:'不改备注',problem_responsibilities:{划伤:[],板件瑕疵:[]}}};S.responsibilityProblem='板件瑕疵'");
run("toggleTag('responsibility','运输')");assert.deepEqual(actual().problem_responsibilities.板件瑕疵,['运输']);assert.deepEqual(actual().problem_responsibilities.划伤,[]);
run("S.responsibilityProblem='划伤';toggleTag('responsibility','生产');toggleTag('responsibility','安装')");
assert.deepEqual(actual().problem_responsibilities.划伤,['生产','安装']);assert.equal(actual().note,'不改备注');
console.log('Compact active-problem responsibility checks passed.');
// Pagination limits visible suggestions without losing later problem candidates.
const elements={};
context.$=id=>elements[id]||(elements[id]={children:[],scrollTop:0,
  replaceChildren(){this.children=[];},append(child){this.children.push(child);}});
context.buttonTag=(name,active,action)=>({textContent:name,active,action});
vm.runInContext(source.slice(source.indexOf('function renderRecommendations('),source.indexOf('function adoptRecommendations(')),context);
run("S.current={classified:false,has_draft:false,label:{defects:[],image_state:''}};S.recommendation={status:'ready',candidates:Array.from({length:19},(_,i)=>({name:'问题'+i})),image_state_suggestion:{state:'unrelated'}}");
const rendered=[];
for(let page=0;page<5;page++){
  run('S.recommendationPage='+page+';renderRecommendations()');
  assert.equal(elements.recommendationTags.children.length,4);
  rendered.push(...elements.recommendationTags.children.map(b=>b.textContent));
  assert.equal(elements.recommendationPage.textContent,(page+1)+'/5');
}
assert.equal(new Set(rendered).size,20);
assert.equal(elements.recommendationNext.disabled,true);
run('S.current.classified=true;renderRecommendations()');
assert.equal(elements.recommendationTags.children.length,0);
assert.equal(elements.adoptRecommendations.hidden,true);
console.log('Recommendation pagination and protected-record rendering checks passed.');
(async()=>{
  const calls=[];
  const workflow=vm.createContext({S:{current:{id:1,label:{defects:['划伤']}},labelStep:1,index:0},
    renderTags(){},async saveDraft(){calls.push('draft');},
    async api(path){calls.push(path);return path.startsWith('/api/images')?{items:[],total:0}:{};},
    params(){return '';},async refreshState(){},async reload(){},toast(){}});
  vm.runInContext(source.slice(source.indexOf('async function commit('),source.indexOf('async function refreshState(')),workflow);
  await vm.runInContext('commit()',workflow);
  assert.deepEqual(calls,['draft']);assert.equal(vm.runInContext('S.labelStep',workflow),2);
  await vm.runInContext('commit()',workflow);
  assert.equal(calls.filter(p=>p==='/api/commit').length,1);
  console.log('Problem-first then responsibility confirmation checks passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
