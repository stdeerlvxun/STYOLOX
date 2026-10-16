const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('web/app.js','utf8'),html=fs.readFileSync('web/index.html','utf8'),css=fs.readFileSync('web/style.css','utf8');
class Element{
  constructor(){this.children=[];this.hidden=true;this.dataset={};this.value='';this.textContent='';this.attributes={};}
  append(child){this.children.push(child);}replaceChildren(){this.children=[];}setAttribute(name,value){this.attributes[name]=String(value);}focus(){}
}
const nodes={},el=id=>nodes[id]??=new Element();let requests=0,rejectNext=false,hiddenEditors=0;
const latest={key:'latest',model_id:'detector-new',method:'location',reviewed_images:300,distinct_cases:101,whole_image_unchanged:240,whole_image_no_change:200,fully_predicted_images:100,fully_predicted_unchanged_images:60,predicted_type_reviewed_boxes:250,predicted_type_correct_boxes:150,whole_image_rate_lower95:.76,case_rate_lower95:.63,fully_predicted_case_lower95:.43,reviewed_machine_boxes:400,unchanged_position_boxes:300,moved_machine_boxes:30,deleted_machine_boxes:70,human_added_boxes:20,problem_type_assigned:100,problem_type_changed:6,shortfalls:['整图原样通过率未达到 98%','需要独立盲测验证']};
const empty={key:'empty',model_id:'zero-sample',reviewed_images:0,distinct_cases:0,whole_image_unchanged:0,reviewed_machine_boxes:0,unchanged_position_boxes:0,whole_image_rate_lower95:null};
const data={groups:[latest,empty],current:latest,latest_generated_group_key:'new-unreviewed',auto_enabled:false};
const ctx=vm.createContext({$:el,S:{reviewQuality:null,reviewQualityLoading:false,reviewQualityError:'',reviewQualityModel:'',reviewQualityTab:'results'},
  Number,String,document:{createElement(){return new Element();}},setLabelPanel(){},setReviewArchive(){},
  AnnotationUI:{hideEditor(){hiddenEditors++;}},async api(path,body){assert.equal(path,'/api/review-quality/state');assert.equal(body,undefined,'Quality is read-only');requests++;if(rejectNext){rejectNext=false;throw new Error('network unavailable');}return data;}});
vm.runInContext(source.slice(source.indexOf('function reviewQualityRate('),source.indexOf('function renderReviewArchive(')),ctx);
const run=code=>vm.runInContext(code,ctx);
(async()=>{
  assert.equal(requests,0,'No quality aggregation is triggered by initialization');
  run('setReviewQuality(true)');await new Promise(resolve=>setImmediate(resolve));
  assert.equal(requests,1);assert.equal(el('reviewQualityPanel').hidden,false);assert.equal(hiddenEditors,1);
  assert.equal(el('reviewQualityModel').value,'latest','Opening defaults to the latest reviewed group');
  assert.equal(el('reviewQualityImageRate').textContent,'80.0%（240 / 300）');assert.equal(el('reviewQualityBoxRate').textContent,'75.0%（300 / 400）');
  assert.equal(el('reviewQualityFullRate').textContent,'60.0%（60 / 100）');assert.equal(el('reviewQualityTypeRate').textContent,'60.0%（150 / 250）','Inherited or human-assigned types do not inflate model-type accuracy');
  assert.match(el('reviewQualityCoverage').textContent,/300 张 · 覆盖 101 个售后单/);
  assert.equal(el('reviewQualityChanges').textContent,'移位 30 框 · 删除 70 框 · 人工补出 20 框');
  assert.equal(el('reviewQualityTypes').textContent,'整图无修改提交 200 / 300 张 · 人工赋类型 100 框 · 类型纠正 6 框');
  assert.match(el('reviewQualityStatus').textContent,/最新生成模型另组统计/);
  assert.match(el('reviewQualityConfidence').textContent,/完整预测按售后单统计.*43.0%/,'Automation evidence uses the complete prediction case bound, not position-only or correlated image confidence');
  assert.equal(el('reviewQualityShortfalls').children.length,2);assert.equal(el('reviewQualityResults').hidden,false);
  ctx.S.reviewQualityTab='basis';run('renderReviewQuality()');assert.equal(el('reviewQualityResults').hidden,true);assert.equal(el('reviewQualityBasis').hidden,false);
  assert.equal(el('reviewQualityBasisTab').attributes['aria-pressed'],'true');assert.equal(requests,1,'Tab changes use the loaded report without recomputing');
  latest.shortfalls=Array.from({length:12},(_,i)=>'未达到条件 '+i);run('renderReviewQuality()');assert.equal(el('reviewQualityShortfalls').children.length,5);assert.equal(el('reviewQualityBasisPage').textContent,'1 / 3');ctx.S.reviewQualityBasisPage=2;run('renderReviewQuality()');assert.equal(el('reviewQualityShortfalls').children.length,2);assert.equal(el('reviewQualityBasisNext').disabled,true);
  ctx.S.reviewQualityModel='empty';run('renderReviewQuality()');assert.match(el('reviewQualityImageRate').textContent,/暂无可计入样本/);assert.doesNotMatch(el('reviewQualityImageRate').textContent,/100/);assert.match(el('reviewQualityConfidence').textContent,/暂无样本/);
  assert.equal(requests,1,'Changing models is local and never mixes group denominators');
  run('setReviewQuality()');assert.equal(el('reviewQualityPanel').hidden,true);assert.equal(requests,1);
  run('setReviewQuality(true)');await new Promise(resolve=>setImmediate(resolve));assert.equal(requests,2);assert.equal(el('reviewQualityModel').value,'latest','A new open defaults back to latest reviewed evidence');
  rejectNext=true;await run('loadReviewQuality()');assert.equal(requests,3);assert.equal(el('reviewQualityRefresh').disabled,false);assert.match(el('reviewQualityStatus').textContent,/读取失败，请点刷新重试/);assert.equal(el('reviewQualityStatus').dataset.error,'true');
  await run('loadReviewQuality()');assert.equal(requests,4);assert.equal(el('reviewQualityStatus').dataset.error,'false');
  ctx.S.reviewQuality={groups:[],current:null};run('renderReviewQuality()');assert.equal(el('reviewQualityModel').disabled,true);assert.match(el('reviewQualityStatus').textContent,/暂无已提交/);
  ctx.S.reviewQuality={groups:[empty],current:null};ctx.S.reviewQualityModel='';run('renderReviewQuality()');assert.match(el('reviewQualityStatus').textContent,/暂无已提交/,'A zero-review generated group is not misrepresented as the latest reviewed model');
  for(const id of ['reviewQualityToggle','reviewQualityPanel','reviewQualityModel','reviewQualityRefresh','reviewQualityResultsTab','reviewQualityBasisTab','reviewQualityImageRate','reviewQualityBoxRate','reviewQualityChanges','reviewQualityShortfalls','reviewQualityContinue'])assert.match(html,new RegExp('id="'+id+'"'));
  assert.match(html,/人工审核通过率/);assert.match(html,/原样提交算通过，只看过不计/);assert.match(html,/位置、类型均未改/);assert.match(html,/人工给未知框选类型不算位置错误/);assert.match(html,/自动终判尚未开放/);
  assert.doesNotMatch(source.slice(source.indexOf('async function poll('),source.indexOf("on('openRoot'")),/loadReviewQuality|review-quality\/state|loadTrainingRounds|training-rounds\/state/,'Periodic poll must not run expensive quality or training aggregation');
  assert.equal((source.match(/\/api\/review-quality\/state/g)||[]).length,1);
  assert.match(source,/if\(e\.target\.closest\('#reviewQualityPanel'\)\)e\.stopImmediatePropagation\(\)/,'Panel keys cannot accidentally submit the current image');
  assert.match(css,/\.review-quality-panel\{[^}]*overflow:visible/,'The compact two-page surface does not introduce a scrollbar');
  console.log('Review quality UI: explicit read-only loading, model separation, exact denominators, zero evidence, correction counts, local tabs, error recovery and annotation keyboard guard passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
