const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('web/app.js','utf8'),html=fs.readFileSync('web/index.html','utf8');
class Element{
  constructor(){this.children=[];this.hidden=true;this.dataset={};this.value='';this.textContent='';}
  append(...values){this.children.push(...values);}replaceChildren(){this.children=[];}setAttribute(name,value){this[name]=value;}
}
const nodes={},el=id=>nodes[id]??=new Element(),clock=Date.parse('2026-10-16T12:00:00Z');
let annotationWork=false,annotationLoading=false,reloads=0,timers=0;
const state={counts:{total:100},scan:{running:false},annotation_proposal_counts:{total:20,pending:0},review_batches:{current:{batch_id:'batch-1',total:20,reviewed:20,pending:0},worker:{enabled:true,status:'generating',checked_at:new Date(clock).toISOString()}}};
const ctx=vm.createContext({$:el,S:{current:null,total:0,dirty:false,busy:false,navigating:false,loaded:true,index:0,reviewArchivePage:0,reviewBatches:state.review_batches},
  Date,URLSearchParams,Number,String,JSON,document:{createElement(){return new Element();}},setLabelPanel(){},
  AnnotationUI:{hasWork:()=>annotationWork,isLoading:()=>annotationLoading,hideEditor(){}},
  async refreshState(){return state;},async reload(){reloads++;},setTimeout(){timers++;},
  lastReviewQueueSignature:'',lastIndexed:100,scanWasRunning:false});
vm.runInContext(source.slice(source.indexOf('function reviewWorkerMessage('),source.indexOf('function renderStatusControls(')),ctx);
vm.runInContext(source.slice(source.indexOf('async function poll('),source.indexOf("on('openRoot'")),ctx);
const run=code=>vm.runInContext(code,ctx);
el('statusFilter').value='auto_boxes';
const fresh={enabled:true,checked_at:new Date(clock).toISOString()};
for(const status of ['disabled','checked','queue_sufficient','generating','refilled','registration_recovered','no_more_eligible_images','no_boxes_this_round','no_candidates_imported','waiting_for_gpu','waiting_for_worker','error']){
  ctx.worker={...fresh,status};const result=run(`reviewWorkerMessage(worker,${clock})`);
  assert.ok(result.text);assert.notEqual(result.text,'等待下次自动检查','Every published worker status has a useful explanation');
}
for(const status of ['refilled','registration_recovered']){
  ctx.worker={...fresh,status,pending:20};
  assert.equal(run(`reviewWorkerMessage(worker,${clock},0).text`),'本批已完成，等待后台补充','A consumed live queue must override the prior refill event and its stale pending count');
  assert.doesNotMatch(run(`reviewWorkerMessage(worker,${clock},4).text`),/已完成/,'A live pending queue must not be called finished');
}
ctx.worker={...fresh,status:'generating',checked_at:new Date(clock-600001).toISOString()};
assert.match(run(`reviewWorkerMessage(worker,${clock}).text`),/电脑未连接/,'Old heartbeat cannot imply active work');
ctx.worker={...fresh,status:'generating',checked_at:''};assert.match(run(`reviewWorkerMessage(worker,${clock}).text`),/电脑未连接/);
ctx.worker={enabled:false,status:'generating'};assert.match(run(`reviewWorkerMessage(worker,${clock}).text`),/暂停/);
assert.equal(run('canReloadReviewQueue()'),true);
for(const patch of [{current:{id:1}},{dirty:true},{busy:true},{navigating:true},{total:1}]){
  Object.assign(ctx.S,patch);assert.equal(run('canReloadReviewQueue()'),false);Object.assign(ctx.S,{current:null,total:0,dirty:false,busy:false,navigating:false});
}
annotationWork=true;assert.equal(run('canReloadReviewQueue()'),false);annotationWork=false;
annotationLoading=true;assert.equal(run('canReloadReviewQueue()'),false);annotationLoading=false;
el('statusFilter').value='all';assert.equal(run('canReloadReviewQueue()'),false);el('statusFilter').value='auto_boxes';
ctx.S.reviewBatches={active_batches:[{batch_id:'active',total:20,reviewed:8,pending:10,blocked:2}],recent_archives:Array.from({length:5},(_,i)=>({batch_id:i===0?'safe /?&编号':`arch-${i}`,total:20,archived_at:'2026-10-16T11:00:00Z'})),archives_count:8,worker:{enabled:false}};
run('renderReviewArchive()');assert.equal(el('reviewArchiveList').children.length,4);assert.equal(el('reviewArchivePage').textContent,'1 / 2');
assert.match(el('reviewArchiveList').children[0].children[1].textContent,/已审核 8 \/ 20 · 待审核 10 · 待处理 2/);
const download=el('reviewArchiveList').children[1].children[2];assert.equal(new URL(download.href,'http://localhost').searchParams.get('batch'),'safe /?&编号');
assert.equal(download.download,'');assert.match(el('reviewArchiveSummary').textContent,/展示最近 5 份/);
ctx.S.reviewArchivePage=1;run('renderReviewArchive()');assert.equal(el('reviewArchiveList').children.length,2);assert.equal(el('reviewArchiveNext').disabled,true);
ctx.S.reviewBatches={current:{total:20,pending:0,reviewed:18,blocked:2},worker:{enabled:false}};assert.match(run('reviewQueueMessage().title'),/待处理/,'Blocked records are not called completed');
ctx.S.proposalCounts={pending:4};assert.match(run('reviewQueueMessage().title'),/当前筛选/,'A filtered empty view is not an exhausted global queue');
for(const id of ['reviewBatchSummary','reviewBatchProgress','reviewBatchWorkerStatus','reviewBatchArchiveToggle','reviewArchivePanel','reviewArchiveClose','reviewArchiveList','reviewArchivePrevious','reviewArchiveNext'])assert.match(html,new RegExp('id="'+id+'"'));
(async()=>{
  ctx.S.current={id:9,label:{note:'正在审核'}};ctx.S.total=4;state.annotation_proposal_counts.pending=10;state.counts.total=101;
  await run('poll()');assert.equal(reloads,0,'New batches and index changes must not reload an open review');assert.equal(ctx.S.current.label.note,'正在审核');assert.equal(ctx.S.reviewQueueReloadPending,true);
  ctx.S.current=null;ctx.S.total=0;ctx.S.dirty=true;await run('poll()');assert.equal(reloads,0,'Unsaved work blocks queue refresh');
  ctx.S.dirty=false;await run('poll()');assert.equal(reloads,1,'Empty idle review receives new batch automatically');assert.equal(ctx.S.reviewQueueReloadPending,false);
  await run('poll()');assert.equal(reloads,1,'Unchanged queue is not reloaded each poll');
  assert.ok(timers>=4);
  console.log('Review batches: worker freshness, exact statuses, four-row archive pagination, encoded downloads and edit-safe automatic refill passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
