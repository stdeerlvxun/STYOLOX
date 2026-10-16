'use strict';
const $ = id => document.getElementById(id);
const clientId = () => typeof crypto.randomUUID==='function' ? crypto.randomUUID() :
  Array.from(crypto.getRandomValues(new Uint8Array(16)),b=>b.toString(16).padStart(2,'0')).join('');
const S = {token:'',root:null,tags:{defect:[],responsibility:[]},rules:{},responsibilityEdit:new Set(),counts:{},filters:{defect:new Set(),responsibility:new Set()},
  pages:new Map(),pagePending:new Map(),total:0,index:0,current:null,selected:new Set(),view:0,focus:0,client:clientId(),
  dirty:false,revision:0,saveTail:Promise.resolve(),saveTimer:null,busy:false,navigating:false,nav:0,blobs:new Map(),pending:new Map(),
  controllers:new Map(),thumbJobs:[],thumbActive:0,zoom:1,rotation:0,panX:0,panY:0,original:false,importKey:null,loaded:false,
  recommendationsEnabled:false,recommendation:null,recommendationTimer:null,recommendationPage:0,responsibilityProblem:'',labelStep:1,reviewEnabled:false,links:{}};
let toastTimer, searchTimer, renderFrame, scanWasRunning=false, lastIndexed=0;
async function api(path,body){
  const options = body === undefined ? {} : {method:'POST',headers:{'Content-Type':'application/json','X-Local-Token':S.token},body:JSON.stringify(body)};
  const response=await fetch(path,options), data=await response.json();
  if(!response.ok) throw new Error(data.error || '操作失败，请重试');
  return data;
}
function toast(message,error=false){$('toast').textContent=message;$('toast').classList.toggle('error',error);$('toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('toast').hidden=true,error?7000:3500);}
function fail(error){if(error.name!=='AbortError')toast(error.message || String(error),true);}
function on(id,event,action){$(id).addEventListener(event,e=>{Promise.resolve(action(e)).catch(fail);});}
function renderStatusControls(){
  const status=$('statusFilter').value,buttons=$('commonStatusFilters').querySelectorAll('button[data-status]');
  let common=false;
  for(const button of buttons){const active=button.dataset.status===status;button.setAttribute('aria-pressed',String(active));common||=active;}
  const option=$('statusFilter').querySelector('option[value="'+status+'"]');
  $('statusMoreLabel').textContent=common?'更多筛选':'更多：'+(option?.textContent||'图片状态');
  $('statusMoreLabel').title=$('statusMoreLabel').textContent;
  $('autoBoxCount').textContent=String(S.proposalCounts?.pending||0);
  $('reviewKind').hidden=!status.startsWith('error_review');
}
function changeStatusFilter(status){
  if(!$('statusFilter').querySelector('option[value="'+status+'"]'))return;
  $('statusFilter').value=status;$('statusDetails').open=false;
  renderStatusControls();S.selected.clear();renderSelection();return reload();
}
function params(extra={}){
  const q=new URLSearchParams({status:$('statusFilter').value,q:$('search').value.trim(),...extra});
  if($('statusFilter').value.startsWith('error_review')&&$('reviewKind').value)q.set('review_kind',$('reviewKind').value);
  for(const group of ['defect','responsibility'])for(const tag of S.filters[group])q.append(group,tag);
  return q;
}
function imageURL(item,kind,prefetch=false){
  return '/api/image?'+new URLSearchParams({id:item.id,kind,c:S.client,v:S.view,f:S.focus,...(prefetch?{prefetch:'1'}:{})});
}
function cacheKey(item,kind){return item.id+':'+item.mtime_ns+':'+item.size+':'+kind;}
function remember(key,url){
  S.blobs.set(key,url);
  while(S.blobs.size>120){const oldest=S.blobs.keys().next().value;if(S.current&&oldest.startsWith(S.current.id+':')){const value=S.blobs.get(oldest);S.blobs.delete(oldest);S.blobs.set(oldest,value);continue;}URL.revokeObjectURL(S.blobs.get(oldest));S.blobs.delete(oldest);}
}
async function blob(item,kind,prefetch=false){
  const key=cacheKey(item,kind);
  if(S.blobs.has(key))return S.blobs.get(key);
  if(S.pending.has(key))return S.pending.get(key);
  const controller=new AbortController();S.controllers.set(key,{controller,kind});
  const promise=(async()=>{
    const response=await fetch(imageURL(item,kind,prefetch),{signal:controller.signal});
    if(!response.ok){const error=await response.json();if(response.status===409)throw new DOMException('已取消','AbortError');throw new Error(error.error || '图片无法读取');}
    const url=URL.createObjectURL(await response.blob());remember(key,url);return url;
  })();
  S.pending.set(key,promise);
  try{return await promise;}finally{if(S.pending.get(key)===promise){S.pending.delete(key);S.controllers.delete(key);}}
}
function cancelImages(all=false){
  for(const [key,value]of S.controllers){if(all||value.kind!=='thumb'){value.controller.abort();S.pending.delete(key);S.controllers.delete(key);}}
  if(all)S.thumbJobs=[];
}
async function updateView(){await api('/api/view',{client:S.client,view:S.view,focus:S.focus});}
function columns(){const width=$('thumbViewport').clientWidth;return width<210?1:2;}
const ROW=144;
function meta(index){const page=S.pages.get(Math.floor(index/40));return page&&page[index%40];}
async function getPage(page){
  if(S.pages.has(page))return S.pages.get(page);
  if(S.pagePending.has(page))return S.pagePending.get(page);
  const view=S.view;
  const task=(async()=>{
    const result=await api('/api/images?'+params({offset:page*40,limit:40}));
    if(view!==S.view)return [];
    S.pages.set(page,result.items);S.total=result.total;
    while(S.pages.size>8){const oldest=S.pages.keys().next().value;if(oldest===page)break;S.pages.delete(oldest);}
    return result.items;
  })();
  S.pagePending.set(page,task);
  try{return await task;}finally{if(S.pagePending.get(page)===task)S.pagePending.delete(page);}
}
function queueThumb(item){
  const key=cacheKey(item,'thumb');
  if(S.blobs.has(key)){attachThumb(item,S.blobs.get(key));return;}
  if(S.thumbJobs.some(job=>job.key===key)||S.pending.has(key))return;
  S.thumbJobs.push({item,key,view:S.view});pumpThumbs();
}
function attachThumb(item,url){
  const element=$('thumbSpace').querySelector('img[data-image="'+item.id+'"]');
  if(element&&element.src!==url)element.src=url;
}
function pumpThumbs(){
  while(S.thumbActive<2&&S.thumbJobs.length){
    const job=S.thumbJobs.shift();if(job.view!==S.view)continue;S.thumbActive++;
    blob(job.item,'thumb').then(url=>{if(job.view===S.view)attachThumb(job.item,url);}).catch(error=>{
      if(error.name!=='AbortError'){const img=$('thumbSpace').querySelector('img[data-image="'+job.item.id+'"]');if(img)img.alt='暂时无法读取';}
    }).finally(()=>{S.thumbActive--;pumpThumbs();});
  }
}
function renderThumbs(){
  const viewport=$('thumbViewport'),space=$('thumbSpace'),cols=columns();
  space.style.height=(Math.ceil(S.total/cols)*ROW+8)+'px';
  const first=Math.max(0,Math.floor(viewport.scrollTop/ROW)-2)*cols;
  const end=Math.min(S.total,(Math.ceil((viewport.scrollTop+viewport.clientHeight)/ROW)+2)*cols);
  space.replaceChildren();
  const needPages=new Set();
  for(let i=first;i<end;i++){
    const item=meta(i),card=document.createElement('div');card.className='thumb'+(!item?' placeholder':'');
    card.style.top=(Math.floor(i/cols)*ROW+8)+'px';card.style.left=cols===1?'8px':(i%2===0?'8px':'calc(50% + 2px)');
    if(!item){needPages.add(Math.floor(i/40));space.append(card);continue;}
    card.classList.toggle('active',S.current?.id===item.id);card.classList.toggle('selected',S.selected.has(item.id));
    card.tabIndex=0;card.setAttribute('role','button');card.setAttribute('aria-label','查看第'+(i+1)+'张 '+item.file_name);
    card.addEventListener('click',()=>selectIndex(i).catch(fail));
    card.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();e.stopPropagation();selectIndex(i).catch(fail);}});
    const image=document.createElement('img');image.alt='';image.dataset.image=item.id;image.loading='lazy';card.append(image);
    const check=document.createElement('input');check.type='checkbox';check.className='select-box';check.checked=S.selected.has(item.id);check.setAttribute('aria-label','批量选择第'+(i+1)+'张');
    check.addEventListener('click',e=>e.stopPropagation());check.addEventListener('change',()=>{check.checked?S.selected.add(item.id):S.selected.delete(item.id);renderThumbs();renderSelection();});card.append(check);
    if(item.label){const tag=document.createElement('span');tag.className='classified';tag.textContent='已分类';card.append(tag);}
    const caption=document.createElement('span');caption.className='caption';caption.textContent=(item.reported_at?.slice(0,10)||'日期待补')+' · '+item.file_name;caption.title=item.relative_path;card.append(caption);
    space.append(card);queueThumb(item);
  }
  for(const page of needPages)getPage(page).then(()=>{if(S.pages.has(page))scheduleRender();}).catch(fail);
}
function scheduleRender(){if(!renderFrame)renderFrame=requestAnimationFrame(()=>{renderFrame=null;renderThumbs();});}
function renderSelection(){$('selectionCount').textContent=S.selected.size?'已选 '+S.selected.size+' 张':'未选中图片';$('batch').disabled=!S.selected.size;$('batchPanel').hidden=true;$('confirmReviewSelected').disabled=!S.selected.size;}
function renderWorkflow(){
  document.querySelector('.label-panel').dataset.step=String(S.labelStep||1);
  $('workflowTitle').textContent=S.labelStep===2?'2 · 确认责任':'1 · 先找问题点';
  $('backToProblems').hidden=S.labelStep!==2;
  $('commit').textContent=S.labelStep!==2&&S.current?.label.defects.length?'下一步 · 确认责任 ↵':$('statusFilter').value==='error_review'?'保存分类并完成复核 ↵':'确认保存并下一张 ↵';
  $('reviewActions').hidden=!S.reviewEnabled||$('statusFilter').value!=='model_unrelated';
  $('rejectUnrelated').hidden=!S.current?.model_unrelated||S.current?.classified||S.current?.has_draft;
  $('rejectUnrelated').disabled=S.busy||S.navigating||S.dirty;
  if($('workspaceCommit'))$('workspaceCommit').disabled=!S.current||S.busy||S.navigating;
  if($('workspacePrevious'))$('workspacePrevious').disabled=!S.current||S.index<=0||S.busy||S.navigating;
  for(const id of ['imageStateToggle','labelPanelToggle'])if($(id))$(id).disabled=!S.current||S.busy||S.navigating;
}
function setLabelPanel(view=null){
  const panel=$('labelPanel');if(!panel)return;
  panel.hidden=!view;
  if(view){panel.dataset.panelView=view;$('labelPanelTitle').textContent=view==='state'?'图片状态与备注':'图片信息 / 分类';globalThis.AnnotationUI?.hideEditor?.();}
  $('imageStateToggle').setAttribute('aria-expanded',String(view==='state'));
  $('labelPanelToggle').setAttribute('aria-expanded',String(view==='details'));
}
async function submitWorkspace(){
  if(!S.current||S.busy||S.navigating)return;
  if(globalThis.AnnotationUI?.isLoading?.())throw new Error('正在读取当前图片的补标，请稍后再保存');
  const label=S.current.label;
  if(['','defect'].includes(label.image_state)&&globalThis.AnnotationUI?.hasWork()){
    await globalThis.AnnotationUI.submitImage();return;
  }
  if(!label.defects.length&&!label.image_state){setLabelPanel('state');throw new Error('请先框选问题位置，或选择图片状态');}
  const pending=label.problem_responsibilities?label.defects.filter(problem=>!label.problem_responsibilities[problem]?.length):[];
  if(pending.length){S.labelStep=2;S.responsibilityProblem=pending[0];setLabelPanel('details');renderTags();throw new Error('请先为“'+pending[0]+'”选择责任，拿不准可选待核');}
  S.labelStep=2;await commit();
}
function buttonTag(name,active,action){
  const button=document.createElement('button');button.type='button';button.textContent=name;button.setAttribute('aria-pressed',String(active));button.addEventListener('click',action);return button;
}
function syncResponsibilities(label){
  const mapping=label.problem_responsibilities;
  for(const problem of Object.keys(mapping))if(!label.defects.includes(problem))delete mapping[problem];
  for(const problem of label.defects)if(!Object.hasOwn(mapping,problem))mapping[problem]=S.rules[problem]?[S.rules[problem]]:[];
  label.responsibilities=label.defects.length?[...new Set(Object.values(mapping).flat())]:['待核'];
}
function toggleProblemResponsibility(problem,name){
  if(!S.current||S.busy||S.navigating)return;
  const mapping=S.current.label.problem_responsibilities,value=mapping[problem],index=value.indexOf(name);
  mapping[problem]=index>=0?value.filter(v=>v!==name):name==='待核'?['待核']:[...value.filter(v=>v!=='待核'),name];
  syncResponsibilities(S.current.label);edited();
}
function renderResponsibilities(locked){
  const target=$('responsibilityTags'),label=S.current?.label,mapping=label?.problem_responsibilities;
  const problems=mapping?label.defects:[],pending=problems.filter(p=>!mapping[p]?.length),select=$('responsibilityProblem');
  if(!problems.includes(S.responsibilityProblem))S.responsibilityProblem=pending[0]||problems[0]||'';
  select.replaceChildren();
  for(const problem of problems){const option=document.createElement('option');option.value=problem;option.textContent=problem+' · '+(mapping[problem]?.join('、')||'待选责任');select.append(option);}
  if(!problems.length){const option=document.createElement('option');option.value='';option.textContent='先选择问题点';select.append(option);}
  select.value=S.responsibilityProblem;select.disabled=locked||!problems.length;
  $('nextResponsibility').disabled=locked||!pending.length;
  $('restoreDefaultResponsibility').disabled=locked||!S.rules[S.responsibilityProblem]||!problems.length;
  const hint=!label?'选择问题点后显示责任。':!mapping?'历史责任保留；编辑问题点后应用规则。':pending.length?'还有 '+pending.length+' 项责任待选；支持多选。':'责任已选齐，可直接保存。';
  $('responsibilityHint').textContent=hint;$('responsibilityHint').title=hint;
  const values=problems.length?(mapping[S.responsibilityProblem]||[]):(label?.responsibilities||[]);target.replaceChildren();
  const advice=S.links?.[S.responsibilityProblem],role=advice?.suggestion;
  const association=advice?.resolved_cases?'（'+(advice.responsibility_case_counts[role]||0)+'/'+advice.resolved_cases+'单关联）':'';
  $('responsibilityAdvice').textContent=role?'建议 '+role+' · '+(advice.basis==='user_experience_prior'?'经验优先':advice.basis==='approved_rule'?'已确认规则':'人工关联')+association:'暂无可靠关联，请人工选择';
  $('responsibilityAdvice').title=$('responsibilityAdvice').textContent;
  $('applyResponsibilityAdvice').disabled=locked||!role||!mapping||!problems.length||values.length>0;
  for(const name of S.tags.responsibility){
    const button=buttonTag(name,values.includes(name),()=>problems.length?toggleProblemResponsibility(S.responsibilityProblem,name):toggleTag('responsibility',name));
    button.disabled=locked;target.append(button);
  }
}
function renderTags(){
  const panel=$('defectTags').closest('.label-scroll'),scrollTop=panel.scrollTop;
  const locked=!S.current||S.busy||S.navigating;
  for(const group of ['defect','responsibility']){
    const field=group==='defect'?'defects':'responsibilities',target=$(group+'Tags'),filters=$(group+'Filters');target.replaceChildren();filters.replaceChildren();
    for(const [i,name]of S.tags[group].entries()){
      const button=buttonTag(name,!!S.current?.label[field].includes(name),()=>toggleTag(group,name));button.disabled=locked;
      button.title=name+(i<5?'（'+(group==='responsibility'?'Shift + ':'')+(i+1)+'）':'');if(group==='defect')target.append(button);
      filters.append(buttonTag(name,S.filters[group].has(name),()=>{S.filters[group].has(name)?S.filters[group].delete(name):S.filters[group].add(name);reload().catch(fail);}));
    }
  }
  renderResponsibilities(locked);
  for(const button of $('imageStates').children){button.disabled=locked;button.setAttribute('aria-pressed',String(S.current?.label.image_state===button.dataset.state));}
  $('note').disabled=locked;$('commit').disabled=locked;
  $('filterNumber').textContent=S.filters.defect.size+S.filters.responsibility.size?'('+ (S.filters.defect.size+S.filters.responsibility.size)+')':'';
  renderRecommendations();
  renderWorkflow();
  globalThis.AnnotationUI?.refresh();
  panel.scrollTop=scrollTop;
}
function canRecommend(){return !!(S.recommendationsEnabled&&S.current&&!S.current.classified&&!S.current.has_draft);}
function renderRecommendations(){
  const panel=$('recommendationPanel');panel.hidden=!S.recommendationsEnabled;if(panel.hidden)return;
  const eligible=canRecommend(),result=eligible?(S.recommendation||{status:'queued'}):
    {status:!S.current?'empty':S.current.classified?'classified':'draft'};
  const target=$('recommendationTags'),scrollTop=target.scrollTop;target.replaceChildren();
  const locked=!eligible||S.busy||S.navigating||!['','defect'].includes(S.current?.label.image_state);
  const status=result.status;
  $('recommendationStatus').textContent=status==='ready'?(result.image_state_suggestion?'疑似无关，仅供复核；问题点建议仍保留。':result.candidates.length?'仅供看图复核，可逐项采用或手动补充。':'没有达到推荐标准的问题点，请继续人工看图；不代表无问题。'):
    status==='empty'?'选择未分类图片后显示问题点建议。':
    status==='classified'?'已有人工分类，保留原结果，不生成建议。':
    status==='draft'?'已有人工暂存，保留原编辑，不生成建议。':
    status==='error'?'建议暂时无法生成，可重试或继续手动分类。':
    status==='protected'?'已存在人工编辑，继续使用当前标签。':
    status==='manual'?'已开始人工编辑，可继续手动选择问题点。':
    status==='disabled'?'模型建议暂不可用，可继续手动分类。':
    status==='unavailable'?'原图暂不可用，可恢复连接后重试。':'正在后台生成建议，你可以先看图或手动分类…';
  $('recommendationStatus').title=$('recommendationStatus').textContent;
  const items=[...(result.image_state_suggestion?.state==='unrelated'?[{state:true}]:[]),...(result.candidates||[])];
  const pageCount=Math.max(1,Math.ceil(items.length/4));S.recommendationPage=Math.min(S.recommendationPage||0,pageCount-1);
  $('recommendationPage').textContent=(S.recommendationPage+1)+'/'+pageCount;
  $('recommendationPrevious').disabled=S.busy||S.navigating||S.recommendationPage===0;
  $('recommendationNext').disabled=S.busy||S.navigating||S.recommendationPage>=pageCount-1;
  for(const candidate of items.slice(S.recommendationPage*4,S.recommendationPage*4+4)){
  if(candidate.state){
    const selected=S.current.label.image_state==='unrelated';
    const button=buttonTag('疑似无关图片',selected,adoptUnrelatedRecommendation);
    button.className='state-suggestion';
    button.title='采用“无关图片”状态，仍需按 Enter 确认';button.disabled=!eligible||S.busy||S.navigating||selected;target.append(button);
  }else{
    const selected=S.current.label.defects.includes(candidate.name);
    const button=buttonTag(candidate.name,selected,()=>adoptRecommendations([candidate.name]));
    button.title=candidate.name+(selected?'（已加入）':'');button.disabled=locked||selected;target.append(button);
  }
  }
  target.scrollTop=scrollTop;
  $('adoptRecommendations').hidden=status!=='ready'||!result.candidates.length;
  $('adoptRecommendations').disabled=locked||result.candidates?.every(c=>S.current.label.defects.includes(c.name));
  $('retryRecommendations').hidden=!['error','disabled','unavailable','manual'].includes(status);
  $('retryRecommendations').disabled=S.busy||S.navigating;
}
function adoptRecommendations(names){
  if(!canRecommend()||S.busy||S.navigating)return;
  const label=S.current.label;
  if(!['','defect'].includes(label.image_state)){toast('已选择图片状态，保留当前选择；如需采用建议，请先手动选择问题点。');return;}
  const allowed=new Set((S.recommendation?.candidates||[]).map(c=>c.name));
  const additions=names.filter(name=>allowed.has(name)&&!label.defects.includes(name));
  if(!additions.length)return;
  if(!label.problem_responsibilities)label.problem_responsibilities=Object.fromEntries(label.defects.map(p=>[p,[...label.responsibilities]]));
  label.defects.push(...additions);label.image_state='defect';syncResponsibilities(label);
  S.responsibilityProblem=additions.find(p=>!S.rules[p])||additions[0];
  S.current.recommendation_applied=true;edited();
}
function adoptUnrelatedRecommendation(){
  if(!canRecommend()||S.busy||S.navigating||S.recommendation?.image_state_suggestion?.state!=='unrelated')return;
  const label=S.current.label;
  if(label.defects.length||!['','unrelated'].includes(label.image_state)){
    toast('当前已有问题点或其他状态，已保留；请先人工核对并清除原选择。');return;
  }
  label.image_state='unrelated';label.problem_responsibilities={};label.responsibilities=['待核'];
  S.current.recommendation_applied=true;edited();
}
async function requestRecommendations(item,nav,retry=false,attempt=0){
  clearTimeout(S.recommendationTimer);
  if(!canRecommend()||nav!==S.nav||item.id!==S.current.id)return;
  if(S.dirty&&!retry){S.recommendation={status:'manual'};renderRecommendations();return;}
  try{
    const result=await api('/api/recommendation?'+new URLSearchParams({id:item.id,c:S.client,v:S.view,f:S.focus,...(retry?{retry:'1'}:{})}));
    if(nav!==S.nav||item.id!==S.current?.id)return;
    S.recommendation=result;renderRecommendations();
    if(['queued','running','cancelled'].includes(result.status)&&attempt<40){
      S.recommendationTimer=setTimeout(()=>requestRecommendations(item,nav,false,attempt+1),1500);
    }
  }catch(error){if(nav===S.nav){S.recommendation={status:'error'};renderRecommendations();}}
}
function saveIndicator(text,type=''){$('saveStatus').textContent=text;$('saveStatus').className=type;}
function edited(){
  if(!S.current||S.busy||S.navigating)return;S.dirty=true;S.revision++;saveIndicator('待暂存','unsaved');renderTags();clearTimeout(S.saveTimer);S.saveTimer=setTimeout(()=>saveDraft().catch(fail),350);
}
function toggleTag(group,name){
  if(!S.current||S.busy||S.navigating)return;
  const label=S.current.label;
  if(group==='responsibility'&&label.problem_responsibilities&&label.defects.length){
    const problem=(label.defects.includes(S.responsibilityProblem)?S.responsibilityProblem:'')||label.defects.find(p=>!label.problem_responsibilities[p]?.length)||label.defects.find(p=>!S.rules[p])||label.defects.find(p=>S.responsibilityEdit.has(p));
    if(problem)return toggleProblemResponsibility(problem,name);
    toast('当前已自动定责，点击对应问题点的“调整”可修改责任');return;
  }
  if(group==='defect'&&!label.problem_responsibilities){
    label.problem_responsibilities=Object.fromEntries(label.defects.map(p=>[p,S.rules[p]?[S.rules[p]]:[...label.responsibilities]]));
  }
  const field=group==='defect'?'defects':'responsibilities',value=S.current.label[field],index=value.indexOf(name);
  index<0?value.push(name):value.splice(index,1);
  if(group==='defect'&&index<0)S.responsibilityProblem=name;
  if(group==='defect'){label.image_state=value.length?'defect':'';syncResponsibilities(label);}
  else if(name!=='待核'&&index<0){S.current.label.responsibilities=value.filter(v=>v!=='待核');}
  else if(name==='待核'&&index<0){S.current.label.responsibilities=['待核'];}
  edited();
}
async function saveDraft(){
  await globalThis.AnnotationUI?.saveDraft();
  clearTimeout(S.saveTimer);
  if(!S.current||!S.dirty){await S.saveTail;return;}
  const id=S.current.id,revision=S.revision,label=structuredClone(S.current.label);
  saveIndicator('正在暂存…');
  const record=S.current,protectedSave=!!record.recommendation_applied;
  const task=S.saveTail.catch(()=>{}).then(async()=>{
    const result=await api('/api/draft',{id,label,only_unclassified:protectedSave,expected_draft:record.recommendation_draft_fingerprint||null});
    record.recommendation_draft_fingerprint=result.draft_fingerprint;
    record.draft_fingerprint=result.draft_fingerprint;record.has_draft=true;return result;
  });S.saveTail=task;
  try{await task;if(S.current?.id===id&&S.revision===revision){S.dirty=false;saveIndicator('已暂存','saved');}}
  catch(error){if(S.current?.id===id){S.dirty=true;saveIndicator('暂存失败，请重试','unsaved');}throw error;}
}
async function reload({target=null,index=0,keep=false}={}){
  if(!keep)await saveDraft();
  const view=++S.view;cancelImages(true);S.pagePending.clear();await updateView();
  const result=await api('/api/images?'+params({offset:Math.floor(index/40)*40,limit:40,...(target?{locate:target}:{})}));
  if(view!==S.view)return;
  S.pages.clear();S.pages.set(Math.floor(result.offset/40),result.items);S.total=result.total;
  const previousIndex=S.index;
  S.index=result.position>=0&&result.position!==null?result.position:Math.min(index,Math.max(0,S.total-1));
  $('resultCount').textContent=S.total+' 张';renderTags();
  if(!keep||!S.current){
    if(S.total)await selectIndex(S.index,{internal:true});
    else empty();
  }
  if(keep&&S.current)$('position').textContent=(S.index+1)+' / '+S.total;
  if(!keep||S.index!==previousIndex)$('thumbViewport').scrollTop=Math.floor(S.index/columns())*ROW;
  renderThumbs();
}
function empty(){
  S.current=null;S.dirty=false;$('photo').hidden=true;$('imageBusy').hidden=true;$('retryImage').hidden=true;
  globalThis.AnnotationUI?.empty();
  $('imageMessage').hidden=false;$('imageMessage').replaceChildren();
  const strong=document.createElement('strong'),span=document.createElement('span');
  const reviewed=$('statusFilter').value==='error_review'&&S.errorReviewCounts?.total>0&&S.errorReviewCounts.pending===0;
  const boxesReviewed=$('statusFilter').value==='auto_boxes'&&S.proposalCounts?.total>0&&S.proposalCounts.pending===0;
  strong.textContent=boxesReviewed?'本批自动框审核已完成':reviewed?'本轮错例复核已完成，结果已保存':scanWasRunning?'正在索引图片':'当前筛选没有图片';
  span.textContent=boxesReviewed?'已确认的位置补标已保存；可继续补框或分类，也可点“已补框”查看。':reviewed?'在“更多筛选”中选择“本轮错例已复核”查看；位置框需另行框选保存。':scanWasRunning?'找到图片后即可开始分类。':'可在“更多筛选”查看全部图片，或清除搜索和标签筛选。';
  $('imageMessage').append(strong,span);
  if($('statusFilter').value==='auto_boxes'&&!scanWasRunning){
    const actions=document.createElement('div');actions.className='empty-actions';
    for(const [status,label]of [['annotation_pending','继续补框'],['unclassified','开始分类']]){
      const button=document.createElement('button');button.textContent=label;
      button.addEventListener('click',()=>changeStatusFilter(status)?.catch(fail));actions.append(button);
    }
    $('imageMessage').append(actions);
  }
  $('position').textContent='没有可显示的图片';$('fileName').textContent='原图不移动、不复制';$('commit').disabled=true;$('locate').disabled=true;$('note').value='';saveIndicator('未选择图片');renderTags();
}
async function selectIndex(index,{internal=false}={}){
  if(index<0||index>=S.total||(S.busy&&!internal))return;
  const nav=++S.nav,view=S.view;
  clearTimeout(S.recommendationTimer);
  S.navigating=true;renderTags();
  try {
  await saveDraft();
  await getPage(Math.floor(index/40));if(nav!==S.nav||view!==S.view)return;
  const item=meta(index);if(!item)return;
  S.focus++;cancelImages();await updateView();
  const detail=await api('/api/item?id='+item.id);if(nav!==S.nav||view!==S.view)return;
  S.current=detail;S.recommendation=null;S.recommendationPage=0;S.responsibilityProblem='';S.labelStep=1;S.responsibilityEdit.clear();S.index=index;S.dirty=false;S.revision++;S.original=false;
  document.body.classList.remove('library-open');
  setLabelPanel();
  $('note').value=detail.label.note;saveIndicator(detail.has_draft?'已恢复暂存':detail.classified?'已分类':'尚未分类',detail.has_draft||detail.classified?'saved':'');
  $('position').textContent=(index+1)+' / '+S.total;$('fileName').textContent=detail.relative_path;$('fileName').title=detail.full_path;
  $('commit').disabled=false;$('locate').disabled=false;$('contextDetails').open=false;
  $('caseContext').textContent='展开后按需读取售后单信息。';resetTransform();renderTags();renderThumbs();
  const row=Math.floor(index/columns()),top=row*ROW,viewport=$('thumbViewport');
  if(top<viewport.scrollTop||top+ROW>viewport.scrollTop+viewport.clientHeight)viewport.scrollTop=Math.max(0,top-ROW);
  showPhoto(detail,false,nav).catch(fail);
  globalThis.AnnotationUI?.loadForImage(detail).catch(fail);
  requestRecommendations(detail,nav).catch(()=>{});
  } finally {
    if(nav===S.nav){S.navigating=false;renderTags();}
  }
}
async function showPhoto(item,original=false,nav=S.nav){
  $('imageBusy').hidden=false;$('retryImage').hidden=true;if(!original)$('photo').hidden=true;
  $('imageMessage').hidden=true;const start=performance.now();
  try{
    const url=await blob(item,original?'original':'preview');if(nav!==S.nav)return;
    $('photo').src=url;$('photo').alt='售后附件 '+item.file_name;$('photo').hidden=false;$('imageBusy').hidden=true;
    if(original){S.original=true;$('loadHint').textContent='已加载原始清晰度';}
    else if(performance.now()-start<1500){$('loadHint').textContent='后台预读后续 2 张';prefetch(nav).catch(()=>{});}
    else $('loadHint').textContent='读取较慢，暂缓后台预读';
  }catch(error){
    if(error.name==='AbortError'||nav!==S.nav)return;
    $('photo').hidden=true;$('imageBusy').hidden=true;$('imageMessage').hidden=false;$('imageMessage').textContent='图片暂时无法读取：'+error.message+'。已有标签和进度已保留。';$('retryImage').hidden=false;
  }
}
async function prefetch(nav){
  for(let step=1;step<=2;step++){
    if(nav!==S.nav)return;const index=S.index+step;if(index>=S.total)return;
    await getPage(Math.floor(index/40));if(nav!==S.nav)return;
    const item=meta(index);if(item)await blob(item,'preview',true);
  }
}
function resetTransform(){S.zoom=1;S.rotation=0;S.panX=0;S.panY=0;transform();}
function transform(){$('photo').style.transform='translate('+S.panX+'px,'+S.panY+'px) scale('+S.zoom+') rotate('+S.rotation+'deg)';globalThis.AnnotationUI?.renderOverlay();}
async function commit(){
  if(!S.current||S.busy||S.navigating)return;
  if(S.labelStep!==2&&S.current.label.defects.length){await saveDraft();S.labelStep=2;renderTags();return;}
  S.busy=true;renderTags();
  try{
    await saveDraft();const id=S.current.id,index=S.index;
    const reviewRun=$('statusFilter').value==='error_review'?S.current.error_review?.run_id:null;
    const saved=await api('/api/commit',{id,label:S.current.label,only_unclassified:!!S.current.recommendation_applied,expected_draft:S.current.recommendation_draft_fingerprint||null,review_run:reviewRun});S.dirty=false;
    const result=await api('/api/images?'+params({offset:index,limit:1}));let next=index;
    if(result.items[0]?.id===id)next=index+1;
    await refreshState();await reload({index:Math.min(next,Math.max(0,result.total-1))});toast(saved.review_completed?'分类已保存，错例复核已完成；位置框另行补标':'已确认并保存标签');
  }finally{S.busy=false;renderTags();}
}
async function refreshState(){
  const state=await api('/api/state');S.token=state.token;S.root=state.root;S.counts=state.counts;
  S.rules=state.responsibility_rules||{};
  S.links=state.responsibility_links||{};
  S.recommendationsEnabled=!!state.recommendations?.enabled;
  S.reviewEnabled=!!state.model_review?.enabled;
  S.errorReviewCounts=state.error_review_counts;
  const tagsChanged=JSON.stringify(S.tags)!==JSON.stringify(state.tags);S.tags=state.tags;
  if(document.activeElement!==$('rootPath'))$('rootPath').value=state.root.display_path||state.root.path;
  $('sourceNote').textContent=state.server_mode?'原图在 NAS · 标签保存在 101':'直接查看原图 · 标签保存在本机';
  $('locate').textContent=state.server_mode?'复制原文件路径':'定位原文件';
  if(!$('manifestPath').value)$('manifestPath').value=state.default_manifest;
  $('counts').textContent='已索引 '+state.counts.total+' · 已分类 '+state.counts.classified+' · 未分类 '+(state.counts.total-state.counts.classified);
  if(S.reviewEnabled)$('counts').textContent+=' · 模型无关待审核 '+state.model_review.pending;
  if(state.error_review_counts?.total)$('counts').textContent+=' · 错例复核 '+(state.error_review_counts.total-state.error_review_counts.pending)+'/'+state.error_review_counts.total+' · 位置补标 '+(state.annotation_counts?.confirmed||0)+'张';
  const proposals=state.annotation_proposal_counts;
  S.proposalCounts=proposals;
  if(proposals?.total)$('counts').textContent+=' · 自动框待审 '+proposals.pending;
  const proposalOption=$('statusFilter').querySelector('option[value="auto_boxes"]');
  if(proposalOption)proposalOption.textContent='待我审核 · 自动框（'+(proposals?.pending||0)+'）';
  const errorOption=$('statusFilter').querySelector('option[value="error_review"]');
  if(errorOption)errorOption.textContent='错例待复核 / 补标（'+(state.error_review_counts?.pending||0)+'）';
  renderStatusControls();
  $('scanStatus').textContent=state.scan.running?'后台扫描中，已发现 '+state.scan.found+' 张；可以先分类已索引图片。':state.scan.error||'目录索引已更新 · 原图按需读取 · 预览缓存上限 256MB';
  $('scanStatus').classList.toggle('error',!!state.scan.error);
  $('rescan').disabled=state.scan.running;$('openRoot').disabled=state.scan.running;
  if(tagsChanged)renderTags();else renderWorkflow();return state;
}
async function poll(){
  try{
    const state=await refreshState();
    if(S.busy){setTimeout(poll,2500);return;}
    if(S.loaded&&(state.counts.total!==lastIndexed||(scanWasRunning&&!state.scan.running))){
      await reload({target:S.current?.id,index:S.index,keep:!!S.current});
    }
    lastIndexed=state.counts.total;scanWasRunning=state.scan.running;
  }catch(error){$('scanStatus').textContent='本机服务连接暂时中断，请重新打开桌面入口。已有标签保存在本机。';$('scanStatus').classList.add('error');}
  setTimeout(poll,2500);
}
on('openRoot','click',async()=>{await saveDraft();const state=await api('/api/root',{path:$('rootPath').value});S.root=state.root;S.selected.clear();renderSelection();await refreshState();await reload({target:state.root.last_image});});
on('rescan','click',async()=>{await api('/api/scan',{});await refreshState();});
on('statusFilter','change',()=>changeStatusFilter($('statusFilter').value));
for(const button of $('commonStatusFilters').querySelectorAll('button[data-status]'))button.addEventListener('click',()=>changeStatusFilter(button.dataset.status)?.catch(fail));
on('reviewKind','change',()=>reload());on('search','input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>reload().catch(fail),300);});
on('clearFilters','click',()=>{S.filters.defect.clear();S.filters.responsibility.clear();return reload();});
on('thumbViewport','scroll',scheduleRender);window.addEventListener('resize',scheduleRender);
on('note','input',()=>{if(S.current){S.current.label.note=$('note').value;edited();}});
for(const button of $('imageStates').children)button.addEventListener('click',()=>{if(S.current&&!S.busy&&!S.navigating){const label=S.current.label;label.image_state=button.dataset.state;label.defects=[];if(label.problem_responsibilities){label.problem_responsibilities={};label.responsibilities=['待核'];}edited();}});
for(const [id,group]of [['addDefect','defect'],['addResponsibility','responsibility']])on(id,'submit',async e=>{e.preventDefault();const input=$(id).elements.name;S.tags=await api('/api/tag',{group,name:input.value});input.value='';renderTags();});
on('previous','click',()=>selectIndex(S.index-1));on('next','click',()=>selectIndex(S.index+1));on('skip','click',()=>selectIndex(S.index+1));on('commit','click',commit);
on('showLibrary','click',()=>document.body.classList.toggle('library-open'));
on('workspacePrevious','click',()=>selectIndex(S.index-1));
on('workspaceCommit','click',submitWorkspace);
on('imageStateToggle','click',()=>setLabelPanel(!$('labelPanel').hidden&&$('labelPanel').dataset.panelView==='state'?null:'state'));
on('labelPanelToggle','click',()=>setLabelPanel(!$('labelPanel').hidden&&$('labelPanel').dataset.panelView==='details'?null:'details'));
on('labelPanelClose','click',()=>setLabelPanel());
on('backToProblems','click',()=>{S.labelStep=1;renderTags();});
on('selectReviewPage','click',async()=>{const page=Math.floor(S.index/40);await getPage(page);for(const item of S.pages.get(page)||[])S.selected.add(item.id);renderSelection();renderThumbs();});
on('confirmReviewSelected','click',async()=>{
  if(S.busy||!S.selected.size)return;S.busy=true;renderTags();
  try{await saveDraft();const result=await api('/api/review/unrelated',{ids:[...S.selected],accept:true});S.selected.clear();renderSelection();await refreshState();await reload({index:S.index});toast('已人工确认 '+result.saved+' 张无关图片，可撤销');}
  finally{S.busy=false;renderTags();}
});
on('rejectUnrelated','click',async()=>{
  if(!S.current||S.busy||S.navigating||S.dirty)return;
  const id=S.current.id;await api('/api/review/unrelated',{ids:[id],accept:false});
  $('statusFilter').value='unclassified';S.selected.clear();renderSelection();await refreshState();await reload({target:id});toast('已记录模型判错，请继续标注实际问题点或图片状态');
});
on('adoptRecommendations','click',()=>adoptRecommendations((S.recommendation?.candidates||[]).map(c=>c.name)));
on('retryRecommendations','click',()=>requestRecommendations(S.current,S.nav,true));
on('recommendationPrevious','click',()=>{S.recommendationPage=Math.max(0,S.recommendationPage-1);renderRecommendations();});
on('recommendationNext','click',()=>{S.recommendationPage++;renderRecommendations();});
on('responsibilityProblem','change',()=>{S.responsibilityProblem=$('responsibilityProblem').value;renderTags();});
on('applyResponsibilityAdvice','click',()=>{
  const problem=S.responsibilityProblem,label=S.current?.label,role=S.links[problem]?.suggestion;
  if(S.busy||S.navigating||!role||!label?.problem_responsibilities||label.problem_responsibilities[problem]?.length)return;
  label.problem_responsibilities[problem]=[role];syncResponsibilities(label);edited();
});
on('responsibilityMatrix','toggle',()=>{
  if(!$('responsibilityMatrix').open)return;const rows=$('responsibilityRows');rows.replaceChildren();
  for(const [problem,value]of Object.entries(S.links)){
    const tr=document.createElement('tr');
    for(const text of [problem,value.suggestion||'人工选择',String(value.resolved_cases),Object.entries(value.responsibility_case_counts).sort((a,b)=>b[1]-a[1]).map(([role,n])=>role+' '+n+'单').join('、')||'暂无明确样本']){
      const td=document.createElement('td');td.textContent=text;tr.append(td);
    }rows.append(tr);
  }
});
on('nextResponsibility','click',()=>{const label=S.current?.label;if(!label?.problem_responsibilities)return;const pending=label.defects.filter(p=>!label.problem_responsibilities[p]?.length);S.responsibilityProblem=pending.find(p=>p!==S.responsibilityProblem)||pending[0]||S.responsibilityProblem;renderTags();});
on('restoreDefaultResponsibility','click',()=>{const label=S.current?.label,problem=S.responsibilityProblem;if(S.busy||S.navigating||!label?.problem_responsibilities||!S.rules[problem])return;label.problem_responsibilities[problem]=[S.rules[problem]];syncResponsibilities(label);edited();});
on('undo','click',async()=>{await saveDraft();const result=await api('/api/undo',{});await refreshState();await reload({target:result.image_id});toast('已撤销 '+result.undone+' 张的上次确认，选择保留为暂存');});
on('locate','click',async()=>{
  if(!S.current)return;
  const result=await api('/api/locate',{id:S.current.id});
  if(result.mode==='copy'){
    if(navigator.clipboard&&window.isSecureContext){await navigator.clipboard.writeText(result.path);toast('原文件路径已复制，可在资源管理器粘贴打开');}
    else{
      const field=$('rootPath'),previous=field.value;field.value=result.path;field.focus();field.select();
      const copied=document.execCommand('copy');
      if(copied){field.value=previous;toast('原文件路径已复制，可在资源管理器粘贴打开');}
      else toast('原文件路径已显示在顶部，请按 Ctrl+C 复制');
    }
  }
});on('fit','click',resetTransform);on('rotate','click',()=>{S.rotation=(S.rotation+90)%360;transform();});
on('fullResolution','click',()=>S.current&&showPhoto(S.current,true));on('retryImage','click',()=>S.current&&showPhoto(S.current));
on('contextDetails','toggle',async()=>{if($('contextDetails').open&&S.current){const id=S.current.id,data=await api('/api/context?id='+id);if(id===S.current?.id)$('caseContext').textContent=JSON.stringify(data,null,2);}});
$('photoStage').addEventListener('wheel',e=>{if(!S.current||e.target.closest('#boxEditor'))return;e.preventDefault();S.zoom=Math.min(8,Math.max(.5,S.zoom*(e.deltaY<0?1.15:1/1.15)));transform();if(S.zoom>1.8&&!S.original){S.original=true;showPhoto(S.current,true).catch(fail);}},{passive:false});
let drag=null;
on('photoStage','pointerdown',e=>{if(e.button!==0||e.target.closest('button,#boxEditor'))return;drag={x:e.clientX,y:e.clientY,px:S.panX,py:S.panY};$('photoStage').setPointerCapture(e.pointerId);});
on('photoStage','pointermove',e=>{if(drag){S.panX=drag.px+e.clientX-drag.x;S.panY=drag.py+e.clientY-drag.y;transform();}});
on('photoStage','pointerup',()=>drag=null);on('photoStage','pointercancel',()=>drag=null);
on('clearSelection','click',()=>{S.selected.clear();renderSelection();renderThumbs();});
on('batch','click',()=>{
  if(!S.current)return;
  const value=S.current.label;
  setLabelPanel('details');
  $('batchSummary').textContent='将当前图片标签追加到 '+S.selected.size+' 张：问题点 '+(value.defects.join('、')||value.image_state||'未选择')+'；责任 '+value.responsibilities.join('、')+(value.note?'；备注也将追加。':'。')+'原有标签保留，完成后可撤销。';
  $('batchPanel').hidden=false;$('batchPanel').scrollIntoView({block:'nearest'});
});
on('cancelBatch','click',()=>$('batchPanel').hidden=true);
on('applyBatch','click',async()=>{await saveDraft();const result=await api('/api/batch',{ids:[...S.selected],label:S.current.label});S.selected.clear();renderSelection();await refreshState();await reload({index:S.index});toast('已批量确认 '+result.saved+' 张');});
on('toggleImport','click',()=>$('importPanel').hidden=!$('importPanel').hidden);
on('previewImport','click',async()=>{const result=await api('/api/import/preview',{path:$('manifestPath').value});S.importKey=result.key;$('importSummary').textContent='可确认 '+result.completed_count+' 张，责任待补暂存 '+result.draft_count+' 张；冲突 '+result.conflicts+' 张；需手工处理的文件夹：'+(result.unknown_folders.join('、')||'无')+'。'+result.notice;$('applyImport').hidden=!(result.completed_count+result.draft_count);});
on('applyImport','click',async()=>{await saveDraft();const result=await api('/api/import/apply',{key:S.importKey});$('applyImport').hidden=true;await refreshState();await reload({index:S.index});toast('已导入 '+result.imported+' 张确认标签、'+result.drafts+' 张暂存');});
on('export','click',async()=>{await saveDraft();const link=document.createElement('a');link.href='/api/export?'+params({format:$('exportFormat').value});link.download='';link.click();toast('导出当前筛选的已确认标签；暂存请先按 Enter 确认');});
document.addEventListener('keydown',e=>{
  if(e.defaultPrevented)return;
  if(e.key==='Escape'){setLabelPanel();for(const details of document.querySelectorAll('.add-tag-details[open],#contextDetails[open],#responsibilityMatrix[open]'))details.open=false;return;}
  if(e.key==='Enter'&&e.target.closest('#recommendationPanel'))return;
  if(e.target.matches('input,textarea,select,[contenteditable=true]'))return;
  if(e.key==='Enter'&&e.target.closest('button')&&!['commit','workspaceCommit'].includes(e.target.closest('button').id))return;
  if(e.ctrlKey&&e.key.toLowerCase()==='z'){e.preventDefault();$('undo').click();return;}
  if(e.key==='Enter'){e.preventDefault();(e.target.closest('#labelPanel')?commit():submitWorkspace()).catch(fail);return;}
  if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();selectIndex(S.index+(e.key==='ArrowLeft'?-1:1)).catch(fail);return;}
  if(e.code.startsWith('Digit')&&!e.ctrlKey&&!e.altKey&&!$('labelPanel').hidden){const n=Number(e.code.slice(5))-1,group=e.shiftKey?'responsibility':'defect';if((e.shiftKey&&S.labelStep!==2)||(!e.shiftKey&&S.labelStep===2))return;if(n>=0&&n<5&&S.tags[group][n]){e.preventDefault();toggleTag(group,S.tags[group][n]);}}
});
window.addEventListener('beforeunload',e=>{if(S.dirty){e.preventDefault();e.returnValue='';}});
document.addEventListener('click',e=>{for(const details of document.querySelectorAll('.add-tag-details[open],#contextDetails[open],#responsibilityMatrix[open]'))if(!details.contains(e.target))details.open=false;});
document.addEventListener('click',e=>{if(!$('labelPanel').hidden&&!e.target.closest('#labelPanel,#labelPanelToggle,#imageStateToggle'))setLabelPanel();});
const launchStatus=new URLSearchParams(location.search).get('status');
if(launchStatus&&[...$('statusFilter').options].some(option=>option.value===launchStatus))$('statusFilter').value=launchStatus;
renderStatusControls();
(async()=>{
  const state=await refreshState();scanWasRunning=state.scan.running;lastIndexed=state.counts.total;
  await reload();S.loaded=true;poll();
})().catch(error=>{$('scanStatus').textContent='连接失败：'+error.message;$('scanStatus').classList.add('error');fail(error);});
