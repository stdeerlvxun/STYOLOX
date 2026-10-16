const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync('web/index.html','utf8'),source=fs.readFileSync('web/app.js','utf8');
const filterHtml=html.match(/<select id="statusFilter">([\s\S]*?)<\/select>/)[1];
const options=[...filterHtml.matchAll(/<option value="([^"]+)">([^<]+)<\/option>/g)].map(match=>({value:match[1],textContent:match[2]}));
assert.equal(options.length,14,'All existing API statuses must remain accessible');
assert.equal(new Set(options.map(option=>option.value)).size,14);
const commonHtml=html.match(/<div id="commonStatusFilters"[\s\S]*?<\/div>/)[0];
const buttons=[...commonHtml.matchAll(/<button data-status="([^"]+)"/g)].map(match=>({dataset:{status:match[1]},setAttribute(name,value){this[name]=value;}}));
assert.deepEqual(buttons.map(button=>button.dataset.status),['auto_boxes','unclassified','annotation_pending','annotated']);
const nodes={statusFilter:{value:'unclassified',options,querySelector(selector){return options.find(option=>selector===`option[value="${option.value}"]`);}},
  commonStatusFilters:{querySelectorAll(){return buttons;}},statusMoreLabel:{},autoBoxCount:{},reviewKind:{},statusDetails:{open:true}};
let reloads=0,selections=0;
const context=vm.createContext({$:id=>nodes[id],S:{proposalCounts:{pending:0},selected:new Set([7])},renderSelection(){selections++;},async reload(){reloads++;}});
vm.runInContext(source.slice(source.indexOf('function renderStatusControls('),source.indexOf('function params(')),context);
function run(code){return vm.runInContext(code,context);}
run('renderStatusControls()');assert.equal(nodes.statusMoreLabel.textContent,'更多筛选');assert.equal(buttons[1]['aria-pressed'],'true');assert.equal(nodes.autoBoxCount.textContent,'0');
(async()=>{
  for(const status of options.map(option=>option.value)){
    await run(`changeStatusFilter(${JSON.stringify(status)})`);
    assert.equal(nodes.statusFilter.value,status);
    assert.equal(nodes.reviewKind.hidden,!status.startsWith('error_review'));
    assert.equal(nodes.statusDetails.open,false);
    const common=buttons.some(button=>button.dataset.status===status);
    assert.equal(nodes.statusMoreLabel.textContent.startsWith('更多：'),!common);
    assert.equal(buttons.filter(button=>button['aria-pressed']==='true').length,common?1:0);
  }
  assert.equal(reloads,14);assert.equal(selections,14);assert.equal(run('S.selected.size'),0);
  await run('changeStatusFilter("unknown")');assert.equal(reloads,14,'Invalid statuses must not call the API');
  run('S.proposalCounts.pending=27;renderStatusControls()');assert.equal(nodes.autoBoxCount.textContent,'27');
  assert.match(source,/if\(launchStatus&&\[\.\.\.\$\('statusFilter'\)\.options\]\.some/,'Launch links use the same canonical option set');
  assert.match(source,/\['annotation_pending','继续补框'\],\['unclassified','开始分类'\]/,'Empty automatic queue must offer next tasks');
  const emptyNodes={},node=id=>emptyNodes[id]??={value:id==='statusFilter'?'auto_boxes':'',children:[],replaceChildren(){this.children=[];},append(...children){this.children.push(...children);},addEventListener(){}};
  const emptyContext=vm.createContext({$:node,S:{proposalCounts:{total:12,pending:0}},scanWasRunning:false,
    document:{createElement(){return {children:[],append(...children){this.children.push(...children);},addEventListener(){}};}},saveIndicator(){},renderTags(){}});
  vm.runInContext(source.slice(source.indexOf('function empty('),source.indexOf('async function selectIndex(')),emptyContext);
  vm.runInContext('empty()',emptyContext);
  assert.equal(node('imageMessage').children[0].textContent,'本批自动框审核已完成');
  assert.deepEqual(node('imageMessage').children[2].children.map(button=>button.textContent),['继续补框','开始分类']);
  assert.equal(node('commit').disabled,true);
  console.log('Compact filters: four shortcuts, 14 preserved statuses, canonical state, direct links and empty-queue actions passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
