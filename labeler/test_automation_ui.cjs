const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
class Element{
  constructor(){this.children=[];this.textContent='';this.hidden=false;this.open=true;}
  append(...items){this.children.push(...items);}
  replaceChildren(){this.children=[];}
  addEventListener(){}
}
const nodes={},el=id=>nodes[id]??=new Element();
const context=vm.createContext({document:{getElementById:el,createElement:()=>new Element()}});
vm.runInContext(fs.readFileSync('web/automation.js','utf8'),context);
const render=value=>context.AutomationUI.render(value);
const state={counts:{classified:2312},recommendations:{automatic_classification_enabled:true,model_id:'classifier',
  automatic_counts:{completed:3623,waiting:30630,model_refresh:0,unresolved:0,failed:0}},
  annotation_proposal_counts:{pending:536},automation:{training:{status:'waiting_for_confirmed_images',
    pending_confirmed_images:17,minimum_confirmed_images:20,updated_at:'2026-10-26T00:00:00Z'},
    inference:{status:'refilled',attempted_images:72,generated_images:50,generated_boxes:149,
      total_attempted_images:144,total_generated_images:100,total_generated_boxes:298,total_errors:0}}};
render(state);
assert.match(el('automationSummary').textContent,/17\/20/);
assert.match(el('trainingReason').textContent,/还需 3 张/);
assert.equal(el('trainingProgress').value,17);
assert.match(el('inferenceBatch').textContent,/72 张.*50 张 \/ 149 框/);
assert.equal(el('inferenceStats').children[5].children[1].textContent,'536','Pending images must not be confused with cumulative batches');
assert.equal(el('inferenceStats').children[4].children[1].textContent,'0');
assert.equal(el('inferenceStats').children[3].children[1].textContent,'未记录');
state.automation.training={...state.automation.training,status:'training',epoch:7,max_epochs:50};
render(state);assert.match(el('trainingStatus').textContent,/正在训练/);
assert.equal(el('trainingStats').children[3].children[1].textContent,'7 / 最多 50');
state.automation.training.stale=true;render(state);
assert.match(el('automationSummary').textContent,/状态未更新/);
assert.match(el('trainingReason').textContent,/上报已超时/);
render({});assert.match(el('automationSummary').textContent,/等待上报/);
assert.ok(el('trainingProgress').hidden);
assert.equal(el('classificationStats').children[0].children[1].textContent,'未记录');
console.log('Automation UI: waiting/live/stale/missing state, actual epochs, zero vs unknown and batch/total/pending units passed.');
