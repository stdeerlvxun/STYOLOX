const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('web/app.js','utf8');
const commitSource=source.slice(source.indexOf('async function commit(){'),source.indexOf('async function refreshState(){'));
async function scenario({retained=false,reject=false}={}){
  const calls=[];let releaseSave;
  const saved=new Promise(resolve=>{releaseSave=resolve;});
  const context=vm.createContext({S:{current:{id:7,label:{defects:['崩边']}},index:2,labelStep:2,dirty:true},
    $:()=>({value:'automatic'}),renderTags(){},toast(){calls.push('toast');},params:args=>new URLSearchParams(args),
    async saveDraft(){calls.push('draft');},async refreshState(){throw Error('Statistics unavailable');},
    async api(path){if(path==='/api/commit'){calls.push('write');await saved;if(reject)throw Error('Save rejected');return {};}
      calls.push('queue');return {items:[{id:retained?7:8}],total:8};},
    async reload({index}){calls.push(['navigate',index]);}});
  vm.runInContext(commitSource,context);
  const work=vm.runInContext('commit()',context);
  await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(calls,['draft','write'],'Do not navigate before persistence succeeds');
  await vm.runInContext('commit()',context);
  assert.equal(calls.filter(x=>x==='write').length,1,'Repeated confirmation must not duplicate writes');
  releaseSave();
  if(reject){await assert.rejects(work,/Save rejected/);assert.deepEqual(calls,['draft','write']);assert.equal(context.S.dirty,true);}
  else{await work;assert.deepEqual(calls,['draft','write','queue',['navigate',retained?3:2],'toast']);}
  assert.equal(context.S.busy,false);
}
(async()=>{await scenario();await scenario({retained:true});await scenario({reject:true});
  console.log('Save navigation: persisted success, failed-save retention, queue positioning, duplicate-click guard and statistics outage passed.');
})().catch(error=>{console.error(error);process.exitCode=1;});
