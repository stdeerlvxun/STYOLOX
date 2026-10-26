/* Read-only operational status, refreshed by the existing page poll. */
(() => {
  const $=id=>document.getElementById(id);
  const number=value=>Number.isFinite(value)?value.toLocaleString('zh-CN'):'未记录';
  const when=value=>{
    if(value===null||value===undefined||value==='')return '未上报';
    const date=new Date(typeof value==='number'?value*1000:value);
    return Number.isNaN(date.getTime())?'未上报':date.toLocaleString('zh-CN',{hour12:false});
  };
  const names={
    waiting_for_confirmed_images:'等待更多合规纠错',waiting_for_trainable_confirmed_images:'等待可用于训练的纠错',
    skipped_gpu_busy_or_unavailable:'等待 GPU 空闲',skipped_another_worker_running:'等待当前任务结束',
    preparing:'准备训练数据',training:'正在训练',validating:'正在比较新旧模型',
    trained_not_promoted:'训练完成，继续使用原模型',promoted_awaiting_review_refill:'新检测模型已启用',
    promoted_pending_human_review:'新检测模型已启用',paused:'同批连续失败，已暂停重试',
    generating:'正在检测新图片',refilled:'本批补框完成',waiting_for_gpu:'等待 GPU 空闲',
    waiting_for_worker:'等待当前 GPU 任务',no_boxes_this_round:'本批未检出问题框',
    no_more_eligible_images:'等待新的合格图片',no_candidates_imported:'本批候选未导入',
    disabled:'已暂停',error:'运行异常',queue_sufficient:'等待续批'
  };
  const reasons={
    validation_support_below_minimum:'验证问题框数量不足，保留原模型。',
    validation_recall_below_minimum:'找回能力未达到晋级门槛，保留原模型。',
    validation_recall_regressed:'找回能力下降，保留原模型。',
    validation_f2_has_no_strict_improvement:'综合验证指标未改善，保留原模型。',
    location_metrics_regressed:'位置找回或误报指标退步，保留原模型。',
    same_validation_f2_improved_and_recall_did_not_regress:'同一验证集指标改善，已通过晋级检查。'
  };
  function rows(id, values){
    const target=$(id);target.replaceChildren();
    for(const [label,value] of values){
      const row=document.createElement('div'),term=document.createElement('dt'),description=document.createElement('dd');
      term.textContent=label;description.textContent=typeof value==='string'?value:number(value);
      row.append(term,description);target.append(row);
    }
  }
  function status(id, value){
    const node=$(id);
    node.textContent=value?((value.stale?'状态未更新 · ':'')+(names[value.status]||'等待状态确认')):'等待电脑上报';
    node.className=value?.stale?'status-stale':value?.status==='error'?'status-error':'';
  }
  function render(state){
    const recommendation=state.recommendations||{},counts=recommendation.automatic_counts||{};
    const training=state.automation?.training,inference=state.automation?.inference;
    $('classificationStatus').textContent=recommendation.automatic_classification_enabled?'持续处理，人工纠正不阻塞':recommendation.error?'分类服务异常：'+recommendation.error:'自动分类未启用';
    rows('classificationStats',[
      ['机器结果（未人工确认）',counts.completed],['待首次识别 / 源图更新',counts.waiting],
      ['待更新模型',counts.model_refresh],['结果中未明确识别',counts.unresolved],['失败待重试',counts.failed],
      ['人工分类',state.counts?.classified]]);
    $('classificationModel').textContent='分类模型：'+(recommendation.model_id||'未配置');
    $('classificationUpdated').textContent='最近有效机器结果：'+when(counts.last_success_at);
    status('inferenceStatus',inference);
    rows('inferenceStats',[
      ['累计尝试（张次）',inference?.total_attempted_images],['累计出框（张次）',inference?.total_generated_images],
      ['累计生成框数',inference?.total_generated_boxes],['累计未出框（张次）',inference?.total_images_without_boxes],
      ['累计处理错误',inference?.total_errors],['当前待审核（图片）',state.annotation_proposal_counts?.pending]]);
    $('inferenceBatch').textContent=inference?'本批尝试 '+number(inference.attempted_images)+' 张 · 出框 '+number(inference.generated_images)+' 张 / '+number(inference.generated_boxes)+' 框 · 未出框 '+number(inference.images_without_boxes)+' 张':'尚无补框运行记录';
    $('inferenceModel').textContent='检测模型：'+(inference?.model_id||'未上报');
    $('inferenceUpdated').textContent='状态更新：'+when(inference?.updated_at)+'；最近成功批次记录：'+when(inference?.last_success_at);
    status('trainingStatus',training);
    const pending=training?.pending_confirmed_images,minimum=training?.minimum_confirmed_images;
    const progress=$('trainingProgress');
    progress.hidden=!Number.isFinite(pending)||!Number.isFinite(minimum);
    if(!progress.hidden){progress.max=minimum;progress.value=Math.min(pending,minimum);}
    rows('trainingStats',[
      ['合规新纠错',number(pending)+' / '+number(minimum)+' 张'],
      ['本轮实际训练图片',training?.trainable_confirmed_images],['本轮验证图片',training?.validation_images],
      ['已完成训练轮次',Number.isFinite(training?.epoch)?number(training.epoch)+' / 最多 '+number(training.max_epochs):'未记录']]);
    const remaining=Number.isFinite(pending)&&Number.isFinite(minimum)?Math.max(0,minimum-pending):null;
    $('trainingReason').textContent=training?.stale?'电脑上报已超时，请以最后更新时间为准。':
      training?.reason||reasons[training?.promotion_reason]||training?.promotion_reason||
      (remaining>0?'还需 '+remaining+' 张合规确认纠错；整图分类数量不等于可训练数量。':'训练与验证完成后，才决定是否启用候选模型。');
    $('trainingUpdated').textContent='状态更新：'+when(training?.updated_at);
    $('automationSummary').textContent='后台进度 · '+(!training?'等待上报':training.stale?'状态未更新':
      training.status==='waiting_for_confirmed_images'?number(pending)+'/'+number(minimum)+' 张':names[training.status]||'查看状态');
  }
  $('automationClose')?.addEventListener('click',()=>{$('automationPanel').open=false;});
  globalThis.AutomationUI={render};
})();
