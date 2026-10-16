"""Read-only position acceptance statistics for human-reviewed machine suggestions."""
from collections import Counter,defaultdict
import datetime as dt
import hashlib
import json
import math


def wilson_lower(successes,total):
    """One-sided 95% Wilson lower bound; descriptive, not a blind-test guarantee."""
    if not total:return None
    z=1.6448536269514722;rate=successes/total;z2=z*z
    return max(0.,(rate+z2/(2*total)-z*math.sqrt(rate*(1-rate)/total+z2/(4*total*total)))/(1+z2/total))


def _time(value):
    try:
        stamp=dt.datetime.fromisoformat(value.replace('Z','+00:00'))
        return stamp if stamp.tzinfo is not None else None
    except (ValueError,AttributeError):return None


def _hash(raw):return hashlib.sha256((raw or '').encode('utf-8')).hexdigest()


def _positions(original,final,rejected=False):
    old={box['id']:box for box in original};retained=set();unchanged=moved=added=assigned=changed=0
    for box in final:
        identity=box.get('source_box_id',box['id']);before=old.get(identity)
        if before is None or identity in retained:added+=1;continue
        retained.add(identity)
        shifted=any(abs(box[k]-before[k])>1e-6 for k in ('x','y','w','h'))
        moved+=shifted;unchanged+=not shifted
        assigned+=not before.get('problem') and bool(box.get('problem'))
        changed+=bool(before.get('problem')) and box.get('problem')!=before.get('problem')
    removed=len(old)-len(retained)
    return {'whole_image_unchanged':int(bool(original) and not rejected and not (removed or moved or added)),
        'reviewed_machine_boxes':len(original),'unchanged_position_boxes':unchanged,
        'deleted_machine_boxes':removed,'moved_machine_boxes':moved,'human_added_boxes':added,
        'problem_type_assigned':assigned,'problem_type_changed':changed,
        'inherited_problem_labels':sum(bool(box.get('problem')) for box in original)}


class ReviewQuality:
    def __init__(self,store):self.store=store

    def _terminal_receipts(self,rows,root_id):
        candidates=[row for row in rows if row['label'] and
            row['label'].get('image_state') in ('normal','unrelated') and not row['label'].get('defects')]
        if not candidates:return {}
        ids=sorted({row['image_id'] for row in candidates});receipts={}
        since=min(row['created_at'] for row in candidates)
        # Only the relevant root, image identities and proposal time range are examined.
        for offset in range(0,len(ids),400):
            chunk=ids[offset:offset+400]
            history=self.store.query("SELECT h.id,h.created_at,e.value change_data FROM history h,json_each(h.data) e "
                "WHERE h.root_id=? AND h.undone=0 AND h.kind IN ('commit','batch','review-unrelated') "
                "AND h.created_at>=? AND json_extract(e.value,'$.id') IN ("+','.join('?' for _ in chunk)+') ORDER BY h.id DESC',
                (root_id,since[:19],*chunk))
            for row in history:
                change=json.loads(row['change_data'])
                receipts.setdefault(change['id'],[]).append((row['created_at'],change.get('after')))
        return receipts

    def _review(self,row,receipts):
        data=row['proposal'];label=row['label'];document=row['document'];source=data.get('source',{})
        if not data.get('model_id') or not data.get('method') or not data.get('boxes'):return None,'invalid_proposal'
        if not row['available']:return None,'source_unavailable'
        if any(source.get(k)!=row[k] for k in ('size','mtime_ns')):return None,'source_changed'
        if row['draft_image_id'] is not None:return None,'classification_draft'
        if (label or {}).get('image_state')=='unclear' or (document or {}).get('outcome')=='unclear':
            return None,'unclear'
        state=(label or {}).get('image_state')
        if state in ('normal','unrelated'):
            if label.get('defects'):return None,'ambiguous_image_state'
            created=_time(row['created_at']);updated=_time(row['label_updated_at'])
            if not created or not updated or updated<=created:return None,'state_confirmation_missing'
            if not any(_time(stamp)==updated and after==label for stamp,after in receipts.get(row['image_id'],[])):
                return None,'state_confirmation_missing'
            if document:
                annotation_time=_time(row['annotation_updated_at'])
                if not annotation_time or annotation_time>updated:return None,'annotation_changed_after_state'
            return {'final':[],'rejected':True,'terminal':True,'reviewed_at':row['label_updated_at']},None
        if row['status'] not in ('accepted','rejected') or not _time(row['reviewed_at']):return None,'not_reviewed'
        if document:
            if document.get('status')!='confirmed':return None,'annotation_draft'
            if document.get('proposal_id')!=row['proposal_id']:return None,'annotation_proposal_changed'
            if document.get('label_fingerprint')!=_hash(row['label_data_current']):return None,'classification_changed'
            if any(document.get('source',{}).get(k)!=source.get(k) for k in ('sha256','width','height')):
                return None,'annotation_source_changed'
            if row['status']=='accepted' and document.get('outcome')!='boxed':return None,'ambiguous_annotation'
            if row['status']=='rejected' and document.get('boxes'):return None,'ambiguous_rejection'
            return {'final':document.get('boxes',[]),'rejected':row['status']=='rejected','terminal':False,
                'reviewed_at':row['annotation_updated_at'] or row['reviewed_at']},None
        if row['status']=='accepted':return None,'confirmed_annotation_missing'
        if (row['label_data_current'] or '')!=row['label_data']:return None,'classification_changed'
        return {'final':[],'rejected':True,'terminal':False,'reviewed_at':row['reviewed_at']},None

    @staticmethod
    def _finish(group,cases):
        count=group['reviewed_images'];boxes=group['reviewed_machine_boxes']
        group['whole_image_unchanged_rate']=group['whole_image_unchanged']/count if count else None
        group['whole_image_rate_lower95']=wilson_lower(group['whole_image_unchanged'],count)
        group['unchanged_position_rate']=group['unchanged_position_boxes']/boxes if boxes else None
        group['distinct_cases']=group['reviewed_cases']=len(cases)
        group['unchanged_cases']=sum(all(results) for results in cases.values())
        group['case_unchanged_rate']=group['unchanged_cases']/len(cases) if cases else None
        group['case_rate_lower95']=wilson_lower(group['unchanged_cases'],len(cases))
        shortfalls=[]
        if count<300:shortfalls.append('已审图片不足300张')
        if len(cases)<100:shortfalls.append('不同售后单不足100单')
        if group['whole_image_unchanged_rate'] is None or group['whole_image_unchanged_rate']<.98:
            shortfalls.append('整图位置原样通过率未达到98%')
        if group['case_rate_lower95'] is None or group['case_rate_lower95']<.95:
            shortfalls.append('按售后单计算的95%单侧下界未达到95%')
        if group['images_without_case_id']:shortfalls.append('部分图片缺少售后单分组，不计独立单数')
        group['position_sample_gate_met']=not shortfalls
        shortfalls.extend(['独立盲测尚未通过','问题类型及责任准确率尚未验证'])
        group.update(shortfalls=shortfalls,auto_enabled=False)
        return group

    def state(self):
        root=self.store.current_root()
        result={'schema_version':1,'scope':'human_reviewed_machine_proposals_not_independent_test',
            'source_validation':'archive_index_only_no_nas_read',
            'confidence_scope':'descriptive_review_sample_grouped_by_case_not_independent_blind_test',
            'groups':[],'current':None,'latest_generated_group_key':None,'excluded_reasons':{},
            'scanned_proposals':0,'reviewed_images':0,'auto_enabled':False,
            'criteria':{'whole_image_unchanged_rate':.98,'case_rate_lower95':.95,
                'minimum_images':300,'minimum_cases':100,'basis':'business_acceptance_suggestion_not_industry_standard'}}
        if not root:return result
        with self.store.lock:
            rows=self.store.query('''SELECT p.*,i.case_id,i.available,i.size,i.mtime_ns,
                l.data label_data_current,l.updated_at label_updated_at,d.image_id draft_image_id,
                a.data annotation_data,a.updated_at annotation_updated_at
                FROM (SELECT image_id,proposal_id,data,label_data,status,created_at,reviewed_at FROM annotation_proposals
                  UNION ALL SELECT image_id,proposal_id,data,label_data,status,created_at,reviewed_at FROM annotation_proposal_history) p
                JOIN images i ON i.id=p.image_id LEFT JOIN labels l ON l.image_id=i.id
                LEFT JOIN drafts d ON d.image_id=i.id LEFT JOIN annotation_documents a ON a.image_id=i.id
                WHERE i.root_id=? ORDER BY p.created_at DESC''',(root['id'],))
            unique={};excluded=Counter()
            for row in rows:
                if row['proposal_id'] in unique:continue
                try:
                    row['proposal']=json.loads(row['data'])
                    row['label']=json.loads(row['label_data_current']) if row['label_data_current'] else None
                    row['document']=json.loads(row['annotation_data']) if row['annotation_data'] else None
                except (TypeError,ValueError):excluded['invalid_json']+=1;continue
                unique[row['proposal_id']]=row
            rows=list(unique.values());result['scanned_proposals']=len(rows)
            receipts=self._terminal_receipts(rows,root['id'])
            groups={};reviewed={};case_results=defaultdict(lambda:defaultdict(list))
            latest_generated=None
            keys=('whole_image_unchanged','reviewed_machine_boxes','unchanged_position_boxes','deleted_machine_boxes',
                'moved_machine_boxes','human_added_boxes','problem_type_assigned','problem_type_changed','inherited_problem_labels')
            for row in rows:
                data=row['proposal'];key=json.dumps([data.get('model_id'),data.get('method')],ensure_ascii=False,separators=(',',':'))
                if key not in groups:
                    groups[key]={**dict.fromkeys(keys,0),'key':key,'model_id':data.get('model_id'),'method':data.get('method'),
                        'reviewed_images':0,'terminal_state_rejections':0,'images_without_case_id':0,
                        'latest_reviewed_at':None,'latest_proposal_at':row['created_at'],
                        'type_metric_scope':'assigned_or_inherited_labels_not_model_type_accuracy'}
                if latest_generated is None or row['created_at']>latest_generated[0]:latest_generated=(row['created_at'],key)
                try:review,reason=self._review(row,receipts)
                except (TypeError,ValueError,KeyError):review,reason=None,'unreconstructable_review'
                if not review:excluded[reason]+=1;continue
                previous=reviewed.get(row['image_id'])
                if previous and previous[1]['reviewed_at']>=review['reviewed_at']:
                    excluded['older_review_same_image']+=1;continue
                if previous:excluded['older_review_same_image']+=1
                reviewed[row['image_id']]=(row,review,key)
            for row,review,key in reviewed.values():
                group=groups[key]
                try:stats=_positions(row['proposal']['boxes'],review['final'],review['rejected'])
                except (TypeError,ValueError,KeyError):excluded['unreconstructable_boxes']+=1;continue
                for name,value in stats.items():group[name]+=value
                group['reviewed_images']+=1;group['terminal_state_rejections']+=review['terminal']
                group['latest_reviewed_at']=max(group['latest_reviewed_at'] or '',review['reviewed_at'])
                if row['case_id']:case_results[key][row['case_id']].append(bool(stats['whole_image_unchanged']))
                else:group['images_without_case_id']+=1
            result['groups']=sorted((self._finish(g,case_results[key]) for key,g in groups.items()),
                key=lambda g:(g['latest_proposal_at'],g['latest_reviewed_at'] or ''),reverse=True)
            result['current']=next((g for g in result['groups'] if g['reviewed_images']),None)
            result['latest_generated_group_key']=latest_generated[1] if latest_generated else None
            result['excluded_reasons']=dict(excluded)
            result['reviewed_images']=sum(g['reviewed_images'] for g in result['groups'])
        return result
