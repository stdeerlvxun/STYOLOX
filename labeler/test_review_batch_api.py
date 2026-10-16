"""HTTP contract for batch status, idempotent registration and archive downloads."""
import hashlib
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from PIL import Image
from app import make_server

class ReviewBatchAPITest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        source=self.root/'source';source.mkdir();Image.new('RGB',(100,100),'white').save(source/'board.jpg')
        self.server=make_server(0,self.root/'data',str(source));self.server.daemon_threads=True
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url='http://127.0.0.1:'+str(self.server.server_port)
        for _ in range(100):
            if not self.server.app.scanner.status['running']:break
            time.sleep(.01)
        self.token=self.get('/api/state')['token'];self.store=self.server.app.store
        self.identity=self.store.one('SELECT id FROM images')['id']
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.server.app.store.db.close();self.temp.cleanup()
    def get(self,path):
        with urlopen(self.url+path,timeout=5) as r:return json.load(r)
    def post(self,path,body):
        request=Request(self.url+path,json.dumps(body).encode(),headers={'Content-Type':'application/json','X-Local-Token':self.token})
        with urlopen(request,timeout=5) as r:return json.load(r)
    def test_worker_and_empty_batches(self):
        initial=self.get('/api/review-batches/state');self.assertIsNone(initial['current'])
        self.post('/api/review-batches/worker',{'enabled':True,'status':'waiting_gpu','pending':0,'low_water':10,'batch_size':50})
        state=self.get('/api/state')['review_batches'];self.assertEqual(state['worker']['batch_size'],50)
        self.assertIn('checked_at',state['worker'])
        with self.assertRaises(HTTPError):self.post('/api/review-batches/worker',{'pending':-1})
        self.post('/api/review-batches/register',{'batch_id':'empty','proposal_ids':[]})
        self.assertEqual(self.get('/api/review-batches/state')['archives_count'],0)
    def test_completed_batch_archive_is_json_object_with_confirmed_result(self):
        identity=self.identity
        self.store.commit([(identity,{'defects':['崩边'],'responsibilities':['生产'],'problem_responsibilities':{'崩边':['生产']},'image_state':'defect','note':'保留'})])
        detail=self.store.annotations.detail(identity);source=detail['source']
        manifest={'schema_version':1,'requires_human_review':True,'model_id':'test-model','method':'test',
            'items':[{'image_id':identity,'relative_path':'board.jpg','label_fingerprint':detail['label_fingerprint'],'source':source,
            'boxes':[{'id':'p1','problem':'崩边','x':.1,'y':.1,'w':.2,'h':.2,'score':.8,'origin':'model'}]}]}
        installed=self.store.annotations.install_proposals(manifest);ids=installed['imported_proposal_ids']
        self.assertEqual(len(ids),1)
        self.post('/api/review-batches/register',{'batch_id':'batch-test','proposal_ids':ids,'metadata':{'source_model':'test-model'}})
        self.assertEqual(self.get('/api/review-batches/state')['current']['total'],1)
        self.post('/api/review-batches/register',{'batch_id':'batch-test','proposal_ids':ids})
        proposal=self.store.annotations.detail(identity)['proposal']
        self.post('/api/annotations',{'id':identity,'expected_version':0,'source_sha256':source['sha256'],
            'label_fingerprint':detail['label_fingerprint'],'proposal_id':ids[0],'boxes':proposal['boxes'],
            'status':'confirmed','outcome':'boxed'})
        state=self.get('/api/review-batches/state');self.assertEqual(state['archives_count'],1)
        archive=self.get('/api/review-batches/archive?batch=batch-test')
        self.assertIsInstance(archive,dict);self.assertIn('items',archive);self.assertEqual(len(archive['items']),1)
        self.assertEqual(json.loads(self.store.one('SELECT data FROM labels WHERE image_id=?',(identity,))['data'])['note'],'保留')
        with self.assertRaises(HTTPError):self.get('/api/review-batches/archive?batch=missing')

    def install_unclassified(self,batch):
        detail=self.store.annotations.detail(self.identity)
        manifest={'schema_version':1,'requires_human_review':True,'model_id':'test-location','method':'test',
            'items':[{'image_id':self.identity,'relative_path':'board.jpg','label_fingerprint':detail['label_fingerprint'],
                'source':detail['source'],'boxes':[{'id':'unknown-1','problem':'','x':.1,'y':.1,'w':.2,'h':.2,'score':.8,'origin':'model'}]}]}
        installed=self.store.annotations.install_proposals(manifest)
        self.assertEqual(installed['imported'],1)
        self.post('/api/review-batches/register',{'batch_id':batch,'proposal_ids':installed['imported_proposal_ids']})
        self.assertEqual(self.store.listing({'status':['auto_boxes']})['total'],1)
        self.assertIsNone(self.store.one('SELECT * FROM labels WHERE image_id=?',(self.identity,)))
        return self.store.annotations.detail(self.identity)

    def test_unclassified_selection_confirmation_and_archive(self):
        detail=self.install_unclassified('unknown-confirm')
        proposal=detail['proposal'];boxes=[dict(box) for box in proposal['boxes']]
        body={'id':self.identity,'expected_version':0,'source_sha256':detail['source']['sha256'],
            'label_fingerprint':detail['label_fingerprint'],'proposal_id':proposal['proposal_id'],
            'boxes':boxes,'status':'confirmed','outcome':'boxed'}
        with self.assertRaises(HTTPError):self.post('/api/annotations',body)
        boxes[0].update(problem='崩边',responsibilities=['生产'])
        body['classification_updates']={'defects':['崩边'],'problem_responsibilities':{'崩边':['生产']}}
        self.post('/api/annotations',body)
        self.assertEqual(self.get('/api/review-batches/state')['archives_count'],1)
        receipt=self.get('/api/review-batches/archive?batch=unknown-confirm')['items'][0]
        self.assertEqual(receipt['classification']['defects'],['崩边'])
        self.assertEqual(receipt['final_boxes'][0]['problem'],'崩边')
        quality=self.get('/api/review-quality/state')
        self.assertEqual(len(quality['groups']),1)
        self.assertFalse(quality['auto_enabled'])

    def test_unclassified_explicit_rejection_archives_without_inventing_normal_label(self):
        detail=self.install_unclassified('unknown-reject')
        self.store.annotations.reject_proposal(self.identity,detail['proposal']['proposal_id'])
        self.assertEqual(self.get('/api/review-batches/state')['archives_count'],1)
        receipt=self.get('/api/review-batches/archive?batch=unknown-reject')['items'][0]
        self.assertEqual(receipt['decision'],'rejected')
        self.assertIsNone(receipt['classification']);self.assertTrue(receipt['non_training'])
        self.assertIsNone(self.store.one('SELECT * FROM labels WHERE image_id=?',(self.identity,)))

if __name__=='__main__':unittest.main()
