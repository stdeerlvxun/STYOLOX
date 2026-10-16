"""Isolated integration checks: no real archive files or user labels are modified."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from PIL import Image
from app import make_server, dumps


class WorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.source = cls.base / "archive" / "cases"
        cls.case = cls.source / "2026-10" / "11111111-1111-1111-1111-111111111111"
        images = cls.case / "images" / "main"
        images.mkdir(parents=True)
        for i in range(53):
            Image.new("RGB", (100 + i, 80), (20 + i, 80, 130)).save(images / f"{i:03d}__sample.jpg")
        (cls.case / "case.json").write_text(dumps({"after_sales": {"number": "测试单",
                       "reported_at": "2026-10-03T10:00:00+08:00"}}), encoding="utf-8")
        cls.original_hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.source.rglob("*.jpg")}
        cls.server = make_server(0, cls.base / "labels", str(cls.source))
        cls.url = "http://127.0.0.1:" + str(cls.server.server_port)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.wait_scan()
        cls.token = cls.get("/api/state")["token"]

    @classmethod
    def wait_scan(cls):
        for _ in range(200):
            if not cls.server.app.scanner.status["running"]:
                return
            time.sleep(.01)
        raise AssertionError("scan timeout")

    @classmethod
    def get(cls, route, raw=False):
        with urlopen(cls.url + route, timeout=10) as response:
            data = response.read()
            return data if raw else json.loads(data)

    @classmethod
    def post(cls, route, body):
        request = Request(cls.url + route, data=dumps(body).encode(),
                          headers={"Content-Type": "application/json", "X-Local-Token": cls.token}, method="POST")
        with urlopen(request, timeout=10) as response:
            return json.loads(response.read())

    def setUp(self):
        store = self.server.app.store
        with store.lock, store.db:
            for table in ("labels", "drafts", "history"):
                store.db.execute("DELETE FROM " + table)

    def test_workflow(self):
        page = self.get("/api/images?limit=40")
        self.assertEqual((page["total"], len(page["items"])), (53, 40))
        self.assertEqual(len(self.get("/api/images?limit=40&offset=40")["items"]), 13)
        first, second = page["items"][0]["id"], page["items"][1]["id"]
        label = {"defects": ["崩边", "划伤"], "responsibilities": ["运输", "安装"],
                 "image_state": "defect", "note": "=测试备注"}
        self.post("/api/draft", {"id": first, "label": label})
        self.assertEqual(self.get("/api/state")["counts"]["classified"], 0)
        self.assertEqual(self.get("/api/item?id=" + str(first))["label"], label)
        self.post("/api/commit", {"id": first, "label": label})
        from urllib.parse import urlencode
        self.assertEqual(self.get("/api/images?" + urlencode({"defect": "崩边", "responsibility": "运输"}))["total"], 1)
        self.assertEqual(self.get("/api/images?" + urlencode({"defect": "崩边", "responsibility": "生产"}))["total"], 0)
        self.assertEqual(self.get("/api/images?status=unclassified")["total"], 52)
        self.post("/api/batch", {"ids": [second], "label": label})
        self.assertEqual(self.get("/api/state")["counts"]["classified"], 2)
        self.post("/api/undo", {})
        self.assertEqual(self.get("/api/state")["counts"]["classified"], 1)
        self.assertTrue(self.get("/api/item?id=" + str(second))["has_draft"])
        data = self.get("/api/image?id=" + str(first) + "&kind=thumb", raw=True)
        with Image.open(io.BytesIO(data)) as image:
            self.assertLessEqual(image.width, 320)
        self.assertTrue(self.get("/api/item?id=" + str(first))["sha256"])
        self.assertEqual(self.get("/api/context?id=" + str(first))["after_sales"]["number"], "测试单")
        export = self.get("/api/export?status=classified&format=json")
        self.assertEqual(len(export["items"]), 1)
        self.assertEqual(export["items"][0]["responsibilities"], ["运输", "安装"])
        csv = self.get("/api/export?status=classified&format=csv", raw=True).decode("utf-8-sig")
        self.assertIn("'=测试备注", csv)
        self.post("/api/tag", {"group": "responsibility", "name": "售后处理"})
        self.assertIn("售后处理", self.get("/api/state")["tags"]["responsibility"])
        # A reopened database recovers committed labels and drafts.
        from app import Store
        reopened = Store(self.base / "labels")
        self.assertEqual(reopened.counts()["classified"], 1)
        self.assertTrue(reopened.detail(second)["has_draft"])
        reopened.db.close()
        for path, digest in self.original_hashes.items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_batch_conflict_is_atomic(self):
        ids = [r["id"] for r in self.get("/api/images?limit=2")["items"]]
        normal = {"defects": [], "responsibilities": ["待核"], "image_state": "normal", "note": ""}
        self.post("/api/commit", {"id": ids[1], "label": normal})
        damage = {"defects": ["崩边"], "responsibilities": ["运输"], "image_state": "defect", "note": ""}
        with self.assertRaises(HTTPError):
            self.post("/api/batch", {"ids": ids, "label": damage})
        self.assertEqual(self.get("/api/state")["counts"]["classified"], 1)
        self.assertEqual(self.get("/api/item?id=" + str(ids[1]))["label"]["image_state"], "normal")

    def test_unrelated_state_review_filter_does_not_change_labels(self):
        ids=[r['id'] for r in self.get('/api/images?limit=2')['items']]
        self.post('/api/commit',{'id':ids[0],'label':{'defects':[],'responsibilities':['待核'],'image_state':'unrelated','note':'待复核'}})
        before=self.get('/api/item?id='+str(ids[0]))['label']
        result=self.get('/api/images?status=unrelated')
        self.assertEqual(result['total'],1);self.assertEqual(result['items'][0]['id'],ids[0])
        self.assertEqual(self.get('/api/item?id='+str(ids[0]))['label'],before)

    def test_problem_responsibility_rules(self):
        from app import RESPONSIBILITY_RULES, clean_label
        production = ["缺角", "工艺问题", "残胶", "pet边高", "边高", "软成型工艺", "崩边", "开胶脱胶", "尺寸不符", "封边色差"]
        materials = ["板件色差", "五金质量"]
        self.assertEqual(self.get("/api/state")["responsibility_rules"], RESPONSIBILITY_RULES)
        for problem in production + materials:
            value = clean_label({"defects": [problem], "problem_responsibilities": {}}, complete=True)
            expected = "生产" if problem in production else "原材料"
            self.assertEqual(value["problem_responsibilities"], {problem: [expected]})
            self.assertEqual(value["responsibilities"], [expected])
        for problem in ["划伤", "其他破损", "板件瑕疵", "规范问题", "新增自定义问题"]:
            data = {"defects": ["残胶", problem], "problem_responsibilities": {}}
            self.assertEqual(clean_label(data)["problem_responsibilities"][problem], [])
            with self.assertRaisesRegex(ValueError, problem):
                clean_label(data, complete=True)
            data["problem_responsibilities"][problem] = ["待核"]
            value = clean_label(data, complete=True)
            self.assertEqual(value["responsibilities"], ["生产", "待核"])
        # An explicit exception can override an automatic default.
        value = clean_label({"defects": ["残胶"], "problem_responsibilities": {"残胶": ["运输"]}}, complete=True)
        self.assertEqual(value["responsibilities"], ["运输"])
        # Legacy data stays byte-for-byte semantically unchanged on notes-only edits.
        old = {"defects": ["崩边"], "responsibilities": ["运输"], "image_state": "defect", "note": "旧记录"}
        self.assertEqual(clean_label(old, complete=True), old)

    def test_problem_responsibilities_roundtrip_and_batch(self):
        ids = [r["id"] for r in self.get("/api/images?limit=3")["items"]]
        label = {"defects": ["残胶", "五金质量", "划伤"], "responsibilities": ["错误的汇总"],
                 "problem_responsibilities": {"划伤": ["运输"]}, "image_state": "defect", "note": "多问题测试"}
        self.post("/api/draft", {"id": ids[0], "label": label})
        restored = self.get("/api/item?id=" + str(ids[0]))["label"]
        self.assertEqual(restored["responsibilities"], ["生产", "原材料", "运输"])
        self.post("/api/commit", {"id": ids[0], "label": restored})
        self.assertEqual(self.get("/api/item?id=" + str(ids[0]))["label"], restored)
        exported = self.get("/api/export?status=classified&format=json")["items"][0]
        self.assertEqual(exported["problem_responsibilities"], restored["problem_responsibilities"])
        import csv
        csv_rows = list(csv.DictReader(io.StringIO(self.get("/api/export?status=classified&format=csv", raw=True).decode("utf-8-sig"))))
        self.assertEqual(json.loads(csv_rows[0]["problem_responsibilities"]), restored["problem_responsibilities"])
        # Preserve existing manual choices when appending an automatic problem point.
        self.post("/api/commit", {"id": ids[1], "label": {"defects": ["划伤"], "responsibilities": ["安装"]}})
        self.post("/api/batch", {"ids": [ids[1]], "label": {"defects": ["残胶"], "problem_responsibilities": {}}})
        merged = self.get("/api/item?id=" + str(ids[1]))["label"]
        self.assertEqual(merged["problem_responsibilities"], {"划伤": ["安装"], "残胶": ["生产"]})
        # Missing manual responsibility must not be silently satisfied by automatic production.
        incomplete = {"defects": ["残胶", "划伤"], "problem_responsibilities": {}}
        with self.assertRaises(HTTPError) as caught:
            self.post("/api/commit", {"id": ids[2], "label": incomplete})
        self.assertEqual(caught.exception.code, 400)
        self.assertEqual(self.get("/api/state")["counts"]["classified"], 2)
        self.post("/api/commit", {"id": ids[2], "label": {**incomplete, "problem_responsibilities": {"划伤": ["待核"]}}})
        from urllib.parse import urlencode
        self.assertEqual(self.get("/api/images?" + urlencode({"status": "pending", "defect": "划伤"}))["total"], 1)

    def test_import_and_request_boundary(self):
        item = self.get("/api/images?limit=1")["items"][0]
        directory = self.base / "archive" / "人工分类" / "batch"
        category = directory / "01_崩边"
        category.mkdir(parents=True)
        (category / "copied.jpg").write_bytes(b"fixture-name-only")
        manifest = self.base / "manifest.json"
        manifest.write_text(dumps({"classification_directory": str(directory), "items": [
            {"filename": "copied.jpg", "relative_path": "cases/" + item["relative_path"]}]}), encoding="utf-8")
        preview = self.post("/api/import/preview", {"path": str(manifest)})
        self.assertEqual(preview["completed_count"], 1)
        self.assertEqual(self.get("/api/state")["counts"]["classified"], 0)
        self.post("/api/import/apply", {"key": preview["key"]})
        self.assertEqual(self.get("/api/state")["counts"]["classified"], 1)
        # A request without the page token cannot change data.
        request = Request(self.url + "/api/undo", data=b"{}", method="POST")
        with self.assertRaises(HTTPError) as caught:
            urlopen(request)
        self.assertEqual(caught.exception.code, 403)
        self.assertEqual(self.get("/api/state")["counts"]["classified"], 1)
        # Stale background reads are cancelled before opening a source image.
        self.server.app.images.view("test", "new", "1")
        with self.assertRaises(HTTPError) as caught:
            self.get("/api/image?id=" + str(item["id"]) + "&kind=preview&c=test&v=old&f=1")
        self.assertEqual(caught.exception.code, 409)

    def test_recent_order_and_path_mapping(self):
        from app import PathMapper, business_date
        mapper = PathMapper(self.base / "nas")
        client = r"\\stdeer\home\本地分析图片\cases"
        server = str(self.base / "nas" / "本地分析图片" / "cases")
        self.assertEqual(mapper.server(client), server)
        self.assertEqual(mapper.client(server), client)
        self.assertEqual(business_date({"after_sales": {"reported_at": "2026-10-15T01:00:00Z"}}),
                         ("2026-10-15T09:00:00+08:00", "reported_at"))
        store = self.server.app.store
        ids = [r["id"] for r in self.get("/api/images?limit=2")["items"]]
        try:
            with store.lock, store.db:
                store.db.execute("UPDATE images SET reported_at=? WHERE id=?", ("2026-10-16T10:00:00+08:00", ids[1]))
            listed = self.get("/api/images?limit=2")
            self.assertEqual(listed["items"][0]["id"], ids[1])
            self.assertEqual(self.get("/api/images?locate=" + str(ids[0]))["position"], 1)
        finally:
            with store.lock, store.db:
                store.db.execute("UPDATE images SET reported_at='2026-10-03T10:00:00+08:00'")

    def test_old_tab_handoff_writes_remote_store(self):
        from deployment.handoff import make_proxy
        proxy = make_proxy(self.url, "old-page-token", port=0)
        thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        thread.start()
        try:
            url = "http://127.0.0.1:" + str(proxy.server_port)
            first = self.get("/api/images?limit=1")["items"][0]["id"]
            label = {"defects": ["崩边"], "responsibilities": ["运输"], "image_state": "defect", "note": "handoff"}
            request = Request(url + "/api/commit", data=dumps({"id": first, "label": label}).encode(),
                              headers={"X-Local-Token": "old-page-token", "Content-Type": "application/json"},
                              method="POST")
            with urlopen(request, timeout=5) as response:
                self.assertEqual(json.load(response)["saved"], 1)
            self.assertEqual(self.get("/api/state")["counts"]["classified"], 1)
            self.assertEqual(self.get("/api/item?id=" + str(first))["label"]["note"], "handoff")
        finally:
            proxy.shutdown()
            proxy.server_close()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.server.app.store.db.close()
        cls.temp.cleanup()


if __name__ == "__main__":
    unittest.main(verbosity=2)
