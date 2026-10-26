import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile


spec = importlib.util.spec_from_file_location("package_source", Path(__file__).parents[1] / "tools/package-source.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


class SourceReleaseTests(unittest.TestCase):
    def test_source_and_private_file_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "source.zip"
            for name, accepted in [("training/worker.py", True), ("data/customer.json", False),
                                   (".env", False), ("training/weights.pt", False)]:
                with self.subTest(name=name):
                    with zipfile.ZipFile(target, "w") as archive:
                        archive.writestr(name, "print('source')\n")
                    if accepted:
                        self.assertEqual(release.validate_archive(target), 1)
                    else:
                        with self.assertRaises(ValueError):
                            release.validate_archive(target)

    def test_history_keeps_both_previous_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for name in (release.PACKAGE, release.RECORD):
                (output / name).write_bytes(b"old")
            candidate, record = output / "new.zip", output / "new.json"
            candidate.write_bytes(b"new zip"); record.write_bytes(b"new json")
            release.publish(output, candidate, record)
            self.assertEqual((output / release.PACKAGE).read_bytes(), b"new zip")
            saved = list((output / "history").glob("*/*"))
            self.assertEqual(len(saved), 2)
            self.assertTrue(all(path.read_bytes() == b"old" for path in saved))

    def test_second_install_failure_restores_old_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for name in (release.PACKAGE, release.RECORD):
                (output / name).write_bytes(b"old " + name.encode())
            candidate, record = output / "new.zip", output / "new.json"
            candidate.write_bytes(b"new zip"); record.write_bytes(b"new json")
            real_replace = release.os.replace
            def fail_record(source, target):
                if source == record:
                    raise OSError("injected replacement failure")
                return real_replace(source, target)
            with patch.object(release.os, "replace", side_effect=fail_record):
                with self.assertRaises(OSError):
                    release.publish(output, candidate, record)
            for name in (release.PACKAGE, release.RECORD):
                self.assertEqual((output / name).read_bytes(), b"old " + name.encode())


if __name__ == "__main__":
    unittest.main()
