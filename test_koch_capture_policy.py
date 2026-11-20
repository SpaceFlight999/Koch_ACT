import json
import tempfile
import unittest
from pathlib import Path

from koch_capture_policy import check_capture_dates

WORKSPACE = Path(__file__).resolve().parent


class CapturePolicyTests(unittest.TestCase):
    def test_mixed_pack_cannot_use_old_demo(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as directory:
            root = Path(directory)
            old = root / 'old'
            new = root / 'new'
            for source, created in ((old, '2026-11-01T15:59:59+00:00'),
                                    (new, '2026-11-01T16:00:00+00:00')):
                source.mkdir()
                (source / 'episode.json').write_text(json.dumps({'created_at': created}), encoding='utf-8')
            check_capture_dates([{'source': str(new)}])
            with self.assertRaisesRegex(ValueError, '2026-11-01.*已屏蔽'):
                check_capture_dates([{'source': str(new)}, {'source': str(old)}])

    def test_missing_metadata_cannot_whitelist_model(self):
        with tempfile.TemporaryDirectory(dir=WORKSPACE) as directory:
            with self.assertRaisesRegex(ValueError, '找不到采集元数据'):
                check_capture_dates([{'source': str(Path(directory) / '20261104_000000')}])
            with self.assertRaisesRegex(ValueError, '已屏蔽'):
                check_capture_dates([{'source': str(Path(directory) / '20261101_000000')}])


if __name__ == '__main__':
    unittest.main()
