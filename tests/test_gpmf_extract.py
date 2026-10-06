import tempfile
import unittest
from pathlib import Path

from recording.telemetry import extract_gopro_telemetry


class GpmfExtractTests(unittest.TestCase):
    def test_extract_sample_mp4_to_jsonl(self):
        sample = Path("vendor/gopro/gpmf-parser/samples/hero5.mp4")
        if not sample.exists():
            self.skipTest("gpmf-parser samples not available (submodule not initialized)")

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            result = extract_gopro_telemetry(sample, out_dir, overwrite=True)
            self.assertTrue(result.ok, msg=result.error)
            jsonl_path = out_dir / f"{sample.stem}.jsonl"
            self.assertTrue(jsonl_path.exists())
            lines = [line for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertGreater(len(lines), 0)


if __name__ == "__main__":
    unittest.main()

