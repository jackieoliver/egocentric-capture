import unittest

from gopro.labs import build_profile_labs_command


class LabsCommandTests(unittest.TestCase):
    def test_shutter_lock_included(self):
        profile = {
            "fps": 120,
            "shutter_target": {"preferred": "1/240 or 1/360"},
        }
        cmd, labels = build_profile_labs_command(profile, profile_name="wrist_right")
        self.assertIn("tS180", cmd)
        self.assertIn("shutter:S180", labels)


if __name__ == "__main__":
    unittest.main()
