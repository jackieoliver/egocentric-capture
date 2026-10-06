import unittest

from gopro.sd import parse_sd_card_status, sd_card_status_from_values, sd_extra_from_health


class SdStatusTests(unittest.TestCase):
    def test_parse_sd_present_ok(self):
        status = {"33": 0, "54": 1024 * 1024}  # 1 GiB in KiB
        sd = parse_sd_card_status(status)
        self.assertTrue(sd.present)
        self.assertEqual(sd.remaining_kib, 1024 * 1024)
        self.assertAlmostEqual(sd.remaining_gib or 0.0, 1.0, places=6)
        self.assertFalse(sd.full)
        self.assertEqual(sd.status, "ok")

    def test_parse_sd_missing_ignores_remaining(self):
        status = {"33": 1, "54": 12345}
        sd = parse_sd_card_status(status)
        self.assertFalse(sd.present)
        self.assertIsNone(sd.remaining_kib)
        self.assertIsNone(sd.remaining_gib)
        self.assertIsNone(sd.full)
        self.assertEqual(sd.status, "missing")

    def test_parse_sd_full(self):
        status = {"33": 0, "54": 0}
        sd = parse_sd_card_status(status)
        self.assertTrue(sd.present)
        self.assertEqual(sd.remaining_kib, 0)
        self.assertEqual(sd.remaining_gib, 0.0)
        self.assertTrue(sd.full)
        self.assertEqual(sd.status, "full")

    def test_sd_from_values_unknown(self):
        sd = sd_card_status_from_values(present=None, remaining_kib=None)
        self.assertIsNone(sd.present)
        self.assertIsNone(sd.remaining_kib)
        self.assertIsNone(sd.full)
        self.assertEqual(sd.status, "unknown")

    def test_sd_extra_from_health_prefers_sd_object(self):
        health = {
            "sd": {
                "status": "ok",
                "present": True,
                "full": False,
                "remaining_kib": 2048,
                "remaining_gib": 0.001953125,
                "remaining_gb": 0.002097152,
            }
        }
        extra = sd_extra_from_health(health)
        self.assertEqual(extra["sd_status"], "ok")
        self.assertTrue(extra["sd_present"])
        self.assertFalse(extra["sd_full"])
        self.assertEqual(extra["sd_remaining_kib"], 2048)
        self.assertAlmostEqual(extra["sd_remaining_gib"], 0.001953125, places=9)
        self.assertAlmostEqual(extra["sd_remaining_gb"], 0.002097152, places=9)

    def test_sd_extra_from_health_falls_back_to_flat_keys(self):
        health = {
            "sd_status": "full",
            "sd_present": True,
            "sd_full": True,
            "sd_space_remaining_kib": 0,
            "sd_space_remaining_gib": 0.0,
            "sd_space_remaining_gb_decimal": 0.0,
        }
        extra = sd_extra_from_health(health)
        self.assertEqual(extra["sd_status"], "full")
        self.assertTrue(extra["sd_present"])
        self.assertTrue(extra["sd_full"])
        self.assertEqual(extra["sd_remaining_kib"], 0)
        self.assertEqual(extra["sd_remaining_gib"], 0.0)
        self.assertEqual(extra["sd_remaining_gb"], 0.0)


if __name__ == "__main__":
    unittest.main()

