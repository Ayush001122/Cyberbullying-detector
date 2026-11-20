"""Run with:  python -m unittest discover tests -v"""
import unittest

from app import app


class ApiTests(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        self.c = app.test_client()

    def test_health(self):
        self.assertEqual(self.c.get("/api/health").json["status"], "ok")

    def test_detects_bullying(self):
        r = self.c.post("/api/predict", json={"text": "you are a pathetic loser, nobody likes you"}).json
        self.assertTrue(r["is_toxic"])
        self.assertNotEqual(r["severity"], "none")

    def test_clean_text(self):
        r = self.c.post("/api/predict", json={"text": "Thanks for helping me with my homework"}).json
        self.assertFalse(r["is_toxic"])

    def test_validation(self):
        self.assertEqual(self.c.post("/api/predict", json={"text": "  "}).status_code, 400)

    def test_batch(self):
        r = self.c.post("/api/predict/batch", json={"texts": ["shut up you idiot", "have a great weekend"]}).json
        self.assertEqual(r["summary"]["total"], 2)
        self.assertEqual(r["summary"]["flagged"], 1)

    def test_metrics_and_history(self):
        self.assertIn("results", self.c.get("/api/metrics").json)
        self.assertIn("stats", self.c.get("/api/history").json)


if __name__ == "__main__":
    unittest.main()
