import os
import unittest
from unittest.mock import patch

# Imports need configuration, but these tests never connect to any service.
with patch.dict(os.environ, {
    "TG_API_ID": "1", "TG_API_HASH": "test", "BOT_TOKEN": "1:test",
    "TARGET_CHANNEL_ID": "@test", "GEMINI_API_KEY": "test",
    "SOURCE_CHANNELS": "test",
}):
    from app.services.summarizer import NewsSummarizer
    from app.services.quality_audit import QualityAuditor

from app.services.text_quality import auto_news_quality_issues


GOOD_TEXT = (
    "🇺🇦 <b>У Києві відкрили нову лікарню</b>\n\n"
    "Заклад прийматиме пацієнтів із понеділка. У ньому працюватимуть 50 лікарів."
)
BAD_TEXT = "💥 <b>Жах</b>\n\nБіля магазину зафіксували падіння безпілотника."


class SummarizerQualityTests(unittest.TestCase):
    def setUp(self):
        # Avoid constructing an API client; exercise real offline pipeline methods.
        self.summarizer = NewsSummarizer.__new__(NewsSummarizer)
        self.summarizer.last_fact_check_stats = {}
        self.posts = [{
            "text": "У Києві відкрили нову лікарню. Заклад має 50 місць.",
            "channel_username": "suspilnenews",
        }]
        self.event = {
            "event_id": "hospital", "source_ids": [0], "best_source_id": 0,
            "headline_hint": "У Києві відкрили нову лікарню",
            "summary": "Заклад прийматиме пацієнтів із понеділка.",
            "key_facts": ["У ньому працюватимуть 50 лікарів."],
            "category": "society", "digest_role": "core",
        }

    def test_clickbait_fallback_is_skipped(self):
        self.event["headline_hint"] = "Жах"
        self.assertIsNone(self.summarizer._build_fallback_news_item(self.event, self.posts))

    def test_fallback_deduplicates_summary_and_facts(self):
        sentence = "Заклад прийматиме пацієнтів із понеділка."
        self.event["summary"] = f"{sentence} У ньому працюватимуть 50 лікарів."
        self.event["key_facts"] = [sentence, "У ньому працюватимуть 50 лікарів."]
        item = self.summarizer._build_fallback_news_item(self.event, self.posts)
        self.assertIsNotNone(item)
        self.assertEqual(item["text"].count(sentence), 1)
        self.assertEqual(auto_news_quality_issues(item["text"]), [])

    def test_manual_fallback_remains_available(self):
        self.event.update(headline_hint="Жах", is_priority=True)
        item = self.summarizer._build_fallback_news_item(self.event, self.posts)
        self.assertIsNotNone(item)
        self.assertTrue(item["is_priority"])

    def test_editor_clickbait_is_rejected(self):
        self.assertEqual(self.summarizer._validate_final_news([
            {"event_id": "hospital", "source_id": 0, "text": BAD_TEXT}
        ], [self.event], self.posts, 1), [])

    def test_editor_specific_news_is_preserved(self):
        items = self.summarizer._validate_final_news([
            {"event_id": "hospital", "source_id": 0, "text": GOOD_TEXT}
        ], [self.event], self.posts, 1)
        self.assertEqual([item["text"] for item in items], [GOOD_TEXT])

    def test_fill_uses_next_candidate_after_weak_event(self):
        weak = {**self.event, "event_id": "weak", "headline_hint": "Жах"}
        items = self.summarizer._fill_missing_news([], [weak, self.event], self.posts, 1)
        self.assertEqual([item["event_id"] for item in items], ["hospital"])

    def test_final_gate_blocks_bad_fact_check_rewrite(self):
        self.summarizer.last_fact_check_stats = {
            "expected": 2, "eligible_cases": 2, "checked": 2, "completed": True,
        }
        items = self.summarizer._filter_unusable_auto_news([
            {"event_id": "bad", "text": BAD_TEXT},
            {"event_id": "good", "text": GOOD_TEXT},
        ])
        self.assertEqual([item["event_id"] for item in items], ["good"])
        stats = self.summarizer.last_fact_check_stats
        self.assertEqual(stats["expected"], 1)
        self.assertEqual(stats["eligible_cases"], 1)
        self.assertEqual(stats["checked"], 1)
        self.assertEqual(stats["auto_text_blocked"], 1)

    def test_final_gate_preserves_incomplete_fact_check_warning(self):
        self.summarizer.last_fact_check_stats = {
            "expected": 2, "eligible_cases": 2, "checked": 0, "completed": False,
        }
        self.summarizer._filter_unusable_auto_news([
            {"event_id": "bad", "text": BAD_TEXT},
            {"event_id": "good", "text": GOOD_TEXT},
        ])
        self.assertEqual(self.summarizer.last_fact_check_stats["checked"], 0)
        self.assertFalse(self.summarizer.last_fact_check_stats["completed"])

    def test_final_gate_preserves_manual_news(self):
        for marker in ({"is_priority": True}, {"manual_fact_source_ids": [0]}):
            with self.subTest(marker=marker):
                item = {"event_id": "manual", "text": BAD_TEXT, **marker}
                self.assertEqual(self.summarizer._filter_unusable_auto_news([item]), [item])

    def test_audit_reports_bad_auto_text(self):
        issues = QualityAuditor._check_text_quality([
            {"event_id": "bad", "text": BAD_TEXT},
            {"event_id": "good", "text": GOOD_TEXT},
        ])
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["event_id"], "bad")
        self.assertEqual(issues[0]["type"], "text_quality_issue")

    def test_audit_exempts_all_manual_markers(self):
        for marker in (
            {"is_priority": True}, {"manual_fact_locked": True},
            {"manual_fact_source_ids": [0]}, {"_audit_manual_ids": [123]},
        ):
            with self.subTest(marker=marker):
                self.assertEqual(QualityAuditor._check_text_quality([
                    {"event_id": "manual", "text": BAD_TEXT, **marker}
                ]), [])


if __name__ == "__main__":
    unittest.main()
