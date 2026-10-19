import unittest

from app.services.text_quality import (
    auto_news_quality_issues,
    deduplicate_fallback_sentences,
    weak_auto_headline,
)


class TextQualityTests(unittest.TestCase):
    def test_clickbait_titles_are_rejected(self):
        for title in ("Жах", "💥 Жах!", "Шок", "Увага", "Важлива подія", "Оновлено: Жах"):
            with self.subTest(title=title):
                self.assertTrue(weak_auto_headline(title))

    def test_specific_short_titles_are_allowed(self):
        for title in ("Помер актор", "Ухвалено закон", "У Києві відкрили нову лікарню"):
            with self.subTest(title=title):
                self.assertFalse(weak_auto_headline(title))

    def test_empty_or_emoji_title_is_rejected(self):
        for title in ("", "💥", "<b> </b>"):
            with self.subTest(title=title):
                self.assertTrue(weak_auto_headline(title))

    def test_valid_article_is_allowed(self):
        self.assertEqual(auto_news_quality_issues(
            "🇺🇦 <b>У Києві відкрили нову лікарню</b>\n\n"
            "Заклад прийматиме пацієнтів із понеділка. У ньому працюватимуть 50 лікарів."
        ), [])

    def test_bold_headline_is_required(self):
        self.assertIn("missing_bold_headline", auto_news_quality_issues("Жах\n\nТекст."))

    def test_clickbait_article_is_rejected(self):
        self.assertIn("vague_or_clickbait_headline", auto_news_quality_issues(
            "💥 <b>Жах</b>\n\nБіля магазину зафіксували падіння безпілотника."
        ))

    def test_missing_body_is_rejected(self):
        self.assertIn("empty_body", auto_news_quality_issues(
            "<b>У Києві відкрили нову лікарню</b>\n\n"
        ))

    def test_repeated_sentences_are_rejected(self):
        sentence = "Біля магазину зафіксували падіння безпілотника."
        self.assertIn("repeated_sentences", auto_news_quality_issues(
            f"<b>У місті впав безпілотник</b>\n\n{sentence} {sentence}"
        ))

    def test_repetition_across_html_paragraphs_is_rejected(self):
        sentence = "У лікарні працюватимуть п'ятдесят досвідчених лікарів."
        self.assertIn("repeated_sentences", auto_news_quality_issues(
            f"<b>У Києві відкрили лікарню</b>\n{sentence}\n<i>{sentence}</i>"
        ))

    def test_near_repeats_are_removed_from_fallback(self):
        sentence = "У Києві відкрили нову міську лікарню для пацієнтів."
        self.assertEqual(deduplicate_fallback_sentences([
            sentence, sentence.replace(".", "!"), "Заклад має 50 місць."
        ]), [sentence, "Заклад має 50 місць."])

    def test_distinct_numbers_are_preserved(self):
        self.assertEqual(deduplicate_fallback_sentences([
            "Працюють 50 лікарів.", "Доступні 20 ліжок."
        ]), ["Працюють 50 лікарів.", "Доступні 20 ліжок."])

    def test_caption_boilerplate_is_removed(self):
        self.assertEqual(deduplicate_fallback_sentences([
            "Дивіться відео.", "Подробиці уточнюються.",
            "Прямі кадри конкретного удару або його безпосередніх наслідків.",
            "У Києві відкрили лікарню."
        ]), ["У Києві відкрили лікарню."])


if __name__ == "__main__":
    unittest.main()
