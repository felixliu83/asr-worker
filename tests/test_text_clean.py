import unittest

from modules.text_clean import normalize_punct


class NormalizePunctTestCase(unittest.TestCase):
    def test_cjk_sentence_uses_chinese_punct(self):
        self.assertEqual(normalize_punct("这是中文.", "zh"), "这是中文。")

    def test_latin_segment_keeps_ascii_period(self):
        self.assertEqual(normalize_punct("This is english.", "zh"), "This is english.")

    def test_mixed_sentence_keeps_latin_punct(self):
        self.assertEqual(normalize_punct("测试 test.", "zh"), "测试 test.")

    def test_cjk_punct_with_space(self):
        self.assertEqual(normalize_punct("你好 !", "zh"), "你好 ！")

    def test_leading_punct_followed_by_cjk(self):
        self.assertEqual(normalize_punct("?你好", "zh"), "？你好")

    def test_semicolon_conversion_for_cjk(self):
        self.assertEqual(normalize_punct("你好;", "zh"), "你好；")

    def test_semicolon_kept_for_latin(self):
        self.assertEqual(normalize_punct("Example; text", "zh"), "Example; text")


if __name__ == "__main__":
    unittest.main()
