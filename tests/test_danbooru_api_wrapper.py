import unittest
from unittest.mock import patch, MagicMock
from danbooru_api_wrapper import DanbooruApi
from parameterized import parameterized

# Mock Data
MOCK_TAGS_RENAME_MAP = {
    "api_spam_eggs": "spam_eggs",
    "api_hello_world": "hello_world",
    "api_foo_bar": "foo_bar",
    "api_test_tag": "test_tag",
    "api_sample_tag": "sample_tag"
}
MOCK_TAGS_RENAME_MAP_REVERSE = {v: k for k, v in MOCK_TAGS_RENAME_MAP.items()}

class TestDanbooruApi(unittest.TestCase):

    def setUp(self):
        self.api = DanbooruApi()
        self.api.reverse_tags_renaming = MOCK_TAGS_RENAME_MAP_REVERSE

    @parameterized.expand([
        ("sample tag", "api_sample_tag"),
        ("random tag", "random_tag"),
        ("test tag", "api_test_tag"),
        ("hello world", "api_hello_world"),
        ("spam eggs", "api_spam_eggs")
    ])
    def test_convert_tag_for_api(self, input_tag, expected_tag):
        self.assertEqual(self.api.convert_tag_for_api(input_tag), expected_tag)

    @parameterized.expand([
        ("h4. Header\n{{link}}\n[[link2]]\nTest\r\nNewLine", " Header\n<b>link</b>\n<b>link2</b>\nTest<br>NewLine"),
        ("h4. Heading\n{{formatted}} text", " Heading\n<b>formatted</b> text"),
        ("Plain [[text]] with no header", "Plain <b>text</b> with no header"),
        ("h4. Heading\n\nContent", " Heading\n\nContent"),
        ("{{Link}} only", "<b>Link</b> only")
    ])
    def test_format_wiki_text(self, input_text, expected_text):
        self.assertEqual(DanbooruApi.format_wiki_text(input_text), expected_text)


    @parameterized.expand([
        ("Information\nSee Also\nOther Data", "Information"),
        ("Information\nRelated tags\nTags Info", "Information"),
        ("All data, no sections", "All data, no sections"),
        ("Before\nSee Also", "Before"),
        ("Data\nRelated tags", "Data")
    ])
    def test_strip_sections(self, input_info, expected_info):
        self.assertEqual(self.api.strip_sections(input_info), expected_info)

if __name__ == "__main__":
    unittest.main()
