import unittest
from unittest.mock import patch, Mock
from prompt_generation.prompt_generation import *
from enum import Enum

# Sample Enum for testing
class SampleEnum(Enum):
    FIRST = "FirstValue"
    SECOND = "SecondValue"

class EmptyEnum(Enum):
    pass

class TestChooseFrom(unittest.TestCase):

    @patch("random.choice")
    def test_choosing_from_lists(self, mock_choice):
        mock_choice.side_effect = ["chosen_1", "chosen_2"]
        sources = [["a", "b", "c"], ["d", "e", "f"]]
        result = choose_from(*sources)
        self.assertEqual(result, ["chosen_1", "chosen_2"])

    @patch("random.choice")
    @patch("prompt_generation.prompt_generation.random_from_class")
    def test_choosing_from_mixed_sources(self, mock_random_from_class, mock_choice):
        mock_choice.return_value = "chosen_1"
        mock_random_from_class.return_value = "FirstValue"
        sources = [["a", "b", "c"], SampleEnum]
        result = choose_from(*sources)
        self.assertEqual(result, ["chosen_1", "FirstValue"])

    @patch("prompt_generation.prompt_generation.random_from_class")
    def test_choosing_from_enums(self, mock_random_from_class):
        mock_random_from_class.side_effect = ["FirstValue", "SecondValue"]
        sources = [SampleEnum, SampleEnum]
        result = choose_from(*sources)
        self.assertEqual(result, ["FirstValue", "SecondValue"])

    def test_choosing_with_empty_sources(self):
        sources = [[], None, ["a", "b", "c"]]
        result = choose_from(*sources)
        self.assertEqual(len(result), 1)
        self.assertIn(result[0], ["a", "b", "c"])

    def test_extract_value_from_enum(self):
        self.assertEqual(extract_value_if_enum(SampleEnum.FIRST), "FirstValue")

    def test_extract_value_from_non_enum(self):
        self.assertEqual(extract_value_if_enum("SomeString"), "SomeString")

    @patch("random.choices")
    def test_choose_one(self, mock_choices):
        mock_choices.side_effect = [["source1"], ["source2"]]
        sources = [("source1", 0.8), ("source2", 0.2)]
        self.assertEqual(choose_one(*sources), "source1")
        sources = [("source1", 0.2), ("source2", 0.8)]
        self.assertEqual(choose_one(*sources), "source2")

if __name__ == "__main__":
    unittest.main()
