import unittest

from sys_utils.masking import segment, shift, digit_at


class MaskingTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_alphabet_segmentation_shall_return_segments(self):
        segments: list[str] = segment('abcdefghijklmnopqrstuvwxyz', 5)
        self.assertIsNotNone(segments, 'segments is null')
        expected = ['abcde', 'fghij', 'klmno', 'pqrst', 'uvwxy', 'z']
        self.assertEqual(segments, expected, 'segments is not equal to expected')

    def test_when_alphabet_segmentation_with_empty_alphabet_shall_rais_error(self):
        with self.assertRaises(ValueError):
            segment(None, 3)

    def test_when_shift_text_then_text_is_shifted(self):
        self.assertEqual('cdeab', shift('abcde', 3))
        self.assertEqual('deabc', shift('abcde', 2))
        self.assertEqual('abcde', shift('abcde', 0))
        self.assertEqual('bcdea', shift('abcde', 4))
        self.assertEqual('abcde', shift('abcde', 5))
        self.assertEqual('eabcd', shift('abcde', 6))

    def test_when_extract_digit_from_int_then_return_expected(self):
        self.assertEqual(3, digit_at(359, 2))
        self.assertEqual(3, digit_at(230, 1))
        self.assertEqual(0, digit_at(100, 0))
        self.assertEqual(9, digit_at(109, 0))
        with self.assertRaises(IndexError):
            digit_at(109, 4)
