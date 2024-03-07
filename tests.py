import unittest

from memory.semantic.learn import parse_triplet_string

class TestParseTripletString(unittest.TestCase):

    def test_parse_triplet_string(self):
        triplets_string = "(aa aaa, bbb bb, ccc cc), (ddd ddd, eee ee, fff ff), (ggg ggg, hhh hh, iii ii)"
        expected_result = [
            ('aa aaa', 'bbb bb', 'ccc cc'),
            ('ddd ddd', 'eee ee', 'fff ff'),
            ('ggg ggg', 'hhh hh', 'iii ii')
        ]

        result = parse_triplet_string(triplets_string)
        self.assertEqual(result, expected_result)

if __name__ == '__main__':
    unittest.main()
