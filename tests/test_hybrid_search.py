import unittest
from my_rag_package.hybrid_search import HybridSearch

class TestHybridSearch(unittest.TestCase):
    def test_search(self):
        search = HybridSearch()
        result = search.search("test query")
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
