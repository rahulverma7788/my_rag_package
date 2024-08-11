import unittest
from my_rag_package.reranker import Reranker

class TestReranker(unittest.TestCase):
    def test_rerank(self):
        reranker = Reranker()
        result = reranker.rerank(["result1", "result2"])
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
