import unittest
from my_rag_package.embeddings import EmbeddingModel

class TestEmbeddingModel(unittest.TestCase):
    def test_embedding(self):
        model = EmbeddingModel()
        result = model.embed("test")
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
