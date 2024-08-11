import unittest
from my_rag_package.query_transformations import QueryTransformer

class TestQueryTransformer(unittest.TestCase):
    def test_transform(self):
        transformer = QueryTransformer()
        result = transformer.transform("test query")
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
