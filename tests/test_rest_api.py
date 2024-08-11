import unittest
from my_rag_package.rest_api import create_app

class TestRestAPI(unittest.TestCase):
    def setUp(self):
        self.app = create_app().test_client()
        self.app.testing = True

    def test_search_endpoint(self):
        response = self.app.post('/search', json={'query': 'test query'})
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.json)

if __name__ == '__main__':
    unittest.main()
