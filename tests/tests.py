import unittest
from model import loss_function
import numpy as np


class TestModel(unittest.TestCase):

    def setUp(self):
        pass

    def test_loss_function(self):
        y = np.zeros(100)
        y_pred = np.arange(100)
        # SSE = 99*(99+1)*(2*99+1)/6 = 328350
        # RMSE = (float(SSE) / 100) ** .5
        RMSE = 3283.5 ** .5
        self.assertEqual(loss_function(y, y_pred), RMSE)


if __name__ == '__main__':
    unittest.main()
