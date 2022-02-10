import inspect
import sys
import unittest
import os
import torch
import math

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from compare import LossType, mae_loss, rmse_loss, match_audio

class Compare_Test(unittest.TestCase):

    def test_mae_simple(self):
        x = torch.tensor([1], dtype=torch.float32)
        y = torch.tensor([2], dtype=torch.float32)
        expected = torch.abs(torch.sub(x,y))
        actual = mae_loss(x,y)
        self.assertEqual(expected, actual)

    def test_mae_complex(self):
        x = torch.tensor([1,2,3], dtype=torch.float32)
        y = torch.tensor([3,2,1], dtype=torch.float32)
        expected = (1/3)*torch.sum(torch.abs(torch.sub(x,y)))
        actual = mae_loss(x,y)
        self.assertEqual(expected, actual)

    def test_rmse_simple(self):
        x = torch.tensor([1], dtype=torch.float32)
        y = torch.tensor([2], dtype=torch.float32)
        expected = torch.sqrt(torch.sum(torch.pow(torch.sub(x,y), 2))/1)
        actual = rmse_loss(x,y)
        self.assertEqual(expected, actual)

    def test_rmse_complex(self):
        x = torch.tensor([1,2,3], dtype=torch.float32)
        y = torch.tensor([3,2,1], dtype=torch.float32)
        expected = torch.sqrt(torch.sum(torch.pow(torch.sub(x,y), 2))/3)
        actual = rmse_loss(x,y)
        self.assertEqual(expected, actual)

    def test_mae_sliding_windows(self):
        keyword = torch.tensor([3,4,2], dtype=torch.float32)
        windows = torch.tensor([[1,1,1], [2,2,2], [5,2,6]], dtype=torch.float32)
        expected = torch.tensor([[2], [1], [(8/3)]], dtype=torch.float32)
        actual = match_audio(keyword, windows)
        self.assertEqual(torch.sum(torch.sub(expected, actual)),torch.tensor([0], dtype=torch.float32))

    def test_rmse_sliding_windows(self):
        keyword = torch.tensor([3,4,2], dtype=torch.float32)
        windows = torch.tensor([[1,1,1], [2,2,2], [5,2,6]], dtype=torch.float32)
        expected = torch.tensor([[math.sqrt(14/3)], [math.sqrt(5/3)], [math.sqrt(24/3)]], dtype=torch.float32)
        actual = match_audio(keyword, windows, loss_type=LossType.RMSE)
        self.assertEqual(torch.sum(torch.sub(expected, actual)),torch.tensor([0], dtype=torch.float32))

if __name__== "__main__":
    unittest.main()