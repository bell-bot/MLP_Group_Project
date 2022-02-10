import torch
from torch.nn import L1Loss, MSELoss
from enum import Enum

def mae_loss(x, y):
    loss = L1Loss()
    return torch.reshape(loss(x,y), shape=(1,1))

def rmse_loss(x,y):
    loss = MSELoss()
    return torch.reshape(torch.sqrt(loss(x,y)), shape=(1,1))

class LossType(Enum):
    MAE = "mae"
    RMSE = "rmse"


def compare_window(keyword, window, loss_type=LossType.MAE):

    if loss_type == LossType.MAE:
        return mae_loss(keyword, window)

    elif loss_type == LossType.RMSE:
        return rmse_loss(keyword, window)

def match_audio(keyword, sliding_windows, loss_type=LossType.MAE):
    results = []
    for i in range(len(sliding_windows)):
        t = sliding_windows[i]
        loss = compare_window(keyword, t, loss_type=loss_type)
        results.append(loss)
    return torch.cat(results, dim=0)