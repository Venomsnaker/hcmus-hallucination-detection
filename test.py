import torch


def check_cuda():
    print(torch.cuda.is_available())