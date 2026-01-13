import torch
import torch.nn as nn
from typing import Tuple

class RunningMeanStd(object):
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (), device="cpu"):
        """
        Calculates the running mean and standard deviation of a data stream.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon
        self.epsilon = epsilon
        self.device = device

    def update(self, arr):
        # Ensure input is a tensor on the correct device
        if not isinstance(arr, torch.Tensor):
            arr = torch.tensor(arr, dtype=torch.float32, device=self.device)
        elif arr.device != self.device:
            arr = arr.to(self.device)
        
        # [核心修复]：将数据展平为 [N_Total_Samples, Features]
        # 无论输入是 [Batch, Time, Agent, 1] 还是 [Batch, Features]
        # 我们都只保留最后一个维度作为特征维度，其余维度全部合并
        # 这样算出来的 mean/var 形状就是 (1,) 或者 (Features,)，而不是带有时间维度的形状
        batch = arr.reshape(-1, arr.shape[-1])
            
        batch_mean = torch.mean(batch, dim=0)
        batch_var = torch.var(batch, dim=0)
        batch_count = batch.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count: int):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

    def normalize(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.float32, device=self.device)
        elif batch.device != self.device:
            batch = batch.to(self.device)
            
        return (batch - self.mean) / torch.sqrt(self.var + self.epsilon)

    def denormalize(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.float32, device=self.device)
        elif batch.device != self.device:
            batch = batch.to(self.device)

        return batch * torch.sqrt(self.var + self.epsilon) + self.mean