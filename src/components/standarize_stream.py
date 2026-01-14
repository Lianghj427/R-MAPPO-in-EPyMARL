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

    def update(self, arr, mask=None):
        """
        arr: [Batch, Time, Agent, 1] 或 [Batch*Time*Agent, 1]
        mask: [Batch, Time, Agent, 1] 或 [Batch*Time*Agent, 1]，也就是 alive_mask 或 filled_mask
        """
        # 1. 统一转为 Tensor
        if not isinstance(arr, torch.Tensor):
            arr = torch.tensor(arr, dtype=torch.float32, device=self.device)
            
        # 2. 展平数据 [N, Features]
        batch = arr.reshape(-1, arr.shape[-1])
        
        # 3. [关键修改] 应用 Mask 筛选
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
            
            # 展平 Mask: [N, 1] -> [N]
            mask = mask.reshape(-1)
            
            # 只保留 Mask > 0.5 (即为 1) 的有效数据
            # bool indexing: batch[indices] 会选出有效行
            batch = batch[mask > 0.5]
            
        # 4. 如果筛选后没数据了（极少见），直接返回
        if batch.shape[0] < 1:
            return

        # 5. 计算当前有效 Batch 的统计量
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
            
        return (batch - self.mean) / torch.sqrt(self.var + self.epsilon)

    def denormalize(self, batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.float32, device=self.device)

        return batch * torch.sqrt(self.var + self.epsilon) + self.mean