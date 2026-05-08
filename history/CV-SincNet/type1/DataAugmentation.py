import torch
import numpy as np

class RFFIAugmentor:
    """
    针对 I/Q 信号的数据增强器。
    包含：随机相位旋转、高斯白噪声、幅度缩放。
    """
    def __init__(self, 
                 use_phase_rotate=True, 
                 use_noise=True, 
                 use_amp_scale=True,
                 noise_std=0.005,      # 噪声标准差 (根据信号强度调整)
                 scale_range=0.1):     # 幅度缩放范围 (+-10%)
        
        self.use_phase_rotate = use_phase_rotate
        self.use_noise = use_noise
        self.use_amp_scale = use_amp_scale
        self.noise_std = noise_std
        self.scale_range = scale_range

    def random_phase_rotation(self, x):
        """
        随机相位旋转 (Random Phase Rotation)
        模拟收发端非同步导致的相位偏差。
        x shape: (Batch, 2, Length)
        """
        batch_size = x.shape[0]
        device = x.device

        # 生成随机角度 theta: [0, 2*pi]
        theta = torch.rand(batch_size, device=device) * 2 * np.pi
        
        # 构建旋转矩阵元素
        cos_theta = torch.cos(theta).view(batch_size, 1, 1)
        sin_theta = torch.sin(theta).view(batch_size, 1, 1)

        # I/Q 分离
        I = x[:, 0:1, :]
        Q = x[:, 1:2, :]

        # 旋转公式:
        # I' = I*cos - Q*sin
        # Q' = I*sin + Q*cos
        I_new = I * cos_theta - Q * sin_theta
        Q_new = I * sin_theta + Q * cos_theta

        return torch.cat([I_new, Q_new], dim=1)

    def add_awgn(self, x):
        """
        添加高斯白噪声 (AWGN)
        模拟低信噪比环境。
        """
        noise = torch.randn_like(x) * self.noise_std
        return x + noise

    def random_amplitude_scale(self, x):
        """
        随机幅度缩放
        模拟接收增益的不稳定性。
        """
        batch_size = x.shape[0]
        # 生成缩放因子: 1.0 +/- scale_range
        scale = (torch.rand(batch_size, 1, 1, device=x.device) * 2 - 1) * self.scale_range + 1.0
        return x * scale

    def __call__(self, x):
        """
        应用所有启用的增强
        """
        # 1. 相位旋转 (RFFI 最重要的增强)
        if self.use_phase_rotate:
            x = self.random_phase_rotation(x)
        
        # 2. 幅度缩放
        if self.use_amp_scale:
            x = self.random_amplitude_scale(x)
            
        # 3. 加噪声 (通常放在最后)
        if self.use_noise:
            x = self.add_awgn(x)
            
        return x

if __name__ == "__main__":
    # 简单测试代码
    dummy_input = torch.randn(10, 2, 1024)
    aug = RFFIAugmentor()
    out = aug(dummy_input)
    print(f"Input shape: {dummy_input.shape}, Output shape: {out.shape}")
    print("DataAugmentation module works successfully.")