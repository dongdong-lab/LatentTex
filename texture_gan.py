import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TextureGenerator(nn.Module):
    def __init__(self, z_dim=16, img_shape=(128, 128)):
        """
        生成器：输入形状为 [batch_size, z_dim, 4, 4]，输出形状为 [batch_size, 3, 128, 128]
        """
        super(TextureGenerator, self).__init__()
        self.z_dim = z_dim
        self.img_shape = img_shape

        # 使用 ConvTranspose2d 层进行上采样
        # 1. 将 [batch_size, z_dim, 4, 4] 转换为 [batch_size, 256, 8, 8]
        self.conv1 = nn.ConvTranspose2d(z_dim, 256, kernel_size=4, stride=2, padding=1)  
        # 2. [batch_size, 256, 8, 8] -> [batch_size, 128, 16, 16]
        self.conv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)   
        # 3. [batch_size, 128, 16, 16] -> [batch_size, 64, 32, 32]
        self.conv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)    
        # 4. [batch_size, 64, 32, 32] -> [batch_size, 3, 64, 64]
        self.conv4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)      
        # 5. [batch_size, 3, 64, 64] -> [batch_size, 3, 128, 128]
        self.conv5 = nn.ConvTranspose2d(3, 3, kernel_size=4, stride=2, padding=1)       

        self.relu = nn.ReLU()

    def forward(self, z):
        # z 的形状为 [batch_size, z_dim, 4, 4]
        x = self.conv1(z)    # 输出: [batch_size, 256, 8, 8]
        x = self.relu(x)

        x = self.conv2(x)    # 输出: [batch_size, 128, 16, 16]
        x = self.relu(x)

        x = self.conv3(x)    # 输出: [batch_size, 64, 32, 32]
        x = self.relu(x)

        x = self.conv4(x)    # 输出: [batch_size, 3, 64, 64]
        x = self.relu(x)

        x = self.conv5(x)    # 输出: [batch_size, 3, 128, 128]
        x = torch.sigmoid(x) # 将输出归一化到 [0, 1]
        return x  # 返回 [batch_size, 3, 128, 128]

class TextureGAN(nn.Module):
    def __init__(self, z_dim=16, img_shape=(128, 128)):
        """
        GAN 模型，默认使用 z_dim=16。如果你希望使用形状 (1, 16, 4, 4) 的 z，
        请确保在实例化时传入 z_dim=16。
        """
        super(TextureGAN, self).__init__()
        self.generator = TextureGenerator(z_dim=z_dim, img_shape=img_shape)

    def forward(self, z):
        return self.generator(z)
