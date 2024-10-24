import torch
import numpy as np
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 编码器部分
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        # 中间层
        self.bottleneck = self.conv_block(512, 1024)
        
        # 解码器部分
        self.decoder4 = self.upconv_block(1024, 512)
        self.decoder3 = self.upconv_block(1024, 256)
        self.decoder2 = self.upconv_block(512, 128)
        self.decoder1 = self.upconv_block(256, 64)
        
        # 输出层
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)

        # 初始化参数
        self.initialize_weights()

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool2d(2)(enc1))
        enc3 = self.encoder3(nn.MaxPool2d(2)(enc2))
        enc4 = self.encoder4(nn.MaxPool2d(2)(enc3))
        
        # 中间层
        bottleneck = self.bottleneck(nn.MaxPool2d(2)(enc4))
        
        # 解码
        dec4 = self.decoder4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)  # skip connection
        dec3 = self.decoder3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)  # skip connection
        dec2 = self.decoder2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)  # skip connection
        dec1 = self.decoder1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)  # skip connection

        return self.final_conv(dec1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def predict(self, images_np):
        """
        使用训练好的模型进行预测。

        参数:
        - model: 训练好的模型
        - images_np: 输入的 NumPy 数组，形状为 (B, W, H, C)
        - device: 设备类型，'cpu' 或 'cuda'

        返回:
        - predictions_np: 预测结果的 NumPy 数组，形状为 (B, W, H)
        """
        # 将 NumPy 数组转换为 PyTorch 张量，并确保数据类型为 float32
        images = torch.tensor(images_np, dtype=torch.float32).to(self.device)

        # 调整图像的形状
        images = images.permute(0, 3, 1, 2)  # 从 (B, W, H, C) 转换为 (B, C, H, W)

        # 将模型设置为评估模式
        self.eval()

        with torch.no_grad():  # 禁用梯度计算
            outputs = self.forward(images)  # 进行预测

        # 应用 Sigmoid 激活函数
        outputs = torch.sigmoid(outputs)

        # 将输出转换为 NumPy 数组
        predictions_np = outputs.cpu().numpy()  # 转换为 NumPy 数组，移动到 CPU

        # 取出预测结果，假设输出形状为 (B, 1, H, W)
        predictions_np = predictions_np.squeeze(1)  # 从 (B, 1, H, W) 转换为 (B, H, W)
        predictions_np = (predictions_np > 0.5).astype(np.float32)

        return predictions_np
