import argparse
import torch
import yaml
import os

import torch.nn as nn
import torch.optim as optim

from dataset import MedicalDataset
from models.CNN import UNet
from torch.utils.data import DataLoader, random_split
from evaluate import evaluate_model

def train_model(model, dataset, num_epochs=10, batch_size=16, learning_rate=0.001, device='cpu', savename="Data"):
    # 将模型移动到指定的设备
    model.to(device)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.BCEWithLogitsLoss()  # 或者使用其他损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        for images, masks in dataloader:

            # 将 NumPy 数组转换为 PyTorch 张量，并确保数据类型为 float32
            images = images.to(device).float()
            masks = masks.to(device).float()
            masks = masks / 255

            # 调整图像和掩膜的形状
            images = images.permute(0, 3, 1, 2)  # 从 (batch_size, height, width, channels) 转换为 (batch_size, channels, height, width)
            masks = masks.unsqueeze(1)  # 确保掩膜的形状为 (batch_size, 1, height, width)
            

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    model_path = 'pretrain/UNet_' + savename + '.pth'

    if os.path.exists('pretrain') == False:
        os.mkdir('pretrain')
    torch.save(model.state_dict(), model_path)  # 保存模型的状态字典
    print("Model Save at", model_path)

def train(args, cfg):
    dataset = MedicalDataset(args.dataset, 256)

    # 划分训练集和测试集
    train_size = int(0.8 * len(dataset))  # 80% 用于训练
    test_size = len(dataset) - train_size  # 剩余 20% 用于测试
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    model = UNet(in_channels=3, out_channels=1)
    train_model(model, train_dataset, num_epochs=10, batch_size=16, learning_rate=0.001, device=device, savename=dataset.data_name)

    print("Evaluating...")
    evaluate_model(test_dataset, model, int(args.eval_batch_size))

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m", "--model", default="UNet", help="model name"
    )

    parser.add_argument(
        "-c", "--config_path", default="config/default_config.yaml", help="the path of configuration"
    )

    parser.add_argument(
        "-d", "--dataset", default="Data", help="Name of Dataset"
    )

    parser.add_argument(
        "-b", "--eval_batch_size", default="16", help="Batch size of image processing"
    )

    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        cfg = yaml.safe_load(file)

    
    train(args, cfg)