import numpy as np
import cv2
from sklearn.metrics import confusion_matrix, roc_auc_score

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred).ravel()

    tn, fp, fn, tp = cm

    # 计算敏感性和特异性
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    return sensitivity, specificity, accuracy

import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

def calculate_dice(true_mask, pred_mask):
    intersection = np.sum(true_mask * pred_mask)
    return (2. * intersection) / (np.sum(true_mask) + np.sum(pred_mask) + 1e-6)  # 加上小常数以避免除零错误

def evaluate_model(dataset, model, batch_size=16):
    all_sensitivities = []
    all_specificities = []
    all_accuracies = []
    all_aucs = []
    all_dices = []  # 用于存储 DICE 系数

    num_samples = len(dataset)

    # 处理每个批次
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_images = []
        batch_true_masks = []

        # 收集当前批次的图像和真实掩膜
        for idx in range(start_idx, end_idx):
            image, true_mask = dataset[idx]
            batch_images.append(image)
            _, true_mask = cv2.threshold(true_mask, 127, 1, cv2.THRESH_BINARY)  # 127 是阈值
            batch_true_masks.append(true_mask)

        # 将批次转换为张量
        batch_images_np = np.array(batch_images)
        batch_true_masks_np = np.array(batch_true_masks)

        # 使用模型进行批量预测
        pred_masks = model.predict(batch_images_np)
        # 遍历当前批次的每个样本
        for i in range(len(batch_true_masks)):
            true_mask_np = batch_true_masks_np[i].flatten()
            pred_mask_np = pred_masks[i].flatten()

            # 确保两个数组的长度一致
            if true_mask_np.shape[0] != pred_mask_np.shape[0]:
                print(f"Warning: Shape mismatch for sample {start_idx + i}. Skipping this sample.")
                continue

            # 计算性能指标
            sensitivity, specificity, accuracy = calculate_metrics(true_mask_np, pred_mask_np)

            # 计算 AUC
            auc = roc_auc_score(true_mask_np, pred_mask_np)

            # 计算 DICE 系数
            dice = calculate_dice(true_mask_np, pred_mask_np)

            all_sensitivities.append(sensitivity)
            all_specificities.append(specificity)
            all_accuracies.append(accuracy)
            all_aucs.append(auc)
            all_dices.append(dice)  # 添加 DICE 系数到列表

    # 计算平均指标
    avg_sensitivity = np.mean(all_sensitivities)
    avg_specificity = np.mean(all_specificities)
    avg_accuracy = np.mean(all_accuracies)
    avg_auc = np.mean(all_aucs)
    avg_dice = np.mean(all_dices)  # 计算平均 DICE 系数

    print(f"Average DICE: {avg_dice:.4f}")  # 输出平均 DICE 系数
    print(f"Average Sensitivity: {avg_sensitivity:.4f}")
    print(f"Average Specificity: {avg_specificity:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average AUC: {avg_auc:.4f}")
    
