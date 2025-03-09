import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

# μ - 律压扩函数
def mu_law_compression(x, mu=255):
    # 归一化到 [-1, 1]
    min_x = np.min(x)
    max_x = np.max(x)
    if max_x - min_x == 0:  # 避免除零
        return np.zeros_like(x), min_x, max_x
    x = 2 * (x - min_x) / (max_x - min_x) - 1
    x = np.sign(x) * np.log(1 + mu * np.abs(x)) / np.log(1 + mu)
    return x, min_x, max_x

# 逆 μ - 律压扩函数
def inverse_mu_law_compression(x, min_x, max_x, mu=255):
    x = np.sign(x) * ((1 + mu) ** np.abs(x) - 1) / mu
    # 逆归一化
    x = (x + 1) * (max_x - min_x) / 2 + min_x
    return x

# 基于补丁的处理函数
def patchify(data, patch_size=(128, 64), stride=(128, 32)):
    """
    将数据分割成补丁
    :param data: 输入数据，形状为 (128, 2048)
    :param patch_size: 补丁大小，默认为 (128, 64)
    :param stride: 步长，默认为 (128, 32)
    :return: 补丁列表
    """
    patches = []
    patch_info = []
    h, w = data.shape
    patch_h, patch_w = patch_size
    stride_h, stride_w = stride
    for i in range(0, h - patch_h + 1, stride_h):
        for j in range(0, w - patch_w + 1, stride_w):
            patch = data[i:i + patch_h, j:j + patch_w]
            patches.append(patch)
            patch_info.append((i, j))
    return np.array(patches), patch_info

# 自定义数据集类
class CubdlDataset(Dataset):
    def __init__(self, data_dir):
        self.data_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.npy'):
                    self.data_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = self.data_files[idx]
        data = np.load(data_path)  # 加载数据，形状为 [31, 128, 2048]

        # 获取中间角度的索引
        idx = data.shape[0] // 2
        data1 = data[idx]  # 中间角度的数据，形状为 [128, 2048]
        data3 = data[idx - 1:idx + 2]  # 中间三个角度的数据，形状为 [3, 128, 2048]

        # 压扩处理
        data1, min_x1, max_x1 = mu_law_compression(data1)
        data3_compressed = []
        min_x3 = []
        max_x3 = []
        for angle_data in data3:
            compressed_angle_data, min_x, max_x = mu_law_compression(angle_data)
            data3_compressed.append(compressed_angle_data)
            min_x3.append(min_x)
            max_x3.append(max_x)
        data3 = np.array(data3_compressed)

        # 基于补丁的处理
        patches1, patch_info1 = patchify(data1)  # 对中间角度的数据进行补丁分割
        patches3 = []
        patch_info3 = []
        for angle_data in data3:
            patches, info = patchify(angle_data)
            patches3.extend(patches)
            patch_info3.extend(info)
        patches3 = np.array(patches3)

        # 数据转换为tensor
        patches1 = torch.from_numpy(patches1).float()
        patches3 = torch.from_numpy(patches3).float()

        # 提取所需信息
        file_name = os.path.basename(data_path)
        parts = file_name.split('_')[0]
        prefix = parts[:3]
        number = parts[3:]
        return patches1, patches3, (min_x1, max_x1), (np.array(min_x3), np.array(max_x3)), patch_info1, patch_info3, prefix, number


if __name__ == "__main__":
    data_dir = "dataset/TSH"  # 替换为实际的数据目录
    dataset = CubdlDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for patches1, patches3, (min_x1, max_x1), (min_x3, max_x3), patch_info1, patch_info3, prefix, number in dataloader:
        print("Patches from middle angle shape:", patches1.shape)
        print("Patches from three middle angles shape:", patches3.shape)

        # 模拟推理后恢复 data1
        original_shape = (128, 2048)
        restored_data1 = np.zeros(original_shape)
        patch_h, patch_w = (128, 64)
        for patch, (i, j) in zip(patches1[0].cpu().numpy(), patch_info1):
            restored_data1[i:i + patch_h, j:j + patch_w] = patch

        # 逆归一化操作
        min_x1 = min_x1.numpy()
        max_x1 = max_x1.numpy()
        restored_data1 = inverse_mu_law_compression(restored_data1, min_x1, max_x1)
        print("Restored data1", restored_data1)

        # 模拟推理后恢复 data3
        num_angles = 3  # 假设 data3 有 3 个角度的数据
        restored_data3 = np.zeros((num_angles, 128, 2048))
        num_patches_per_angle = len(patches3[0]) // num_angles

        min_x3 = min_x3.numpy()
        max_x3 = max_x3.numpy()
        for angle_idx in range(num_angles):
            start_idx = angle_idx * num_patches_per_angle
            end_idx = (angle_idx + 1) * num_patches_per_angle
            angle_patches = patches3[0][start_idx:end_idx]
            angle_patch_info = patch_info3[start_idx:end_idx]

            angle_original = np.zeros(original_shape)
            for patch, (i, j) in zip(angle_patches.cpu().numpy(), angle_patch_info):
                angle_original[i:i + patch_h, j:j + patch_w] = patch

            # 逆归一化操作，使用每个角度对应的 min_x 和 max_x
            angle_original = inverse_mu_law_compression(angle_original, min_x3[0][angle_idx], max_x3[0][angle_idx])

            restored_data3[angle_idx] = angle_original
        # print("Restored data3:", restored_data3)
        print("Restored data3", restored_data3)

        break