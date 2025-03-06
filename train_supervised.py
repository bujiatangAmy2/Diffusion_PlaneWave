from cubdl.PixelGrid import make_pixel_grid
from cubdl.das_torch import DAS_PW
from datasets.PWDataLoaders import *
from loader_cubdl_rf import *
from loader_duke_img import *
import os
from typing import Dict
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer, GaussianDiffusionSamplerDDIM
from model_unsupervised import UNet
from scheduler import GradualWarmupScheduler
import torch.nn.functional as F

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # # CIFAR10 dataset
    # dataset = CIFAR10(
    #     root='./CIFAR10', train=True, download=True,
    #     transform=transforms.Compose([
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]))
    # dataloader = DataLoader(
    #     dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)


    # # duke_rf dataset
    # dataset = DUKEDataset(csv_file=csv_paths['train'], datapath=datapath)


    # # 设置变换
    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)),  # 根据需求调整尺寸
    #     transforms.ToTensor(),
    # ])

    # folder_path = 'dataset/mark'
    # dataset = DukeImgDataset(folder_path, transform=transform, patch_size=(64, 64), overlap=(8, 8))
    #
    # # 创建DataLoader
    # dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # TSH dataset
    # 数据目录
    data_dir = 'dataset/TSH'
    # 创建数据集实例
    dataset = CubdlDataset(data_dir)
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)


    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"],
                     in_channels=1, out_channels=3).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    l = []
    sub_patch = modelConfig["sub_patch"]

    # start training
    for e in range(modelConfig["epoch"]):
        epoch_loss = 0
        num_batches = 0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            # for images, labels in tqdmDataLoader:  # for CIFAR
            # for sample in tqdmDataLoader:  # for DUKEDataset
            # for das, dtce in tqdmDataLoader:  # for DukeImgDataset
            for data1_patches, data3_patches, (min_x1, max_x1), (min_x3, max_x3), patch_info1, patch_info3, prefix, number in tqdmDataLoader:  # 补丁级数据加载
                # train
                optimizer.zero_grad()
                data1_patches = data1_patches.to(device)
                data3_patches = data3_patches.to(device)
                # # 展平补丁批次维度
                # data1_patches = data1_patches.view(-1, *data1_patches.shape[2:]).to(device)
                # data3_patches = data3_patches.view(-1, *data3_patches.shape[2:]).to(device)
                # print(data1_patches.shape, data3_patches.shape)
                # das = sample["das"].to(device).unsqueeze(1)  # for DUKEDataset
                # tce = sample["dtce"].to(device).unsqueeze(1)
                # das = das.to(device)  # for DukeImgDataset
                # tce = dtce.to(device)
                # x_0 = images.to(device)
                # loss = trainer(x_0).sum() / 1000.

                # loss = trainer(data3_patches).sum() / 1000.
                # loss.backward()

                # 修改前
                # num_sub_batches = (data3_patches.size(1) + sub_patch - 1) // sub_patch

                # 修改后
                num_3ch_patches = data3_patches.size(1) // 3  # 计算3通道补丁总数
                num_sub_batches = (num_3ch_patches + sub_patch - 1) // sub_patch  # 基于3通道补丁分块

                total_loss = 0

                for i in range(num_sub_batches):
                    # 修改前
                    # start = i * sub_patch
                    # end = min(start + sub_patch, data3_patches.size(1))

                    # # supervised
                    # sub_data1_patches = data1_patches[:, start:end, :, :]
                    # sub_data3_patches = data3_patches[:, start * 3:end * 3, :, :]
                    # sub_data3_patches = sub_data3_patches.view(data1_patches.size(0), end - start, 3, 128, 64)

                    # 修改后
                    start_3ch = i * sub_patch
                    end_3ch = min(start_3ch + sub_patch, num_3ch_patches)

                    # 获取对应的单通道索引范围
                    start_ch = start_3ch * 3
                    end_ch = end_3ch * 3

                    # 提取data1和data3的对应补丁
                    sub_data1_patches = data1_patches[:, start_3ch:end_3ch, :, :]  # [B, sub_patch, H, W]
                    sub_data3_patches = data3_patches[:, start_ch:end_ch, :, :]  # [B, sub_patch*3, H, W]

                    # 重塑data3为3通道格式
                    sub_data3_patches = sub_data3_patches.view(
                        data3_patches.size(0),
                        end_3ch - start_3ch,  # 当前子批次的3通道补丁数
                        3,  # 通道数
                        128, 64
                    )

                    t = torch.randint(0, modelConfig["T"], (sub_data1_patches.size(0),), device=device)
                    predicted_3_patches = net_model(sub_data1_patches, t)
                    loss = F.mse_loss(predicted_3_patches, sub_data3_patches)

                    loss.backward()
                    total_loss += loss.item()

                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])

                # add loss
                # epoch_loss += loss.item()
                epoch_loss += total_loss
                num_batches += 1

                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    # "img shape: ": x_0.shape,
                    "img shape: ": data3_patches.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        # 平均损失
        avg_loss = epoch_loss / num_batches
        l.append(avg_loss)

        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(l, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(modelConfig["save_weight_dir"], 'loss_curve.png'))
    plt.show()


def eval(modelConfig: Dict):
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"],
                     attn=modelConfig["attn"], num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(
            os.path.join(modelConfig["save_weight_dir"], modelConfig["test_load_weight"]),
            map_location=device,
            weights_only=True  # 添加此行以消除警告
        )
        model.load_state_dict(ckpt)
        model.eval()
        sampler = GaussianDiffusionSampler(model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

        data_dir = 'dataset/TSH'
        dataset = CubdlDataset(data_dir)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        for data1_patches, data3_patches, (min_x1, max_x1), (min_x3, max_x3), patch_info1, patch_info3, prefix, number in dataloader:
            print(prefix, number)
            print("patch_info3 示例:", patch_info3[:5])
            # 分块推理逻辑（与训练一致）
            B, num_patches, H, W = data1_patches.shape
            sub_patch = modelConfig["sub_patch"]
            num_sub_batches = (num_patches + sub_patch - 1) // sub_patch
            sampled_patches = []

            for i in range(num_sub_batches):
                start = i * sub_patch
                end = min(start + sub_patch, num_patches)
                sub_data1 = data1_patches[:, start:end, :, :].to(device)
                sub_output = sampler(sub_data1)
                sampled_patches.append(sub_output.cpu())

            sampled_patches = torch.cat(sampled_patches, dim=1)

            # 恢复 data3
            restored_data3 = np.zeros((3, 128, 2048))
            # 按角度分离补丁索引
            num_patches_per_angle = len(patch_info3) // 3
            for angle_idx in range(3):
                start_idx = angle_idx * num_patches_per_angle
                end_idx = (angle_idx + 1) * num_patches_per_angle
                angle_patch_info = patch_info3[start_idx:end_idx]
                angle_patches = sampled_patches[0, :, angle_idx, :, :].numpy()

                for patch_idx, (i, j) in enumerate(angle_patch_info):
                    restored_data3[angle_idx, i:i + 128, j:j + 64] = angle_patches[patch_idx]

            # 修复逆变换的形状问题
            min_x3 = min_x3[0].numpy()
            max_x3 = max_x3[0].numpy()
            print("min_x3:", min_x3)  # 应为每个角度的实际最小值
            print("max_x3:", max_x3)  # 应为每个角度的实际最大值
            for angle_idx in range(3):
                restored_data3[angle_idx] = inverse_mu_law_compression(
                    restored_data3[angle_idx],
                    min_x3[angle_idx],
                    max_x3[angle_idx]
                )
            print("模型输出范围:", sampled_patches.min(), sampled_patches.max())  # 应接近 [-1, 1]
            print("恢复后数据范围:", restored_data3.min(), restored_data3.max())  # 应接近原始 RF 信号范围
            das(restored_data3, prefix, number)
            break

# def eval(modelConfig: Dict):
#     """
#     模型评估与生成图像
#     """
#     with torch.no_grad():
#         device = torch.device(modelConfig["device"])
#         model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
#                      num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
#         ckpt = torch.load(os.path.join(
#             modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
#         model.load_state_dict(ckpt)
#         print("model load weight done.")
#         model.eval()
#
#         # 创建 DDIM 采样器
#         sampler = GaussianDiffusionSamplerDDIM(
#             model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], eta=modelConfig["eta"]
#         ).to(device)
#
#         # 从标准正态分布中采样噪声图像
#         noisyImage = torch.randn(
#             size=[modelConfig["batch_size"], 4, modelConfig["img_size"], modelConfig["img_size"]], device=device
#         )
#         saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
#         save_image(saveNoisy, os.path.join(
#             modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"]), nrow=modelConfig["nrow"])
#
#         # 使用 DDIM 采样去噪
#         sampledImgs = sampler(noisyImage)
#         sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
#         save_image(sampledImgs, os.path.join(
#             modelConfig["sampled_dir"], modelConfig["sampledImgName"]), nrow=modelConfig["nrow"])


def das(data, prefix, number):
    prefix = prefix[0]
    number = int(number[0])
    P, xlims, zlims = load_data(prefix, number)

    # Define pixel grid limits (assume y == 0)
    wvln = P.c / P.fc
    dx = wvln / 2.5
    dz = dx  # Use square pixels
    grid = make_pixel_grid(xlims, zlims, dx, dz)
    fnum = 1

    data = np.reshape(data, (128, 1, -1))
    data = np.transpose(data, (1, 0, 2))
    qdata = np.imag(hilbert(data, axis=-1))
    # Make data torch tensors
    x = (data, qdata)

    # Make 1-angle image
    # idx = len(P.angles) // 2  # Choose center angle
    das1 = DAS_PW(P, grid, 0, rxfnum=fnum)
    idas1, qdas1 = das1(x)
    idas1, qdas1 = idas1.detach().cpu().numpy(), qdas1.detach().cpu().numpy()
    iq1 = idas1 + 1j * qdas1
    bimg1 = 20 * np.log10(np.abs(iq1))  # Log-compress
    bimg1 -= np.amax(bimg1)  # Normalize by max value

    # Display images via matplotlib
    extent = [xlims[0] * 1e3, xlims[1] * 1e3, zlims[1] * 1e3, zlims[0] * 1e3]
    plt.imshow(bimg1, vmin=-60, cmap="gray", extent=extent, origin="upper")
    plt.title("1 angle")
    plt.show()
    return