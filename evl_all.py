import os
import torch
import sys
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from data_loader_man_full import HumanDataset
from utils import download_model_if_doesnt_exist
import networks
from torch.utils.data import DataLoader
import argparse
import re
import torchvision.transforms as T

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'


class DepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(DepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_image):
        features = self.encoder(input_image)
        outputs = self.decoder(features)
        disp = outputs[("disp", 0)]
        return disp


def disp_to_depth(disp, min_depth, max_depth):
    """
    将深度网络输出的disp转换为深度（depth）。
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def get_mean_depth_diff(adv_disp1, ben_disp2, scene_car_mask):
    """
    E_d: 评估深度差异的平均值
    """
    scaler = 5.4
    dep1_adv = torch.clamp(
        disp_to_depth(torch.abs(adv_disp1), 0.1, 100)[1] * scene_car_mask.unsqueeze(0) * scaler,
        max=50
    )
    dep2_ben = torch.clamp(
        disp_to_depth(torch.abs(ben_disp2), 0.1, 100)[1] * scene_car_mask.unsqueeze(0) * scaler,
        max=50
    )
    mean_depth_diff = torch.sum(torch.abs(dep1_adv - dep2_ben)) / torch.sum(scene_car_mask)
    return mean_depth_diff


def get_affected_ratio(disp1, disp2, scene_car_mask):
    """
    R_a: 评估受到影响像素的比例（depth差异大于1的部分）
    """
    scaler = 5.4
    dep1 = torch.clamp(
        disp_to_depth(torch.abs(disp1), 0.1, 100)[1] * scene_car_mask.unsqueeze(0) * scaler,
        max=50
    )
    dep2 = torch.clamp(
        disp_to_depth(torch.abs(disp2), 0.1, 100)[1] * scene_car_mask.unsqueeze(0) * scaler,
        max=50
    )
    ones = torch.ones_like(dep1)
    zeros = torch.zeros_like(dep1)
    affected_ratio = torch.sum(
        scene_car_mask.unsqueeze(0) * torch.where(torch.abs(dep1 - dep2) > 1, ones, zeros)
    ) / torch.sum(scene_car_mask)
    return affected_ratio


def load_texture_from_png(png_path, h, w, device):
    """
    加载 PNG 文件并转换为 (H, W, 3) 的张量，然后可用于你的人体数据集中。
    """
    img = Image.open(png_path).convert("RGB")
    transform = T.Resize((h, w))
    img_resized = transform(img)
    img_tensor = T.ToTensor()(img_resized).permute(1, 2, 0)  # 转 [H, W, C]
    img_tensor = img_tensor.unsqueeze(0).to(device)          # 转 [1, H, W, C]
    img_tensor.requires_grad_(True)
    return img_tensor


def extract_weather_from_filename(filename):
    """
    从文件名解析天气类别。如果是 Town04 且原本是 w2/w3，则视为 w4。
    如果无法解析则返回 'unknown'。
    """
    match_town = re.search(r"Town\d+", filename)  # 获取 Town 信息
    town = match_town.group() if match_town else "Unknown"

    weather_mapping = {
        "w1": "1",
        "w2": "2",
        "w3": "3"
    }

    match_w = re.search(r"w\d", filename)  # 获取 w1, w2, w3
    weather_code = match_w.group() if match_w else "unknown"

    # Town04 情况特殊
    if town == "Town04" and weather_code in ["w2", "w3"]:
        return "4"

    return weather_mapping.get(weather_code, "unknown")


def evaluate(args):
    """
    同时评估：天气 + 角度 的平均深度差异 E_d 和影响比例 R_a，并将结果保存到文件。
    """
    # ------------------------------------------------
    # 1. 准备深度估计模型
    # ------------------------------------------------
    model_name = "mono+stereo_1024x320"
    download_model_if_doesnt_exist(model_name)
    encoder_path = os.path.join("models", model_name, "encoder.pth")
    depth_decoder_path = os.path.join("models", model_name, "depth.pth")

    # 构建模型
    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    # 加载权重
    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    # 封装模型
    depth_model = DepthModelWrapper(encoder, depth_decoder).to(args.device)
    depth_model.eval()
    for para in depth_model.parameters():
        para.requires_grad_(False)

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    input_resize = transforms.Resize([feed_height, feed_width])

    # ------------------------------------------------
    # 2. 构建数据集
    # ------------------------------------------------
    dataset = HumanDataset(args.eval_dir, args.img_size, args.obj_name, args.camou_mask, args.device)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ------------------------------------------------
    # 3. 准备纹理
    # ------------------------------------------------
    resolution = 8
    expand_kernel = torch.nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(args.device)
    expand_kernel.weight.data.fill_(0)
    expand_kernel.bias.data.fill_(0)
    for i in range(3):
        expand_kernel.weight[i, i, :, :].data.fill_(1)

    if args.texture_mode == "npy":
        # 从 npy 文件加载
        camou_para = np.load(args.texture_npy)
        camou_para = torch.from_numpy(camou_para).float().to(args.device)
        # camou_para = torch.rand([1, 128, 128, 3]).float().to(args.device)
    elif args.texture_mode == "png":
        # 从 png 文件加载
        camou_para = load_texture_from_png(args.texture_png, 128, 128, args.device)
    else:
        raise ValueError("Invalid texture mode. Choose 'npy' or 'png'.")

    # 扩展纹理到 (1, H, W, 3) 范围并裁剪到[0,1]
    camou_para_expanded = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    camou_para_expanded = torch.clamp(camou_para_expanded, 0, 1)
    dataset.set_textures(camou_para_expanded)

    # ------------------------------------------------
    # 4. 初始化统计量容器
    # ------------------------------------------------
    # (1) 按天气统计
    weather_stats = {}
    # (2) 按角度统计
    angle_stats = {}
    # (3) 全局统计
    total_E_d, total_R_a = 0.0, 0.0
    total_samples = 0

    tqdm_loader = tqdm(loader, desc="Evaluating")

    # ------------------------------------------------
    # 5. 遍历数据并统计
    # ------------------------------------------------
    for i, (index, adv_img, ben_img, target_mask, img, imgs_pred, imgs_pred0) in enumerate(tqdm_loader):
        # 5.1 提取文件名/角度/天气
        filename = dataset.file_name(i)  # 需要确保 HumanDataset 有此方法
        angle = dataset.get_angle(i)     # 需要确保 HumanDataset 有此方法
        weather = extract_weather_from_filename(filename)

        # 5.2 预处理图像
        adv_img = input_resize(adv_img).to(args.device)
        ben_img = input_resize(ben_img).to(args.device)
        target_mask2 = input_resize(target_mask).to(args.device)[:, 0, :, :]

        # 5.3 执行深度估计并计算指标
        with torch.no_grad():
            adv_disp = depth_model(adv_img)
            ben_disp = depth_model(ben_img)

        mean_depth_diff = get_mean_depth_diff(adv_disp, ben_disp, target_mask2)
        affected_ratio = get_affected_ratio(adv_disp, ben_disp, target_mask2)

        # 全局累加
        total_E_d += mean_depth_diff.item()
        total_R_a += affected_ratio.item()
        total_samples += 1

        # 5.4 按天气统计
        if weather not in weather_stats:
            weather_stats[weather] = {
                "total_mean_depth_diff": 0.0,
                "total_affected_ratio": 0.0,
                "num_samples": 0
            }
        weather_stats[weather]["total_mean_depth_diff"] += mean_depth_diff.item()
        weather_stats[weather]["total_affected_ratio"] += affected_ratio.item()
        weather_stats[weather]["num_samples"] += 1

        # 5.5 按角度统计
        if angle not in angle_stats:
            angle_stats[angle] = {
                "E_d_sum": 0.0,
                "R_a_sum": 0.0,
                "count": 0
            }
        angle_stats[angle]["E_d_sum"] += mean_depth_diff.item()
        angle_stats[angle]["R_a_sum"] += affected_ratio.item()
        angle_stats[angle]["count"] += 1

        # 5.6 打印当前进度
        tqdm_loader.set_postfix({
            "E_d": f"{mean_depth_diff:.4f}",
            "R_a": f"{affected_ratio:.4f}",
            "Weather": weather,
            "Angle": angle
        })

    # ------------------------------------------------
    # 6. 最终结果计算与输出
    # ------------------------------------------------
    # (1) 全局平均
    avg_E_d = total_E_d / total_samples if total_samples > 0 else 0.0
    avg_R_a = total_R_a / total_samples if total_samples > 0 else 0.0

    # 为了同时在终端显示与写文件，这里先把结果组织成字符串
    results_str_list = []

    results_str_list.append("========== Overall Evaluation Results ==========")
    results_str_list.append(f"Total Samples = {total_samples}")
    results_str_list.append(f"  Average E_d = {avg_E_d:.4f}")
    results_str_list.append(f"  Average R_a = {avg_R_a:.4f}")
    results_str_list.append("")

    # (2) 按天气输出
    results_str_list.append("========== Weather-wise Evaluation Results ==========")
    for weather, stats in weather_stats.items():
        num_samples = stats["num_samples"]
        if num_samples > 0:
            avg_mean_depth_diff = stats["total_mean_depth_diff"] / num_samples
            avg_affected_ratio = stats["total_affected_ratio"] / num_samples
        else:
            avg_mean_depth_diff = 0.0
            avg_affected_ratio = 0.0
        results_str_list.append(f"- Weather = {weather} | Samples = {num_samples}")
        results_str_list.append(f"    Average E_d = {avg_mean_depth_diff:.4f}")
        results_str_list.append(f"    Average R_a = {avg_affected_ratio:.4f}")
    results_str_list.append("")

    # (3) 按角度输出
    results_str_list.append("========== Angle-wise Evaluation Results ==========")
    for angle in sorted(angle_stats.keys()):
        count = angle_stats[angle]["count"]
        if count > 0:
            avg_E_d_angle = angle_stats[angle]["E_d_sum"] / count
            avg_R_a_angle = angle_stats[angle]["R_a_sum"] / count
        else:
            avg_E_d_angle = 0.0
            avg_R_a_angle = 0.0
        results_str_list.append(f"- Angle = {angle} | Samples = {count}")
        results_str_list.append(f"    Average E_d = {avg_E_d_angle:.4f}")
        results_str_list.append(f"    Average R_a = {avg_R_a_angle:.4f}")
    results_str_list.append("")

    # 将结果打印到终端
    for line in results_str_list:
        print(line)

    # ------------------------------------------------
    # 7. 写入结果到指定文件
    # ------------------------------------------------
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w") as f:
        for line in results_str_list:
            f.write(line + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, default="/mnt/data/JZD/carla/dataset2/")
    parser.add_argument("--camou_mask", type=str, default="/mnt/data/JZD/3D2Fool/man/mask.jpg")
    parser.add_argument("--obj_name", type=str, default="/mnt/data/JZD/3D2Fool/man/man.obj")
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:1"))
    parser.add_argument("--img_size", type=tuple, default=(320, 1024))
    parser.add_argument(
        "--texture_mode",
        type=str,
        choices=["npy", "png"],
        default="npy",
        help="Choose texture source: 'npy' for .npy file, 'png' for .png file"
    )
    parser.add_argument(
        "--texture_npy",
        type=str,
        default="/mnt/data/JZD/3D2Fool/man_train_gan-2stage-e10-jin-tv_loss-100-1e-3/camou_epoch_19_stage2.npy",
        help="Path to the .npy texture file"
    )
    parser.add_argument(
        "--texture_png",
        type=str,
        default="/mnt/data/JZD/3D2Fool/img2/SAAM2.png",
        help="Path to the .png texture file"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./evl/wea.txt",
        help="Where to save the final evaluation results."
    )

    args = parser.parse_args()
    evaluate(args)
