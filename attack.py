import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import math
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T

from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    TexturesUV,
    materials
)

import networks
from utils import download_model_if_doesnt_exist
from data_loader import HumanDataset
from texture_gan import TextureGAN

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
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def get_mean_depth_diff(adv_disp1, ben_disp2, scene_car_mask):
    scaler = 5.4
    dep1_adv = torch.clamp(disp_to_depth(torch.abs(adv_disp1), 0.1, 100)[1] * scene_car_mask.unsqueeze(0) * scaler, max=50)
    dep2_ben = torch.clamp(disp_to_depth(torch.abs(ben_disp2), 0.1, 100)[1] * scene_car_mask.unsqueeze(0) * scaler, max=50)
    mean_depth_diff = torch.sum(dep1_adv - dep2_ben) / torch.sum(scene_car_mask)
    return mean_depth_diff


def get_affected_ratio(disp1, disp2, scene_car_mask):
    scaler = 5.4
    dep1 = torch.clamp(disp_to_depth(torch.abs(disp1), 0.1, 100)[1] * scene_car_mask.unsqueeze(0) * scaler, max=50)
    dep2 = torch.clamp(disp_to_depth(torch.abs(disp2), 0.1, 100)[1] * scene_car_mask.unsqueeze(0) * scaler, max=50)
    ones = torch.ones_like(dep1)
    zeros = torch.zeros_like(dep1)
    affected_ratio = torch.sum(scene_car_mask.unsqueeze(0) * torch.where((dep1 - dep2) > 1, ones, zeros)) / torch.sum(scene_car_mask)
    return affected_ratio


def loss_smooth(img):
    b, c, w, h = img.shape
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2)
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2)
    return torch.square(torch.sum(s1 + s2)) / (b * c * w * h)


def loss_nps(img, color_set):
    # img: [batch_size, h, w, 3]
    # color_set: [color_num, 3]
    _, h, w, c = img.shape
    color_num, c = color_set.shape
    img1 = img.unsqueeze(1)
    color_set1 = color_set.unsqueeze(1).unsqueeze(1).unsqueeze(0)
    gap = torch.min(torch.sum(torch.abs(img1 - color_set1) / 255, -1), 1).values
    return torch.sum(gap) / h / w


def load_texture_from_png(png_path, h, w, device):


    img = Image.open(png_path).convert("RGB")

    transform = T.Resize((h, w))
    img_resized = transform(img)
    img_tensor = T.ToTensor()(img_resized).permute(1, 2, 0)  
    img_tensor = img_tensor.unsqueeze(0).to(device)  
    img_tensor.requires_grad_(True)

    return img_tensor


def train_gen(args):
    torch.autograd.set_detect_anomaly(True)
    model_name = "mono+stereo_1024x320"  # weights fine-tuned on Carla dataset
    download_model_if_doesnt_exist(model_name)
    encoder_path = os.path.join("models", model_name, "encoder.pth")
    depth_decoder_path = os.path.join("models", model_name, "depth.pth")

    # LOADING PRETRAINED MODEL
    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    depth_model = DepthModelWrapper(encoder, depth_decoder).to(args.device)

    depth_model.eval()
    for para in depth_model.parameters():
        para.requires_grad_(False)

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    input_resize = transforms.Resize([feed_height, feed_width])

    H, W = args.camou_shape, args.camou_shape
    resolution = 8
    h, w = int(H / resolution), int(W / resolution)

    expand_kernel = torch.nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(args.device)
    expand_kernel.weight.data.fill_(0)
    expand_kernel.bias.data.fill_(0)
    for i in range(3):
        expand_kernel.weight[i, i, :, :].data.fill_(1)


    expand_kernel.eval()

    color_set = torch.tensor(
        [[0, 0, 0], [255, 255, 255], [0, 18, 79], [5, 80, 214], [71, 178, 243], [178, 159, 211], [77, 58, 0],
         [211, 191, 167], [247, 110, 26], [110, 76, 16]]).to(args.device).float() / 255

    texture_gan = TextureGAN(z_dim=16, img_shape=(h, w)).to(args.device)
    optimizer = optim.Adam(texture_gan.generator.parameters(), lr=args.lr)

    dataset = HumanDataset(args.train_dir, args.img_size, args.obj_name, args.camou_mask, args.device)
    loader = DataLoader(
        dataset=dataset,
        batch_size=8,  
        shuffle=True,  
        num_workers=0,  
    )

    # 1. 恢复优化器的状态（如果有 checkpoint）
    start_epoch = 0
    checkpoint_path = os.path.join(args.log_dir, "checkpoint_stage1.pth")  
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch'] + 1 
        texture_gan.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resumed training from epoch {start_epoch}")

    # 2. 开始训练a
    for epoch in range(start_epoch, 10):
        print('-' * 30 + f'epoch begin: {epoch}' + '-' * 30)
        tqdm_loader = tqdm(loader)
        for i, (index, total_img, total_img0, mask, img, imgs_pred, imgs_pred0) in enumerate(tqdm_loader):
            z = torch.nn.Parameter(torch.randn(1, 16, 4, 4, device=args.device))  
            camou_para = texture_gan.generator(z)  
            camou_para.requires_grad_(True)

            camou_para = camou_para.permute(0, 2, 3, 1)  


            camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  

            dataset.set_textures(camou_para1)

            batch_data = dataset.get_list(index)  
            index, total_img, total_img0, mask, img, imgs_pred, imgs_pred0 = batch_data

            input_image = input_resize(total_img)  # [1, 3, height, width]
            input_image0 = input_resize(total_img0)  # [1, 3, height, width]

            outputs = depth_model(input_image)

            adv_loss = torch.sum(30 * torch.pow(outputs * mask, 2)) / torch.sum(mask)
            tv_loss = loss_smooth(camou_para) * 1e-3
            nps_loss = loss_nps(camou_para, color_set) * 5
            loss = adv_loss 

            log_file = os.path.join(args.log_dir, "training_log_stage1.txt")

            if i % 10 == 0:
                log_message = (f"Epoch [{epoch}/15], Batch [{i}/{len(tqdm_loader)}]:\n"
                               f"  adv_loss: {adv_loss.item():.4f}\n"
                               f"  tv_loss: {tv_loss.item():.4f}\n"
                               f"  nps_loss: {nps_loss.item():.4f}\n"
                               f"  total_loss: {loss.item():.4f}\n")


                print(log_message)


                with open(log_file, 'a') as f:  
                    f.write(log_message)


            optimizer.zero_grad()  
            loss.backward(retain_graph=True)        
            optimizer.step()       

            if i % 10 == 0:
                save_dir = os.path.join(args.log_dir, "img_stage1")
                os.makedirs(save_dir, exist_ok=True)

                total_img_np = total_img.data.cpu().numpy()[0] * 255
                total_img_np = Image.fromarray(np.transpose(total_img_np, (1, 2, 0)).astype('uint8'))
                total_img_np.save(os.path.join(save_dir, f'epoch_{epoch}_batch_{i}_test_total.jpg'))

                total_img_np0 = total_img0.data.cpu().numpy()[0] * 255
                total_img_np0 = Image.fromarray(np.transpose(total_img_np0, (1, 2, 0)).astype('uint8'))
                total_img_np0.save(os.path.join(save_dir, f'epoch_{epoch}_batch_{i}_test_total0.jpg'))


        torch.save({
            'epoch': epoch,
            'model_state_dict': texture_gan.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")


        camou_png = cv2.cvtColor((camou_para1[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.log_dir, f'{epoch}_camou_stage1.png'), camou_png)
        np.save(os.path.join(args.log_dir, f'{epoch}_camou_stage1.npy'), camou_para.detach().cpu().numpy())

def train_z(args):
    torch.autograd.set_detect_anomaly(True)
    model_name = "mono+stereo_1024x320"  # weights fine-tuned on Carla dataset
    download_model_if_doesnt_exist(model_name)
    encoder_path = os.path.join("models", model_name, "encoder.pth")
    depth_decoder_path = os.path.join("models", model_name, "depth.pth")

    # LOADING PRETRAINED MODEL
    encoder = networks.ResnetEncoder(18, False)
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
    depth_decoder.load_state_dict(loaded_dict)

    depth_model = DepthModelWrapper(encoder, depth_decoder).to(args.device)

    depth_model.eval()
    for para in depth_model.parameters():
        para.requires_grad_(False)

    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    input_resize = transforms.Resize([feed_height, feed_width])

    H, W = args.camou_shape, args.camou_shape
    resolution = 8
    h, w = int(H / resolution), int(W / resolution)

    expand_kernel = torch.nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(args.device)
    expand_kernel.weight.data.fill_(0)
    expand_kernel.bias.data.fill_(0)
    for i in range(3):
        expand_kernel.weight[i, i, :, :].data.fill_(1)


    expand_kernel.eval()

    color_set = torch.tensor(
        [[0, 0, 0], [255, 255, 255], [0, 18, 79], [5, 80, 214], [71, 178, 243], [178, 159, 211], [77, 58, 0],
            [211, 191, 167], [247, 110, 26], [110, 76, 16]]).to(args.device).float() / 255

    texture_gan = TextureGAN(z_dim=16, img_shape=(h, w)).to(args.device)


    checkpoint_stage1 = os.path.join(args.log_dir, "checkpoint_stage1.pth")
    if os.path.exists(checkpoint_stage1):
        checkpoint = torch.load(checkpoint_stage1)
        texture_gan.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded generator weights from {checkpoint_stage1}")
    else:
        print(f"Checkpoint {checkpoint_stage1} not found. Please complete stage1 training first.")
        return


    for param in texture_gan.generator.parameters():
        param.requires_grad = False


    z = torch.nn.Parameter(torch.randn(1, 16, 4, 4, device=args.device))  # 修改为 (1, 16, 4, 4)
    print(f"z.requires_grad: {z.requires_grad}")
    delta = torch.nn.Parameter(torch.zeros(1, 3, h, w, device=args.device))
    
    optimizer_z = optim.Adam([z], lr=args.lr)
    optimizer = optim.Adam([z, delta], lr=args.lr)
    dataset = HumanDataset(args.train_dir, args.img_size, args.obj_name, args.camou_mask, args.device)
    loader = DataLoader(
        dataset=dataset,
        batch_size=8,  
        shuffle=True,  
        num_workers=0,  
    )

    
    for epoch in range(0, 20): 
        print('-' * 30 + f'Stage 2 - Epoch begin: {epoch}' + '-' * 30)
        tqdm_loader = tqdm(loader)
        for i, (index, total_img, total_img0, mask, img, imgs_pred, imgs_pred0) in enumerate(tqdm_loader):


            gen_texture = texture_gan.generator(z)  # [1, 3, h, w]
            camou_para = gen_texture + delta
            camou_para.requires_grad_(True)

            camou_para = camou_para.permute(0, 2, 3, 1)  # [1, h, w, 3]
            camou_para.retain_grad()


            camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [1, H, W, 3]

            dataset.set_textures(camou_para1)


            batch_data = dataset.get_list(index)  
            index, total_img, total_img0, mask, img, imgs_pred, imgs_pred0 = batch_data

            input_image = input_resize(total_img)  # [1, 3, height, width]
            input_image0 = input_resize(total_img0)  # [1, 3, height, width]


            outputs = depth_model(input_image)

            adv_loss = torch.sum(100 * torch.pow(outputs * mask, 2)) / torch.sum(mask)
            tv_loss = loss_smooth(camou_para) * 1e-3
            nps_loss = loss_nps(camou_para, color_set) * 5
            loss = adv_loss + tv_loss + nps_loss


            log_file = os.path.join(args.log_dir, "training_log_stage2.txt")

            if i % 10 == 0:
                log_message = (f"Stage 2 - Epoch [{epoch}/10], Batch [{i}/{len(tqdm_loader)}]:\n"
                                f"  adv_loss: {adv_loss.item():.4f}\n"
                                f"  tv_loss: {tv_loss.item():.4f}\n"
                                f"  nps_loss: {nps_loss.item():.4f}\n"
                                f"  total_loss: {loss.item():.4f}\n")

                print(log_message)

                with open(log_file, 'a') as f:  
                    f.write(log_message)


            optimizer.zero_grad()
            torch.autograd.grad(loss, camou_para, retain_graph=True)
            loss.backward(retain_graph=True)
            print(f"z.grad norm: {z.grad.norm().item()}")
            print(f"camou_para.grad norm: {camou_para.grad.norm().item()}")

            optimizer.step()
            # if i % 100 == 0:
            #     save_dir = os.path.join(args.log_dir, "img_stage2")
            #     os.makedirs(save_dir, exist_ok=True)

            #     camou_png = cv2.cvtColor((camou_para1[0].detach().cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            #     cv2.imwrite(os.path.join(save_dir, f'epoch_{epoch}_batch_{i}_camou_stage2.png'), camou_png)
            #     np.save(os.path.join(save_dir, f'epoch_{epoch}_batch_{i}_camou_stage2.npy'), camou_para.detach().cpu().numpy())

        z_save_path = os.path.join(args.log_dir, f"z_epoch_{epoch}_stage2.npy")
        np.save(z_save_path, z.detach().cpu().numpy())
        print(f"Saved optimized z at {z_save_path}")

        camou_save_path = os.path.join(args.log_dir, f"camou_epoch_{epoch}_stage2.npy")
        np.save(camou_save_path, camou_para.detach().cpu().numpy())
        print(f"Saved camou at {camou_save_path}")

    final_generator_path = os.path.join(args.log_dir, "texture_gan_stage1_final.pth")
    torch.save(texture_gan.state_dict(), final_generator_path)
    print(f"Saved final generator weights at {final_generator_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--camou_mask", type=str, default='./man/mask.jpg',
                        help="Path to the combined camouflage texture mask")
    parser.add_argument("--camou_shape", type=int, default=1024, help="Shape of camouflage texture")
    parser.add_argument("--obj_name", type=str, default='./man/man.obj')
    parser.add_argument("--device", type=str, default="cuda:2", help="Device to use, e.g., 'cuda:0'")
    parser.add_argument("--train_dir", type=str, default='./carla/dataset/')
    parser.add_argument("--img_size", type=int, nargs=2, default=[320, 1024],
                        help="Image size as two integers: height and width")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--log_dir", type=str, default='./man_train/')
    parser.add_argument("--use_png_texture", action='store_true', help="use PNG file as texture seed")
    parser.add_argument("--texture_png_path", type=str, default=None, help="path to PNG file for texture seed")
    args = parser.parse_args()


    os.makedirs(args.log_dir, exist_ok=True)
    train_gen(args)
    train_z(args)
