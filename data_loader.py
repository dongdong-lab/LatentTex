import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
import random
import math
import pickle
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
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
    SoftPhongShader,
    HardPhongShader,
    TexturesUV,
    BlendParams,
    SoftSilhouetteShader,
    materials
)


class HumanDataset(Dataset):
    def __init__(self, data_dir, img_size, obj_name, camou_mask, device=torch.device("cuda:0"), tex_trans_flag=True,
                 phy_trans_flag=True):

        self.tex_trans_flag = tex_trans_flag
        self.phy_trans_flag = phy_trans_flag
        self.data_dir = data_dir
        with open('./carla/positions.pkl', 'rb') as ann_file:
            self.ann = pickle.load(ann_file)
        self.files = os.listdir(self.data_dir)
        print('数据集长度: ', len(self.files))

        self.img_size = img_size
        self.device = device
        self.camou_mask = torch.from_numpy(cv2.imread(camou_mask) / 255).to(device).unsqueeze(0).float()

        self.verts, self.faces, self.aux = load_obj(
            obj_name,
            load_textures=True,
            create_texture_atlas=False,
            texture_atlas_size=4,
            texture_wrap='repeat',
            path_manager=None,
        )
        self.camou0 = list(self.aux.texture_images.values())[0].to(self.device)[None]  

        self.mesh = load_objs_as_meshes([obj_name], device=device)
        self.verts_uvs = self.aux.verts_uvs.to(device)  
        self.faces_uvs = self.faces.textures_idx.to(device)  

        self.raster_settings = RasterizationSettings(
            image_size=self.img_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            max_faces_per_bin=250000
        )

        self.lights = PointLights(device=device, location=[[100.0, 85, 100.0]])
        self.cameras = ''
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings
            ),
            shader=HardPhongShader(
                device=device,
                cameras=self.cameras,
                lights=self.lights
            )
        )

    def Blur_trans(self, img1, img0, dist):
        kernel_size_list = [1, 3, 5, 7]  

        if dist < 250:  
            if np.random.rand(1) < 0.2:  
                kernel_size = kernel_size_list[np.random.randint(0, 2)]  
                delta = np.random.rand(1).tolist()[0] * dist / 700  
                if delta > 0.5:  
                    delta = 0.5  
                img1 = T.GaussianBlur(kernel_size, delta)(img1)
                img0 = T.GaussianBlur(kernel_size, delta)(img0)
        elif dist < 300:  
            if np.random.rand(1) < 0.4: 
                kernel_size = kernel_size_list[np.random.randint(1, 3)]  
                delta = np.random.rand(1).tolist()[0] * dist / 900  
                if delta > 1: 
                    delta = 1  
                img1 = T.GaussianBlur(kernel_size, delta)(img1)
                img0 = T.GaussianBlur(kernel_size, delta)(img0)
        else:  
            if np.random.rand(1) < 0.8:  
                kernel_size = kernel_size_list[np.random.randint(2, 4)]  
                delta = np.random.rand(1).tolist()[0] * dist / 1500  
                if delta > 1.5:  
                    delta = 1.5  
                img1 = T.GaussianBlur(kernel_size, delta)(img1)
                img0 = T.GaussianBlur(kernel_size, delta)(img0)

        return img1, img0

    def Color_trans(self, img1, img0, brightness=[0.7, 1.3], contrast=[0.9, 1.1], saturation=[0.9, 1.1], hue=[-0.05, 0.05]):
        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))
        def Color_change(img, b, c, s, h):
            img = TF.adjust_brightness(img, b)
            img = TF.adjust_contrast(img, c)
            img = TF.adjust_saturation(img, s)
            img = TF.adjust_hue(img, h)
            return img
        img1 = Color_change(img1, b, c, s, h)
        img0 = Color_change(img0, b, c, s, h)
        return img1, img0

    def myColor_trans(self, img1, img0, flag=0, brightness=[0.7, 1.3], contrast=[0.9, 1.1], saturation=[0.9, 1.1],
                      hue=[-0.05, 0.05]):
        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        def Color_change(img, b, c, s, h):
            img = TF.adjust_brightness(img, b)  
            img = TF.adjust_contrast(img, c)  
            img = TF.adjust_saturation(img, s)  
            img = TF.adjust_hue(img, h)  
            return img

        img1 = Color_change(img1, b, c, s, h)
        img0 = Color_change(img0, b, c, s, h)

        if flag == 1:
            img1, img0 = self.add_shadow(img1, img0)  
        elif flag == 2:
            img1, img0 = self.add_exposure(img1, img0)  

        return img1, img0

    def add_shadow(self, img1, img0, shadow_dimension=5):
        x1 = 0
        x2 = self.img_size[1]  
        y1 = 0
        y2 = self.img_size[0]  
        mask = np.ones([self.img_size[0], self.img_size[1], 3])  
        vertex = []

        for dimensions in range(shadow_dimension): 
            vertex.append((random.randint(x1, x2), random.randint(y1, y2)))  
        vertices = np.array([vertex], dtype=np.int32)  

        b = np.random.rand(1).tolist()[0] * 0.2 + 0.7  
        cv2.fillPoly(mask, vertices, (b, b, b))  

        mask = torch.from_numpy(mask).float().to(self.device).permute(2, 0, 1).unsqueeze(0)

        mask = T.GaussianBlur(3, 1.5)(mask)

        img11 = img1 * mask[:, :, :, :]  
        img00 = img0 * mask[:, :, :, :]  

        return img11, img00

    def add_exposure(self, img1, img0, exposure_dimension=5):
        x1 = 0
        x2 = self.img_size[1]  
        y1 = 0
        y2 = self.img_size[0]  
        mask = np.ones([self.img_size[0], self.img_size[1], 3])  
        vertex = []

        for dimensions in range(exposure_dimension):  
            vertex.append((random.randint(x1, x2), random.randint(y1, y2)))  

        vertices = np.array([vertex], dtype=np.int32)

        b = np.random.rand(1).tolist()[0] * 0.2 + 1.1 

        cv2.fillPoly(mask, vertices, (b, b, b))  

        mask = torch.from_numpy(mask).float().to(self.device).permute(2, 0, 1).unsqueeze(0)

        mask = T.GaussianBlur(3, 1.5)(mask)

        img11 = torch.clamp(img1 * mask[:, :, :, :], 0, 1)
        img00 = torch.clamp(img0 * mask[:, :, :, :], 0, 1)

        return img11, img00

    def frog(self, img, A=0.5, beta=0.08):

        (chs, row, col) = img[0].shape
        img1 = img.clone()
        size = math.sqrt(max(row, col))
        center = [row * np.random.rand(1).tolist()[0], col * np.random.rand(1).tolist()[0]]
        center = torch.tensor(center).to(self.device)
        coordinates = torch.stack(torch.meshgrid(torch.arange(row), torch.arange(col)), -1).to(self.device)
        d = -0.04 * torch.sqrt(torch.sum(torch.pow(coordinates-center, 2), 2)) + size
        td = torch.exp(-beta * d)
        img1[0] = img[0] * td + A * (1 - td)
        return img1

    def add_rain(self, img1, img0, no_of_drops=10, drop_length=100, drop_width=2):

        slant = np.random.randint(-5, 5)  

        rain_drops = self.generate_random_lines(no_of_drops, slant * 2, drop_length, drop_width)


        mask = np.ones([self.img_size[0], self.img_size[1], 3])  
        mask_color = (0.85, 0.85, 0.85)

        for rain_drop in rain_drops:
     
            cv2.line(mask,
                     (rain_drop[0], rain_drop[1]),  
                     (rain_drop[0] + slant, rain_drop[1] + drop_length),  
                     mask_color,  
                     drop_width)  

        mask = torch.from_numpy(mask).float().to(self.device).permute(2, 0, 1).unsqueeze(0)

        mask = T.GaussianBlur(3, 1.5)(mask)

        img11 = img1 * mask[:, :, :, :]
        img00 = img0 * mask[:, :, :, :]

        return img11, img00

    def generate_random_lines(self, no_of_drops=10, slant=0, drop_length=100, drop_width=2):
        drops = []
        for i in range(no_of_drops): ## If You want heavy rain, try increasing this
            if slant<0:
                x= np.random.randint(slant+drop_width, self.img_size[1]-drop_width)
            else:
                x= np.random.randint(drop_width, self.img_size[1]-slant-drop_width)
            y= np.random.randint(drop_width, self.img_size[0]-drop_length-drop_width)
            drops.append((x,y))
        return drops


    def EoT(self, img1, img0, index):
        # self.files[index] = 'Town04_w2_0l_cam2.jpg'
        map = self.files[index].split('_')[0]
        weather = self.files[index].split('_')[5]
        eye = self.ann[self.files[index]]['camera_pos'].copy()
        dist = np.sqrt(np.sum(np.power(eye, 2)))
        print("dist:",dist)
        # sys.exit()
        img1, img0 = self.Blur_trans(img1, img0, dist)  
        flag = random.randint(0, 2)
        if weather == 'w1':  # 晴朗
            img1, img0 = self.Color_trans(img1, img0, brightness=[0.9, 1.3])
        elif weather == 'w2':  # 多云
            img1, img0 = self.Color_trans(img1, img0, brightness=[0.8, 1.2])
        elif weather == 'w3':  # 阴雨
            img1, img0 = self.Color_trans(img1, img0, brightness=[0.7, 1.1])
        return img1, img0

    def phy_trans(self, img1, img0, index):
        # self.files[index] = 'Town04_w2_0l_cam2.jpg'
        map = self.files[index].split('_')[0]
        weather = self.files[index].split('_')[5]
        eye = self.ann[self.files[index]]['camera_pos'].copy()
        dist = np.sqrt(np.sum(np.power(eye, 2)))
        img1, img0 = self.Blur_trans(img1, img0, dist)  
        flag = random.randint(0, 2)
        if weather == 'w1':  # 晴朗
            # img1, img0 = self.Color_trans(img1, img0, brightness=[0.9, 1.3])
            img1, img0 = self.myColor_trans(img1, img0, flag, brightness=[0.9, 1.3])  
        elif weather == 'w2':  # 多云
            # img1, img0 = self.Color_trans(img1, img0, brightness=[0.8, 1.2])
            img1, img0 = self.myColor_trans(img1, img0, flag, brightness=[0.8, 1.2])  
            if map == 'Town04':  # 加雾
                A = (dist - 300) / (500 - 300) * 0.3 + np.random.rand(1).tolist()[0] * 0.06 - 0.03  
                beta = (dist - 300) / (500 - 300) * 0.04 + np.random.rand(1).tolist()[0] * 0.01 - 0.005  
                img1 = self.frog(img1, A, beta)
                img0 = self.frog(img0, A, beta)
        elif weather == 'w3':  # 阴雨
            # img1, img0 = self.Color_trans(img1, img0, brightness=[0.7, 1.1])
            img1, img0 = self.add_rain(img1, img0)
            img1, img0 = self.myColor_trans(img1, img0, flag, brightness=[0.7, 1.1]) 
            if map == 'Town04':  # 加雾
                A = (dist - 300) / (500 - 300) * 0.5 + np.random.rand(1).tolist()[0] * 0.1 - 0.05  
                beta = (dist - 300) / (500 - 300) * 0.08 + np.random.rand(1).tolist()[0] * 0.02 - 0.01  
                img1 = self.frog(img1, A, beta)
                img0 = self.frog(img0, A, beta)
        return img1, img0

    def tex_trans(self, camou):
        # mask=[1, 4096, 4096, 3], camou=[1, 1024, 1024, 3]
        camou_column = []
        for i in range(10):
            camou_row_list = []
            for j in range(10):
                camou1 = T.RandomHorizontalFlip(p=0.5)(camou.permute(0, 3, 1, 2)[0])  # 依概率p水平翻转
                camou2 = T.RandomVerticalFlip(p=0.5)(camou1)  # 依概率p垂直翻转
                if np.random.rand(1) > 0.5:
                    camou3 = TF.rotate(camou2, 90)
                else:
                    camou3 = camou2
                # temp = camou3.detach().cpu().permute(1,2,0).numpy()*255
                # cv2.imwrite('./assets/tex1.jpg', cv2.cvtColor(temp, cv2.COLOR_RGB2BGR).astype(np.uint8))
                camou_row_list.append(camou3)
            camou_row = torch.cat(tuple(camou_row_list), 1)
            # print(camou_row.shape)
            camou_column.append(camou_row)
        camou_full = torch.cat(tuple(camou_column), 2).unsqueeze(0)
        # temp = camou_full[0].detach().cpu().permute(1,2,0).numpy()*255
        # cv2.imwrite('./assets/tex2.jpg', cv2.cvtColor(temp, cv2.COLOR_RGB2BGR).astype(np.uint8))
        camou_crop = T.RandomCrop(8192)(camou_full).permute(0, 2, 3, 1)  # 随机裁剪
        # temp = camou_crop[0].detach().cpu().numpy()*255
        # cv2.imwrite('./assets/tex3.jpg', cv2.cvtColor(temp, cv2.COLOR_RGB2BGR).astype(np.uint8))
        # print(camou_crop.shape)
        return camou_crop

    def tex_trans0(self, camou):
        # mask=[1, 4096, 4096, 3], camou=[1, 1024, 1024, 3]
        camou_column = []
        for i in range(8):
            camou_row_list = []
            for j in range(8):
                camou_row_list.append(camou)
            camou_row = torch.cat(tuple(camou_row_list), 1)
            camou_column.append(camou_row)
        camou_full = torch.cat(tuple(camou_column), 2)
        return camou_full

    def set_textures(self, camou):
        # temp = self.tex_trans(camou)
        # camou_png = cv2.cvtColor((temp[0].detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # cv2.imwrite('./res/camou.png', camou_png)
        # print(self.camou0.shape)
        # print(camou.shape)
        # print(self.camou_mask.shape)
        # print(self.tex_trans(camou).shape)

        if self.tex_trans_flag:
            image = self.camou0 * (1-self.camou_mask) + self.tex_trans(camou) * self.camou_mask
        else:
            image = self.camou0 * (1-self.camou_mask) + self.tex_trans0(camou) * self.camou_mask
        # image_png = cv2.cvtColor((image[0].detach().cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # cv2.imwrite('./res/image.png', image_png)

        self.mesh.textures = TexturesUV(verts_uvs=[self.verts_uvs], faces_uvs=[self.faces_uvs], maps=image)


    def __getitem__(self, index):
        file_name = self.files[index]  
        if file_name not in self.ann:
            print(f"Warning: {file_name} 没有对应的注释数据！")
            return None  

        # load camera parameters
        eye = self.ann[self.files[index]]['camera_pos'].copy()
        camera_up = [0.0, 1.0, 0.0]

        R, T = look_at_view_transform(eye=(tuple(eye),), up=(tuple(camera_up),), at=((0, 100, 0),))
        self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, znear=1.0, zfar=500.0, fov=45.0)

        self.renderer.shader.lights = PointLights(device=self.device, location=[eye])

        self.materials = Materials(
            device=self.device,
            specular_color=[[1.0, 1.0, 1.0]],
            shininess=500.0
        )

        self.renderer.rasterizer.cameras = self.cameras
        self.renderer.shader.cameras = self.cameras

        imgs_pred1 = self.renderer(self.mesh, materials=self.materials)[:, ..., :3]
        imgs_pred1 = imgs_pred1.permute(0, 3, 1, 2)  # 转换为 [1, 3, H, W]

        self.mesh0 = self.mesh.clone()
        self.mesh0.textures = TexturesUV(verts_uvs=[self.verts_uvs], faces_uvs=[self.faces_uvs], maps=self.camou0)
        imgs_pred0 = self.renderer(self.mesh0, materials=self.materials)[:, ..., :3]
        imgs_pred0 = imgs_pred0.permute(0, 3, 1, 2)

        if self.phy_trans_flag:

            imgs_pred11, imgs_pred00 = self.phy_trans(imgs_pred1, imgs_pred0, index)
        else:

            imgs_pred11, imgs_pred00 = self.EoT(imgs_pred1, imgs_pred0, index)

 
        file_path = os.path.join(self.data_dir, self.files[index])
        # file_path = '/data/zjh/mde_carla/rgb/Town04_w2_0l_cam2.jpg'
        img = cv2.imread(file_path)  # [640, 1600, 3] bgr
        img = img[40:-100, :, ::-1]  # [500, 1600, 3] rgb
        img_cv = cv2.resize(img, (self.img_size[1], self.img_size[0]))  # [320, 1024, 3]
        img = np.transpose(img_cv, (2, 0, 1))
        img = np.resize(img, (1, img.shape[0], img.shape[1], img.shape[2]))  # [1, 3, 320, 1024]
        img = torch.from_numpy(img).cuda(device=self.device).float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0


        contour = torch.where((imgs_pred1 == 1), torch.zeros(1).to(self.device), torch.ones(1).to(self.device))


        total_img = torch.where((contour == 0.), img, imgs_pred11)  
        total_img0 = torch.where((contour == 0.), img, imgs_pred00)  
        return index, total_img[0], total_img0[0], contour[0], img[0], imgs_pred1[0], imgs_pred0[0]
    
    def get_list(self, indices):

        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        indices_list, total_imgs, total_imgs0, masks, imgs, imgs_pred_list, imgs_pred0_list = [], [], [], [], [], [], []

        for idx in indices:
            data = self[idx]  
            indices_list.append(data[0])
            total_imgs.append(data[1])
            total_imgs0.append(data[2])
            masks.append(data[3])
            imgs.append(data[4])
            imgs_pred_list.append(data[5])
            imgs_pred0_list.append(data[6])


        indices_tensor = torch.tensor(indices_list)
        total_imgs_tensor = torch.stack(total_imgs)  # Shape: [batch_size, 3, H, W]
        total_imgs0_tensor = torch.stack(total_imgs0)  # Shape: [batch_size, 3, H, W]
        masks_tensor = torch.stack(masks)  # Shape: [batch_size, H, W]
        imgs_tensor = torch.stack(imgs)  # Shape: [batch_size, 3, H, W]
        imgs_pred_tensor = torch.stack(imgs_pred_list)  # Shape: [batch_size, 3, H, W]
        imgs_pred0_tensor = torch.stack(imgs_pred0_list)  # Shape: [batch_size, 3, H, W]

        return indices_tensor, total_imgs_tensor, total_imgs0_tensor, masks_tensor, imgs_tensor, imgs_pred_tensor, imgs_pred0_tensor

    def __len__(self):
        return len(self.files)
    
    def file_name(self,index):
        file_name = self.files[index]
        return file_name
    
    def get_angle(self,index):
        file_name = self.files[index]  
        #print(self.files[index])
        if file_name not in self.ann:
            print(f"Warning: {file_name} 没有对应的注释数据！")
            return None  

        # load camera parameters
        angle = self.ann[self.files[index]]['angle']
        return angle


# if __name__ == '__main__':
#     device = torch.device("cuda:2")
#     obj_name = './man/man.obj'
#     camou_mask = './man/mask.jpg'

#     # 设置渲染分辨率和其他参数
#     resolution = 8
#     expand_kernel = torch.nn.ConvTranspose2d(3, 3, resolution, stride=resolution, padding=0).to(device)
#     expand_kernel.weight.data.fill_(0)
#     expand_kernel.bias.data.fill_(0)
#     for i in range(3):
#         expand_kernel.weight[i, i, :, :].data.fill_(1)
#     #camou-设置
#     camou_para = np.load('./man_train/14_camou.npy')
#     # camou_para = np.ones_like(camou_para)*0.8
#     camou_para = torch.from_numpy(camou_para).to(device)
#     camou_para1 = expand_kernel(camou_para.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
#     camou_para1 = torch.clamp(camou_para1, 0, 1)


#     # 加载数据集
#     data_dir = './carla/dataset/'
#     img_size = (320, 1024)

#     dataset = HumanDataset(data_dir, img_size, obj_name, camou_mask, device=device, phy_trans_flag=True)
#     dataset.set_textures(camou_para1)
#     loader = DataLoader(
#         dataset=dataset,
#         batch_size=1,
#         shuffle=True,
#     )
#     os.makedirs("man_test", exist_ok=True)
#     log_dir = './man_test/'
#     tqdm_loader = tqdm(loader)
#     for i, (index, total_img, total_img0, mask, img, imgs_pred, imgs_pred0) in enumerate(tqdm_loader):
#         index = int(index[0])


#         total_img_np = total_img.data.cpu().numpy()[0] * 255
#         total_img_np = Image.fromarray(np.transpose(total_img_np, (1, 2, 0)).astype('uint8'))
#         total_img_np.save(os.path.join(log_dir, str(i) + 'test_total.jpg'))

#         total_img_np0 = total_img0.data.cpu().numpy()[0] * 255
#         total_img_np0 = Image.fromarray(np.transpose(total_img_np0, (1, 2, 0)).astype('uint8'))
#         total_img_np0.save(os.path.join(log_dir, str(i) + 'test_total0.jpg'))


#         Image.fromarray((255 * imgs_pred).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
#             os.path.join(log_dir, str(i) + 'img_pred.png'))
#         Image.fromarray((255 * imgs_pred0).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
#             os.path.join(log_dir, str(i) + 'img_pred0.png'))


#         Image.fromarray((255 * img).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
#             os.path.join(log_dir, str(i) + 'img.png'))
#         Image.fromarray((255 * mask).data.cpu().numpy()[0].transpose((1, 2, 0)).astype('uint8')).save(
#             os.path.join(log_dir, str(i) + 'mask.png'))

#         if i >= 11:
#             break
