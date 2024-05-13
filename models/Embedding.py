from collections import defaultdict

import os
import glob
import cv2
import os.path
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import ImageFile
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from IPython.display import display
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

from datasets.image_dataset import ImagesDataset, image_collate
from models.FeatureStyleEncoder import FSencoder
from models.Net import Net, get_segmentation
from models.encoder4editing.utils.model_utils import setup_model, get_latents
from utils.bicubic import BicubicDownSample
from utils.save_utils import save_gen_image, save_latents

from models.HairMapper.styleGAN2_ada_model.stylegan2_ada_generator import StyleGAN2adaGenerator
from models.HairMapper.classifier.src.feature_extractor.hair_mask_extractor import get_hair_mask, get_parsingNet
from models.HairMapper.mapper.networks.level_mapper import LevelMapper

class Embedding(nn.Module):
    """
    Module for image embedding
    """

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        if net is None:
            self.net = Net(self.opts)
        else:
            self.net = net

        self.encoder = FSencoder.get_trainer(self.opts.device)
        self.e4e, _ = setup_model('pretrained_models/encoder4editing/e4e_ffhq_encode.pt', self.opts.device)

        self.normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.to_bisenet = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.downsample_512 = BicubicDownSample(factor=2)
        self.downsample_256 = BicubicDownSample(factor=4)

    def setup_dataloader(self, images: dict[torch.Tensor, list[str]] | list[torch.Tensor], batch_size=None):
        self.dataset = ImagesDataset(images)
        self.dataloader = DataLoader(self.dataset, collate_fn=image_collate, shuffle=False,
                                     batch_size=batch_size or self.opts.batch_size)

    @torch.inference_mode()
    def get_e4e_embed(self, images: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        device = self.opts.device
        self.setup_dataloader(images, batch_size=len(images))

        for image, _ in self.dataloader:
            image = image.to(device)
            latent_W = get_latents(self.e4e, image)
            latent_F, _ = self.net.generator([latent_W], input_is_latent=True, return_latents=False,
                                             start_layer=0, end_layer=3)
            return {"F": latent_F, "W": latent_W}

    @torch.inference_mode()
    def embedding_images(self, images_to_name: dict[torch.Tensor, list[str]], **kwargs) -> dict[
        str, dict[str, torch.Tensor]]:
        device = self.opts.device
        self.setup_dataloader(images_to_name)

        name_to_embed = defaultdict(dict)
        for image, names in self.dataloader:
            image = image.to(device)

            im_512 = self.downsample_512(image)
            im_256 = self.downsample_256(image)
            im_256_norm = self.normalize(im_256)

            # E4E
            latent_W = get_latents(self.e4e, im_256_norm)

            # FS encoder
            output = self.encoder.test(img=self.normalize(image), return_latent=True)
            latent = output.pop()  # [bs, 512, 16, 16]
            latent_S = output.pop()  # [bs, 18, 512]

            latent_F, _ = self.net.generator([latent_S], input_is_latent=True, return_latents=False,
                                             start_layer=3, end_layer=3, layer_in=latent)  # [bs, 512, 32, 32]

            # BiSeNet
            masks = torch.cat([get_segmentation(image.unsqueeze(0)) for image in self.to_bisenet(im_512)])

            # Mixing if we change the color or shape
            if len(images_to_name) > 1:
                hair_mask = torch.where(masks == 13, torch.ones_like(masks, device=device),
                                        torch.zeros_like(masks, device=device))
                hair_mask = F.interpolate(hair_mask.float(), size=(32, 32), mode='bicubic')

                latent_F_from_W = self.net.generator([latent_W], input_is_latent=True, return_latents=False,
                                                     start_layer=0, end_layer=3)[0]
                latent_F = latent_F + self.opts.mixing * hair_mask * (latent_F_from_W - latent_F)

            for k, names in enumerate(names):
                for name in names:
                    name_to_embed[name]['W'] = latent_W[k].unsqueeze(0)
                    name_to_embed[name]['F'] = latent_F[k].unsqueeze(0)
                    name_to_embed[name]['S'] = latent_S[k].unsqueeze(0)
                    name_to_embed[name]['mask'] = masks[k].unsqueeze(0)
                    name_to_embed[name]['image_256'] = im_256[k].unsqueeze(0)
                    name_to_embed[name]['image_norm_256'] = im_256_norm[k].unsqueeze(0)

            if self.opts.save_all:
                gen_W_im, _ = self.net.generator([latent_W], input_is_latent=True, return_latents=False)
                gen_FS_im, _ = self.net.generator([latent_S], input_is_latent=True, return_latents=False,
                                                  start_layer=4, end_layer=8, layer_in=latent_F)

                exp_name = exp_name if (exp_name := kwargs.get('exp_name')) is not None else ""
                output_dir = self.opts.save_all_dir / exp_name
                for name, im_W, lat_W in zip(names, gen_W_im, latent_W):
                    save_gen_image(output_dir, 'W+', f'{name}.png', im_W)
                    save_latents(output_dir, 'W+', f'{name}.npz', latent_W=lat_W)

                for name, im_F, lat_S, lat_F in zip(names, gen_FS_im, latent_S, latent_F):
                    save_gen_image(output_dir, 'FS', f'{name}.png', im_F)
                    save_latents(output_dir, 'FS', f'{name}.npz', latent_S=lat_S, latent_F=lat_F)

        return name_to_embed



class EmbeddingBald(nn.Module):
    """
    Module for image embedding
    """

    def __init__(self, opts, net=None):
        super().__init__()
        self.opts = opts
        if net is None:
            self.net = Net(self.opts)
        else:
            self.net = net

        self.encoder = FSencoder.get_trainer(self.opts.device)
        self.e4e, _ = setup_model('pretrained_models/encoder4editing/e4e_ffhq_encode.pt', self.opts.device)

        self.normalize = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.to_bisenet = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.downsample_512 = BicubicDownSample(factor=2)
        self.downsample_256 = BicubicDownSample(factor=4)

    def setup_dataloader(self, images: dict[torch.Tensor, list[str]] | list[torch.Tensor], batch_size=None):
        self.dataset = ImagesDataset(images)
        self.dataloader = DataLoader(self.dataset, collate_fn=image_collate, shuffle=False,
                                     batch_size=batch_size or self.opts.batch_size)

    @torch.inference_mode()
    def get_e4e_embed(self, images: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        device = self.opts.device
        self.setup_dataloader(images, batch_size=len(images))

        for image, _ in self.dataloader:
            image = image.to(device)
            latent_W = get_latents(self.e4e, image)
            latent_F, _ = self.net.generator([latent_W], input_is_latent=True, return_latents=False,
                                             start_layer=0, end_layer=3)
            return {"F": latent_F, "W": latent_W}

    @torch.inference_mode()
    def embedding_images(self, images_to_name: dict[torch.Tensor, list[str]], **kwargs) -> dict[
        str, dict[str, torch.Tensor]]:
        device = self.opts.device
        self.setup_dataloader(images_to_name)

        name_to_embed = defaultdict(dict)
        for image, names in self.dataloader:
            image = image.to(device)

            print(f'image SHAPE = {image.shape}')
            img_transforms = T.Compose([
                    T.Resize((256, 256)),
                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
                    )
            # [C, W, H], float32
            start_img = img_transforms(image[0])

            # E4E
            latent_codes_origin = np.reshape(get_latents(self.e4e, start_img.unsqueeze(0))[0].cpu().numpy(), (1,18,512))

            # print(f'SHAPE = {latent_codes_origin.shape}')

            # balding
            model_name = 'stylegan2_ada'
            latent_space_type = 'wp'
            model = StyleGAN2adaGenerator(model_name, logger=None, truncation_psi=1.0)

            mapper = LevelMapper(input_dim=512).eval().cuda()
            ckpt = torch.load('./models/HairMapper/mapper/checkpoints/final/best_model.pt')
            alpha = float(ckpt['alpha']) * 1.2
            mapper.load_state_dict(ckpt['state_dict'], strict=True)
            kwargs = {'latent_space_type': latent_space_type}
            parsingNet = get_parsingNet(save_pth='./models/HairMapper/ckpts/face_parsing.pth')

            mapper_input = latent_codes_origin.copy()
            mapper_input_tensor = torch.from_numpy(mapper_input).cuda().float()
            edited_latent_codes = latent_codes_origin
            edited_latent_codes[:, :8, :] += alpha * mapper(mapper_input_tensor).to('cpu').detach().numpy()

            # origin_img = cv2.imread(origin_img_path)
            # print(f'start_img info <shape, type, dtype> {start_img.shape, type(start_img), start_img.dtype}')
            
            # [W, H, C], uint8
            # origin_img = np.transpose(np.array(((start_img + 1) / 2 * 255).to(torch.uint8).cpu()), axes=(1, 2, 0))
            origin_img = np.transpose(np.array((image[0] * 255).to(torch.uint8).cpu()), axes=(1, 2, 0))

            # Image.fromarray(origin_img).save('/home/bspanfilov/HairFlash/output/origin_img.jpg')
            # print(f'origin_img info {origin_img.shape, type(origin_img), origin_img.dtype}')

            outputs = model.easy_style_mixing(latent_codes=edited_latent_codes,
                                            style_range=range(7, 18),
                                            style_codes=latent_codes_origin,
                                            mix_ratio=0.8,
                                            **kwargs
                                            )

            edited_img = outputs['image'][0][:, :, ::-1][..., [2, 1, 0]]

            # print(f'edited_img info {edited_img.shape, type(edited_img), edited_img.dtype}')
            # Image.fromarray(edited_img).save('/home/bspanfilov/HairFlash/output/edited_img.jpg')

            # --remain_ear: preserve the ears in the original input image.
            hair_mask = get_hair_mask(img_path=origin_img, net=parsingNet, include_hat=True, include_ear=True)

            mask_dilate = cv2.dilate(hair_mask,
                                    kernel=np.ones((50, 50), np.uint8))
            mask_dilate_blur = cv2.blur(mask_dilate, ksize=(30, 30))
            mask_dilate_blur = (hair_mask + (255 - hair_mask) / 255 * mask_dilate_blur).astype(np.uint8)

            face_mask = 255 - mask_dilate_blur

            index = np.where(face_mask > 0)
            cy = (np.min(index[0]) + np.max(index[0])) // 2
            cx = (np.min(index[1]) + np.max(index[1])) // 2
            center = (cx, cy)

            bald_image = torch.tensor(cv2.seamlessClone(origin_img, edited_img, face_mask[:, :, 0], center, cv2.NORMAL_CLONE)).to(device)
            bald_image = bald_image.transpose(1, 2).transpose(0, 1) / 255

            # print(f'bald_image info <shape, type, dtype> {bald_image.shape, type(bald_image), bald_image.dtype}')
            # print(f'image[1] info <shape, type, dtype> {image[1].shape, type(image[1]), image[1].dtype}')

            # to_pil_image(bald_image).save('/home/bspanfilov/HairFlash/output/bald_face_image.jpg')
            # to_pil_image(image[1]).save('/home/bspanfilov/HairFlash/output/shape.jpg')
            # to_pil_image(image[2]).save('/home/bspanfilov/HairFlash/output/color.jpg')

            # image = torch.cat((bald_image.unsqueeze(0),
            #                    image[1].unsqueeze(0),
            #                    image[2].unsqueeze(0)))
            start_batch = image
            
            image = bald_image.unsqueeze(0)
            for i in range(1, len(start_batch)):
                image = torch.cat((image, start_batch[i].unsqueeze(0)))

            im_512 = self.downsample_512(image)
            im_256 = self.downsample_256(image)
            im_256_norm = self.normalize(im_256)

            # E4E
            latent_W = get_latents(self.e4e, im_256_norm)

            # FS encoder
            output = self.encoder.test(img=self.normalize(image), return_latent=True)
            latent = output.pop()  # [bs, 512, 16, 16]
            latent_S = output.pop()  # [bs, 18, 512]

            latent_F, _ = self.net.generator([latent_S], input_is_latent=True, return_latents=False,
                                             start_layer=3, end_layer=3, layer_in=latent)  # [bs, 512, 32, 32]

            # BiSeNet
            masks = torch.cat([get_segmentation(image.unsqueeze(0)) for image in self.to_bisenet(im_512)])

            # Mixing if we change the color or shape
            if len(images_to_name) > 1:
                hair_mask = torch.where(masks == 13, torch.ones_like(masks, device=device),
                                        torch.zeros_like(masks, device=device))
                hair_mask = F.interpolate(hair_mask.float(), size=(32, 32), mode='bicubic')

                latent_F_from_W = self.net.generator([latent_W], input_is_latent=True, return_latents=False,
                                                     start_layer=0, end_layer=3)[0]
                latent_F = latent_F + self.opts.mixing * hair_mask * (latent_F_from_W - latent_F)

            for k, names in enumerate(names):
                for name in names:
                    name_to_embed[name]['W'] = latent_W[k].unsqueeze(0)
                    name_to_embed[name]['F'] = latent_F[k].unsqueeze(0)
                    name_to_embed[name]['S'] = latent_S[k].unsqueeze(0)
                    name_to_embed[name]['mask'] = masks[k].unsqueeze(0)
                    name_to_embed[name]['image_256'] = im_256[k].unsqueeze(0)
                    name_to_embed[name]['image_norm_256'] = im_256_norm[k].unsqueeze(0)

            if self.opts.save_all:
                gen_W_im, _ = self.net.generator([latent_W], input_is_latent=True, return_latents=False)
                gen_FS_im, _ = self.net.generator([latent_S], input_is_latent=True, return_latents=False,
                                                  start_layer=4, end_layer=8, layer_in=latent_F)

                exp_name = exp_name if (exp_name := kwargs.get('exp_name')) is not None else ""
                output_dir = self.opts.save_all_dir / exp_name
                for name, im_W, lat_W in zip(names, gen_W_im, latent_W):
                    save_gen_image(output_dir, 'W+', f'{name}.png', im_W)
                    save_latents(output_dir, 'W+', f'{name}.npz', latent_W=lat_W)

                for name, im_F, lat_S, lat_F in zip(names, gen_FS_im, latent_S, latent_F):
                    save_gen_image(output_dir, 'FS', f'{name}.png', im_F)
                    save_latents(output_dir, 'FS', f'{name}.npz', latent_S=lat_S, latent_F=lat_F)

        return name_to_embed
