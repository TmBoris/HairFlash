import argparse
import os
import sys
from argparse import Namespace
from tempfile import TemporaryDirectory

from models.stylegan2 import dnnlib
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms as T
from tqdm.auto import tqdm

from models.Encoders import ClipBlendingModelWithShapeImgEmbds as BlendingModel
from models.Encoders import ClipModel
from models.Net import Net
from models.face_parsing.model import BiSeNet, seg_mean, seg_std
from utils.bicubic import BicubicDownSample
from utils.train import toggle_grad, WandbLogger, image_grid
from utils.image_utils import DilateErosion

parser = argparse.ArgumentParser(description='Blending trainer')
parser.add_argument('--name_run', type=str, default='test')
args = parser.parse_args()


w_dilate_erosion = torch.Tensor([
    [False, True, False],
    [True, True, True],
    [False, True, False]
]).float()[None, None, ...].cuda()


def dilate_erosion_mask_tensor_gpu(mask, weight, dilate_erosion=5):
    masks = mask.clone().repeat(*([2] + [1] * (len(mask.shape) - 1))).float()
    sum_w = weight.sum().item()
    n = len(mask)

    for _ in range(dilate_erosion):
        masks = F.conv2d(masks, weight,
                         bias=None, stride=1, padding='same', dilation=1, groups=1)
        masks[:n] = (masks[:n] > 0)
        masks[n:] = (masks[n:] == sum_w)

    hair_mask_dilate, hair_mask_erode = masks[:n], masks[n:]

    return hair_mask_dilate, hair_mask_erode


def dilate_erosion_mask_path_gpu(mask, weight, dilate_erosion=5):
    dialate, erosion = dilate_erosion_mask_tensor_gpu(mask, weight, dilate_erosion)
    return dialate, erosion


class Trainer:
    def __init__(self,
                 model=None,
                 optimizer=None,
                 scheduler=None,
                 train_dataloader=None,
                 test_dataloader=None,
                 logger=None
                 ):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        if model is not None:
            self.model.to(self.device)
            self.model = torch.nn.DataParallel(self.model)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.dilate_erosion = DilateErosion(device=self.device)

        self.net = Net(Namespace(size=1024, ckpt='pretrained_models/ffhq.pt', channel_multiplier=2, latent=512, n_mlp=8,
                                 device=self.device)).generator.to(self.device)
        self.net = torch.nn.DataParallel(self.net)
        self.seg = BiSeNet(n_classes=16).to(self.device)
        # self.seg.to(self.device)
        self.seg = torch.nn.DataParallel(self.seg)

        self.seg.module.load_state_dict(torch.load('pretrained_models/seg.pth'))
        for param in self.seg.module.parameters():
            param.requires_grad = False
        self.seg.module.eval()
        toggle_grad(self.net.module, False)

        self.downsample_512 = BicubicDownSample(factor=2)
        self.downsample_256 = BicubicDownSample(factor=4)
        self.downsample_128 = BicubicDownSample(factor=8)

        self.best_loss = float('+inf')
        self.cur_iter = 0

    @torch.no_grad()
    def generate_mask(self, I):
        IM = (self.downsample_512((I + 1) / 2) - seg_mean) / seg_std
        down_seg, _, _ = self.seg(IM)
        current_mask = torch.argmax(down_seg, dim=1).long().float()
        HM_X = torch.where(current_mask == 10, torch.ones_like(current_mask), torch.zeros_like(current_mask))
        HM_X = F.interpolate(HM_X.unsqueeze(1), size=(256, 256), mode='nearest')

        HM_XD, HM_XE = dilate_erosion_mask_path_gpu(HM_X, w_dilate_erosion)
        return HM_XD, HM_XE

    def save_model(self, name, save_online=True):
        with TemporaryDirectory() as tmp_dir:
            torch.save({'model_state_dict': self.model.module.state_dict()}, f'{tmp_dir}/{name}.pth')
            self.logger.save(f'{tmp_dir}/{name}.pth', save_online)

    def calc_loss(self, I_gen, I_face, I_color, mask_face, mask_hair, gen_hair):
        gen_embed = self.model.module.get_image_embed(I_gen * mask_face)
        gt_embed = self.model.module.get_image_embed(I_face * mask_face)
        face_loss = (1 - F.cosine_similarity(gen_embed, gt_embed)).mean()

        gen_embed = self.model.module.get_image_embed(I_gen * gen_hair)
        gt_embed = self.model.module.get_image_embed(I_color * mask_hair)
        hair_loss = (1 - F.cosine_similarity(gen_embed, gt_embed)).mean()

        losses = {'face loss': face_loss, 'hair loss': hair_loss, 'loss': face_loss + hair_loss}
        return losses['loss'], losses

    def train_one_epoch(self):
        self.model.module.train()
        for batch in tqdm(self.train_dataloader):
            color_s, align_s, align_f, color_i, shape_i, face_i, target_mask, HM_2E, HM_3E, HM_XE = map(lambda x: x.to(self.device),
                                                                                                        batch)
            bsz = color_s.size(0)

            blend_s = self.model(align_s[:, 6:], color_s[:, 6:], face_i * target_mask, color_i * HM_3E, shape_i * HM_2E)
            latent_in = torch.cat((torch.zeros(bsz, 6, 512).to(self.device), blend_s), axis=1)
            I_G, _ = self.net([latent_in], input_is_latent=True, return_latents=False, start_layer=4,
                                        end_layer=8, layer_in=align_f)

            loss, info = self.calc_loss(self.downsample_256(I_G), face_i, color_i, target_mask, HM_3E, HM_XE)

            self.optimizer.zero_grad()
            loss.backward()

            total_norm = torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), 5)
            self.optimizer.step()

            self.logger.next_step()
            for key, val in info.items():
                self.logger.log(f'scripts {key}', val.item())
            self.logger.log('scripts grad', total_norm.item())
            self.cur_iter += 1

    @torch.no_grad()
    def validate(self):
        self.model.module.eval()

        sum_losses = lambda x, y: {key: val + x.get(key, 0) for key, val in y.items()}
        files = []
        losses = {}
        to_299 = T.Resize((299, 299))
        images_to_fid = []

        for batch in tqdm(self.test_dataloader):
            color_s, align_s, align_f, color_i, shape_i, face_i, target_mask, HM_2E, HM_3E, HM_XE = map(lambda x: x.to(self.device),
                                                                                                        batch)
            bsz = color_s.size(0)

            blend_s = self.model(align_s[:, 6:], color_s[:, 6:], face_i * target_mask, color_i * HM_3E, shape_i * HM_2E)
            latent_in = torch.cat((torch.zeros(bsz, 6, 512, device=self.device), blend_s), axis=1)
            I_G, _ = self.net([latent_in], input_is_latent=True, return_latents=False, start_layer=4,
                                        end_layer=8, layer_in=align_f)

            _, info = self.calc_loss(self.downsample_256(I_G), face_i, color_i, target_mask, HM_3E, HM_XE)
            losses = sum_losses(losses, info)
            for k in range(bsz):
                files.append([color_i[k].cpu(), face_i[k].cpu(), self.downsample_256(I_G)[k].cpu()])

            images_to_fid.append(to_299((I_G + 1) / 2).clip(0, 1))

        losses['FID CLIP'] = compute_fid_datasets(torch.cat(images_to_fid), CELEBA, CLIP=True)
        for key, val in losses.items():
            if key != 'FID CLIP':
                val = val.item() / len(self.test_dataloader)
            self.logger.log(f'val {key}', val)

        np.random.seed(1927)
        idxs = np.random.choice(len(files), size=100, replace=False)
        images_to_log = [
            image_grid([T.functional.to_pil_image(((img + 1) / 2).clamp(0, 1)) for img in files[idx]], 1, 3) for idx in
            idxs]
        self.logger.log('val images', [wandb.Image(image) for image in images_to_log])

        return losses['loss']

    def train_loop(self, epochs):
        self.validate()
        for epoch in range(epochs):
            self.train_one_epoch()
            loss = self.validate()

            self.save_model('last', save_online=False)
            if loss <= self.best_loss:
                self.best_loss = loss
                self.save_model(f'best', save_online=False)


net_trainer = Trainer()


def load_images_to_torch(paths, imgs=None, use_tqdm=True):
    transform = T.PILToTensor()
    tensor = []
    for path in paths:
        if imgs is None:
            pbar = sorted(os.listdir(path))
        else:
            pbar = imgs

        if use_tqdm:
            pbar = tqdm(pbar)

        for img_name in pbar:
            if '.jpg' in img_name or '.png' in img_name:
                img_path = os.path.join(path, img_name)
                img = Image.open(img_path).resize((299, 299), resample=Image.LANCZOS)
                # if np.array(img).mean() == 0:
                #     continue
                tensor.append(transform(img))
    try:
        return torch.stack(tensor)
    except:
        print(paths, imgs)
        return torch.tensor([], dtype=torch.uint8)


def parallel_load_images(paths, imgs):
    assert imgs is not None

    list_torch_images = Parallel(n_jobs=-1)(delayed(load_images_to_torch)(
        paths, [i], use_tqdm=False) for i in tqdm(imgs)
                                            )
    return torch.cat(list_torch_images)


@torch.inference_mode()
def compute_fid_datasets(fake, target, device=torch.device('cuda'), CLIP=False):
    result = {}

    fake_dataloader = DataLoader(TensorDataset(fake), batch_size=128)
    real_dataloader = DataLoader(TensorDataset(target), batch_size=128)
    background = True
    if CLIP:
        fid = FrechetInceptionDistance(feature=ClipModel(), normalize=True)
    else:
        fid = FrechetInceptionDistance(normalize=True)
    fid.reset()
    fid.to(device).eval()

    with torch.inference_mode():
        for batch in tqdm(fake_dataloader):
            batch = batch[0].to(device)
            fid.update(batch, real=False)

        for batch in tqdm(real_dataloader):
            batch = batch[0].to(device)
            fid.update(batch, real=True)

    return fid.compute()


CELEBA_PATH = "/home/aalanov/Nikolaev_Maxim/CelebA-HQ-img/"
celeba_imgs = []
for file in os.listdir(CELEBA_PATH):
    if 'flip' not in file:
        celeba_imgs.append(file)

CELEBA = parallel_load_images([CELEBA_PATH], celeba_imgs) / 255

FFHQ_PATH = "/home/aalanov/Bobkov_Denis/datasets/FFHQ"
BIG_PATH = "/home/aalanov/Nikolaev_Maxim/HairSwap_fast_for_Maxim/CelebA/blending_train/"
BIG_SUFFIX = '_blender_rotate.npz'
SMALL_PATH = '/home/aalanov/Nikolaev_Maxim/shared/blending_train/'
SMALL_SUFFIX = '_SEAN.npz'
PATH = SMALL_PATH
SUFFIX = SMALL_SUFFIX


def prepare_item(exp, path, files_suffix=SUFFIX):
    im1, im2, im3 = exp

    try:
        color_path = os.path.join(path, 'FS', im3 + files_suffix)
        Color_S = torch.from_numpy(np.load(color_path)['latent_in']).squeeze(0)
        Color_I = T.functional.normalize(T.functional.to_tensor(
            Image.open(os.path.join(FFHQ_PATH, f'{im3}.png'))
        ), [0.5], [0.5])
        Shape_I = T.functional.normalize(T.functional.to_tensor(
            Image.open(os.path.join(FFHQ_PATH, f'{im2}.png'))
        ), [0.5], [0.5])
        Face_I = T.functional.normalize(T.functional.to_tensor(
            Image.open(os.path.join(FFHQ_PATH, f'{im1}.png'))
        ), [0.5], [0.5])

        align_path = os.path.join(path, 'Align_realistic')
        data = np.load(
            os.path.join(align_path, f'{im1}_{im3}{files_suffix}')
        )
        Align_S = torch.from_numpy(data['latent_in']).squeeze(0)
        Align_F = torch.from_numpy(data['latent_F']).squeeze(0)

        return (Color_S, Align_S, Align_F, Color_I, Face_I, Shape_I)
    except:
        return None


class Blending_dataset(Dataset):
    def __init__(self, exps, path):
        super().__init__()
        downsample_256 = BicubicDownSample(factor=4)
        data = Parallel(n_jobs=-1)(
            delayed(prepare_item)(exp, path) for (p1, p2, p3) in tqdm(exps) for exp in [(p1, p2, p3), (p1, p3, p2)])
        data = [elem for elem in data if elem is not None]
        print(f'Load: {len(data)}/{2 * len(exps)}', file=sys.stderr)

        tmp_dataloader = DataLoader(data, batch_size=24, pin_memory=False, shuffle=False)

        self.items = []
        with torch.no_grad():
            for (Color_S, Align_S, Align_F, Color_I, Face_I, Shape_I) in tqdm(tmp_dataloader):
                HM_3D, HM_3E = net_trainer.generate_mask(Color_I.to('cuda'))
                _, HM_2E = net_trainer.generate_mask(Shape_I.to('cuda'))

                HM_1D, _ = net_trainer.generate_mask(Face_I.to('cuda'))
                I_X, _ = net_trainer.net([Align_S.to('cuda')], input_is_latent=True, return_latents=False,
                                                   start_layer=4,
                                                   end_layer=8, layer_in=Align_F.to('cuda'))
                _, HM_XE = net_trainer.generate_mask(I_X)

                target_mask = ((1 - HM_1D) * (1 - HM_3D)).cpu()
                HM_3E = HM_3E.cpu()
                self.items.extend(
                    [item for item in zip(*list(map(lambda x: [item.squeeze(0) for item in torch.split(x, 1)],
                                                    (Color_S,
                                                     Align_S,
                                                     Align_F,
                                                     downsample_256(Color_I.to('cuda')).cpu(),
                                                     downsample_256(Shape_I.to('cuda')).cpu(),
                                                     downsample_256(Face_I.to('cuda')).cpu(),
                                                     target_mask, HM_2E, HM_3E, HM_XE)))
                                          ) if item[-2].any() and item[-1].any()]
                )

        print(f'dataset: {len(self.items)}/{len(data)}', file=sys.stderr)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

exps = []
with open(os.path.join(PATH, 'dataset.exps'), 'r') as file:
    for exp in file.readlines():
        exps.append(list(map(lambda x: x.replace('.png', ''), exp.split())))

from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(exps, test_size=512, random_state=42)


def main():
    train_dataset = Blending_dataset(X_train, PATH)
    test_dataset = Blending_dataset(X_test, PATH)

    train_dataloader = DataLoader(train_dataset, batch_size=84, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=84, shuffle=False)

    logger = WandbLogger(name=args.name_run, project='Barbershop-Blending')
    logger.start_logging()
    logger.save(__file__)

    model = BlendingModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.000001)

    trainer = Trainer(model, optimizer, None, train_dataloader, test_dataloader, logger)

    trainer.train_loop(250)

    logger.wandb.finish()


if __name__ == '__main__':
    main()
