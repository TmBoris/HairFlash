import argparse
import typing as tp
from collections import defaultdict
from functools import wraps
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.io import read_image, ImageReadMode

from models.Alignment import Alignment
from models.Blending import Blending, Blending1WithoutColorMask, Blending2WithShapeS, Blending2WithShapeImg, Blending3WithBWShapeImg, Blendingv1S, Blendingv1ImgEmbeds, BlendingFinal
from models.Embedding import Embedding, EmbeddingBald
from models.Net import Net
from utils.image_utils import equal_replacer
from utils.seed import seed_setter
from utils.shape_predictor import align_face
from utils.time import bench_session

TImage = tp.TypeVar('TImage', torch.Tensor, Image.Image, np.ndarray)
TPath = tp.TypeVar('TPath', Path, str)
TReturn = tp.TypeVar('TReturn', torch.Tensor, tuple[torch.Tensor, ...])


class HairFlash:
    """
    HairFlash implementation with hairstyle transfer interface
    """

    def __init__(self, args):
        self.args = args
        self.net = Net(self.args)
        self.embed = Embedding(args, net=self.net)
        self.align = Alignment(args, net=self.net)
        self.blend = Blending(args, net=self.net)

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor,
                            **kwargs) -> torch.Tensor:
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ('face', 'shape', 'color')):
            images_to_name[image].append(name)

        # Embedding stage
        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)

        # Alignment stage
        align_shape = self.align.align_images('face', 'shape', self.embed.get_e4e_embed, name_to_embed, **kwargs)

        # Shape Module stage for blending
        if shape is not color:
            align_color = self.align.shape_module('face', 'color', name_to_embed, **kwargs)
        else:
            align_color = align_shape

        # Blending and Post Process stage
        final_image = self.blend.blend_images(align_shape, align_color, name_to_embed, **kwargs)
        return final_image

    def swap(self, face_img: TImage | TPath, shape_img: TImage | TPath, color_img: TImage | TPath,
             benchmark=False, align=False, seed=None, exp_name=None, **kwargs) -> TReturn:
        """
        Run HairFlash on the input images to transfer hair shape and color to the desired images.
        :param face_img:  face image in Tensor, PIL Image, array or file path format
        :param shape_img: shape image in Tensor, PIL Image, array or file path format
        :param color_img: color image in Tensor, PIL Image, array or file path format
        :param benchmark: starts counting the speed of the session
        :param align:     for arbitrary photos crops images to faces
        :param seed:      fixes seed for reproducibility, default 3407
        :param exp_name:  used as a folder name when 'save_all' model is enabled
        :return:          returns the final image as a Tensor
        """
        images: list[torch.Tensor] = []
        path_to_images: dict[TPath, torch.Tensor] = {}

        for img in (face_img, shape_img, color_img):
            if isinstance(img, (torch.Tensor, Image.Image, np.ndarray)):
                if not isinstance(img, torch.Tensor):
                    img = F.to_tensor(img)
            elif isinstance(img, (Path, str)):
                path_img = img
                if path_img not in path_to_images:
                    path_to_images[path_img] = read_image(str(path_img), mode=ImageReadMode.RGB)
                img = path_to_images[path_img]
            else:
                raise TypeError(f'Unsupported image format {type(img)}')

            images.append(img)

        if align:
            images = align_face(images)
        images = equal_replacer(images)

        final_image = self.__swap_from_tensors(*images, seed=seed, benchmark=benchmark, exp_name=exp_name, **kwargs)

        if align:
            return [final_image] + images
        return final_image

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)
    

class HairFlash1WithoutColorMask:
    """
    HairFlash implementation with hairstyle transfer interface
    """

    def __init__(self, args):
        self.args = args
        self.net = Net(self.args)
        self.embed = Embedding(args, net=self.net)
        self.align = Alignment(args, net=self.net)
        self.blend = Blending1WithoutColorMask(args, net=self.net)

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor,
                            **kwargs) -> torch.Tensor:
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ('face', 'shape', 'color')):
            images_to_name[image].append(name)

        # Embedding stage
        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)

        # Alignment stage
        align_shape = self.align.align_images('face', 'shape', self.embed.get_e4e_embed, name_to_embed, **kwargs)

        # Shape Module stage for blending
        # if shape is not color:
        #     align_color = self.align.shape_module('face', 'color', name_to_embed, **kwargs)
        # else:
        #     align_color = align_shape

        # Blending and Post Process stage
        final_image = self.blend.blend_images(align_shape, name_to_embed, **kwargs)
        return final_image

    def swap(self, face_img: TImage | TPath, shape_img: TImage | TPath, color_img: TImage | TPath,
             benchmark=False, align=False, seed=None, exp_name=None, **kwargs) -> TReturn:
        """
        Run HairFlash on the input images to transfer hair shape and color to the desired images.
        :param face_img:  face image in Tensor, PIL Image, array or file path format
        :param shape_img: shape image in Tensor, PIL Image, array or file path format
        :param color_img: color image in Tensor, PIL Image, array or file path format
        :param benchmark: starts counting the speed of the session
        :param align:     for arbitrary photos crops images to faces
        :param seed:      fixes seed for reproducibility, default 3407
        :param exp_name:  used as a folder name when 'save_all' model is enabled
        :return:          returns the final image as a Tensor
        """
        images: list[torch.Tensor] = []
        path_to_images: dict[TPath, torch.Tensor] = {}

        for img in (face_img, shape_img, color_img):
            if isinstance(img, (torch.Tensor, Image.Image, np.ndarray)):
                if not isinstance(img, torch.Tensor):
                    img = F.to_tensor(img)
            elif isinstance(img, (Path, str)):
                path_img = img
                if path_img not in path_to_images:
                    path_to_images[path_img] = read_image(str(path_img), mode=ImageReadMode.RGB)
                img = path_to_images[path_img]
            else:
                raise TypeError(f'Unsupported image format {type(img)}')

            images.append(img)

        if align:
            images = align_face(images)
        images = equal_replacer(images)

        final_image = self.__swap_from_tensors(*images, seed=seed, benchmark=benchmark, exp_name=exp_name, **kwargs)

        if align:
            return [final_image] + images
        return final_image

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)

class HairFlash2WihShapeS:
    """
    HairFlash implementation with hairstyle transfer interface
    """

    def __init__(self, args):
        self.args = args
        self.net = Net(self.args)
        self.embed = Embedding(args, net=self.net)
        self.align = Alignment(args, net=self.net)
        self.blend = Blending2WithShapeS(args, net=self.net)

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor,
                            **kwargs) -> torch.Tensor:
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ('face', 'shape', 'color')):
            images_to_name[image].append(name)

        # Embedding stage
        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)

        # Alignment stage
        align_shape = self.align.align_images('face', 'shape', self.embed.get_e4e_embed, name_to_embed, **kwargs)

        # Shape Module stage for blending
        if shape is not color:
            align_color = self.align.shape_module('face', 'color', name_to_embed, **kwargs)
        else:
            align_color = align_shape

        # Blending and Post Process stage
        final_image = self.blend.blend_images(align_shape, align_color, name_to_embed, **kwargs)
        return final_image

    def swap(self, face_img: TImage | TPath, shape_img: TImage | TPath, color_img: TImage | TPath,
             benchmark=False, align=False, seed=None, exp_name=None, **kwargs) -> TReturn:
        """
        Run HairFlash on the input images to transfer hair shape and color to the desired images.
        :param face_img:  face image in Tensor, PIL Image, array or file path format
        :param shape_img: shape image in Tensor, PIL Image, array or file path format
        :param color_img: color image in Tensor, PIL Image, array or file path format
        :param benchmark: starts counting the speed of the session
        :param align:     for arbitrary photos crops images to faces
        :param seed:      fixes seed for reproducibility, default 3407
        :param exp_name:  used as a folder name when 'save_all' model is enabled
        :return:          returns the final image as a Tensor
        """
        images: list[torch.Tensor] = []
        path_to_images: dict[TPath, torch.Tensor] = {}

        for img in (face_img, shape_img, color_img):
            if isinstance(img, (torch.Tensor, Image.Image, np.ndarray)):
                if not isinstance(img, torch.Tensor):
                    img = F.to_tensor(img)
            elif isinstance(img, (Path, str)):
                path_img = img
                if path_img not in path_to_images:
                    path_to_images[path_img] = read_image(str(path_img), mode=ImageReadMode.RGB)
                img = path_to_images[path_img]
            else:
                raise TypeError(f'Unsupported image format {type(img)}')

            images.append(img)

        if align:
            images = align_face(images)
        images = equal_replacer(images)

        final_image = self.__swap_from_tensors(*images, seed=seed, benchmark=benchmark, exp_name=exp_name, **kwargs)

        if align:
            return [final_image] + images
        return final_image

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)
    

class HairFlash2WihShapeImg:
    """
    HairFlash implementation with hairstyle transfer interface
    """

    def __init__(self, args):
        self.args = args
        self.net = Net(self.args)
        self.embed = Embedding(args, net=self.net)
        self.align = Alignment(args, net=self.net)
        self.blend = Blending2WithShapeImg(args, net=self.net)

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor,
                            **kwargs) -> torch.Tensor:
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ('face', 'shape', 'color')):
            images_to_name[image].append(name)

        # Embedding stage
        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)

        # Alignment stage
        align_shape = self.align.align_images('face', 'shape', self.embed.get_e4e_embed, name_to_embed, **kwargs)

        # Shape Module stage for blending
        if shape is not color:
            align_color = self.align.shape_module('face', 'color', name_to_embed, **kwargs)
        else:
            align_color = align_shape

        # Blending and Post Process stage
        final_image = self.blend.blend_images(align_shape, align_color, name_to_embed, **kwargs)
        return final_image

    def swap(self, face_img: TImage | TPath, shape_img: TImage | TPath, color_img: TImage | TPath,
             benchmark=False, align=False, seed=None, exp_name=None, **kwargs) -> TReturn:
        """
        Run HairFlash on the input images to transfer hair shape and color to the desired images.
        :param face_img:  face image in Tensor, PIL Image, array or file path format
        :param shape_img: shape image in Tensor, PIL Image, array or file path format
        :param color_img: color image in Tensor, PIL Image, array or file path format
        :param benchmark: starts counting the speed of the session
        :param align:     for arbitrary photos crops images to faces
        :param seed:      fixes seed for reproducibility, default 3407
        :param exp_name:  used as a folder name when 'save_all' model is enabled
        :return:          returns the final image as a Tensor
        """
        images: list[torch.Tensor] = []
        path_to_images: dict[TPath, torch.Tensor] = {}

        for img in (face_img, shape_img, color_img):
            if isinstance(img, (torch.Tensor, Image.Image, np.ndarray)):
                if not isinstance(img, torch.Tensor):
                    img = F.to_tensor(img)
            elif isinstance(img, (Path, str)):
                path_img = img
                if path_img not in path_to_images:
                    path_to_images[path_img] = read_image(str(path_img), mode=ImageReadMode.RGB)
                img = path_to_images[path_img]
            else:
                raise TypeError(f'Unsupported image format {type(img)}')

            images.append(img)

        if align:
            images = align_face(images)
        images = equal_replacer(images)

        final_image = self.__swap_from_tensors(*images, seed=seed, benchmark=benchmark, exp_name=exp_name, **kwargs)

        if align:
            return [final_image] + images
        return final_image

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)


class HairFlash2WihBWShapeImg:
    """
    HairFlash implementation with hairstyle transfer interface
    """

    def __init__(self, args):
        self.args = args
        self.net = Net(self.args)
        self.embed = Embedding(args, net=self.net)
        self.align = Alignment(args, net=self.net)
        self.blend = Blending3WithBWShapeImg(args, net=self.net)

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor,
                            **kwargs) -> torch.Tensor:
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ('face', 'shape', 'color')):
            images_to_name[image].append(name)

        # Embedding stage
        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)

        # Alignment stage
        align_shape = self.align.align_images('face', 'shape', self.embed.get_e4e_embed, name_to_embed, **kwargs)

        # Shape Module stage for blending
        if shape is not color:
            align_color = self.align.shape_module('face', 'color', name_to_embed, **kwargs)
        else:
            align_color = align_shape

        # Blending and Post Process stage
        final_image = self.blend.blend_images(align_shape, align_color, name_to_embed, **kwargs)
        return final_image

    def swap(self, face_img: TImage | TPath, shape_img: TImage | TPath, color_img: TImage | TPath,
             benchmark=False, align=False, seed=None, exp_name=None, **kwargs) -> TReturn:
        """
        Run HairFlash on the input images to transfer hair shape and color to the desired images.
        :param face_img:  face image in Tensor, PIL Image, array or file path format
        :param shape_img: shape image in Tensor, PIL Image, array or file path format
        :param color_img: color image in Tensor, PIL Image, array or file path format
        :param benchmark: starts counting the speed of the session
        :param align:     for arbitrary photos crops images to faces
        :param seed:      fixes seed for reproducibility, default 3407
        :param exp_name:  used as a folder name when 'save_all' model is enabled
        :return:          returns the final image as a Tensor
        """
        images: list[torch.Tensor] = []
        path_to_images: dict[TPath, torch.Tensor] = {}

        for img in (face_img, shape_img, color_img):
            if isinstance(img, (torch.Tensor, Image.Image, np.ndarray)):
                if not isinstance(img, torch.Tensor):
                    img = F.to_tensor(img)
            elif isinstance(img, (Path, str)):
                path_img = img
                if path_img not in path_to_images:
                    path_to_images[path_img] = read_image(str(path_img), mode=ImageReadMode.RGB)
                img = path_to_images[path_img]
            else:
                raise TypeError(f'Unsupported image format {type(img)}')

            images.append(img)

        if align:
            images = align_face(images)
        images = equal_replacer(images)

        final_image = self.__swap_from_tensors(*images, seed=seed, benchmark=benchmark, exp_name=exp_name, **kwargs)

        if align:
            return [final_image] + images
        return final_image

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)
    

class HairFlashBald:
    """
    HairFlash implementation with hairstyle transfer interface
    """

    def __init__(self, args):
        self.args = args
        self.net = Net(self.args)
        self.embed = EmbeddingBald(args, net=self.net)
        self.align = Alignment(args, net=self.net)
        self.blend = Blending(args, net=self.net)

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor,
                            **kwargs) -> torch.Tensor:
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ('face', 'shape', 'color')):
            images_to_name[image].append(name)

        # Embedding stage
        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)

        # Alignment stage
        align_shape = self.align.align_images('face', 'shape', self.embed.get_e4e_embed, name_to_embed, **kwargs)

        # Shape Module stage for blending
        if shape is not color:
            align_color = self.align.shape_module('face', 'color', name_to_embed, **kwargs)
        else:
            align_color = align_shape

        # Blending and Post Process stage
        final_image = self.blend.blend_images(align_shape, align_color, name_to_embed, **kwargs)
        return final_image

    def swap(self, face_img: TImage | TPath, shape_img: TImage | TPath, color_img: TImage | TPath,
             benchmark=False, align=False, seed=None, exp_name=None, **kwargs) -> TReturn:
        """
        Run HairFlash on the input images to transfer hair shape and color to the desired images.
        :param face_img:  face image in Tensor, PIL Image, array or file path format
        :param shape_img: shape image in Tensor, PIL Image, array or file path format
        :param color_img: color image in Tensor, PIL Image, array or file path format
        :param benchmark: starts counting the speed of the session
        :param align:     for arbitrary photos crops images to faces
        :param seed:      fixes seed for reproducibility, default 3407
        :param exp_name:  used as a folder name when 'save_all' model is enabled
        :return:          returns the final image as a Tensor
        """
        images: list[torch.Tensor] = []
        path_to_images: dict[TPath, torch.Tensor] = {}

        for img in (face_img, shape_img, color_img):
            if isinstance(img, (torch.Tensor, Image.Image, np.ndarray)):
                if not isinstance(img, torch.Tensor):
                    img = F.to_tensor(img)
            elif isinstance(img, (Path, str)):
                path_img = img
                if path_img not in path_to_images:
                    path_to_images[path_img] = read_image(str(path_img), mode=ImageReadMode.RGB)
                img = path_to_images[path_img]
            else:
                raise TypeError(f'Unsupported image format {type(img)}')

            images.append(img)

        if align:
            images = align_face(images)
        images = equal_replacer(images)

        final_image = self.__swap_from_tensors(*images, seed=seed, benchmark=benchmark, exp_name=exp_name, **kwargs)

        if align:
            return [final_image] + images
        return final_image

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)
    

class HairFlashv1S:
    """
    HairFlash implementation with hairstyle transfer interface
    """

    def __init__(self, args):
        self.args = args
        self.net = Net(self.args)
        self.embed = Embedding(args, net=self.net)
        self.align = Alignment(args, net=self.net)
        self.blend = Blendingv1S(args, net=self.net)

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor,
                            **kwargs) -> torch.Tensor:
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ('face', 'shape', 'color')):
            images_to_name[image].append(name)

        # Embedding stage
        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)

        # Alignment stage
        align_shape = self.align.align_images('face', 'shape', self.embed.get_e4e_embed, name_to_embed, **kwargs)

        # Shape Module stage for blending
        if shape is not color:
            align_color = self.align.shape_module('face', 'color', name_to_embed, **kwargs)
        else:
            align_color = align_shape

        # Blending and Post Process stage
        final_image = self.blend.blend_images(align_shape, align_color, name_to_embed, **kwargs)
        return final_image

    def swap(self, face_img: TImage | TPath, shape_img: TImage | TPath, color_img: TImage | TPath,
             benchmark=False, align=False, seed=None, exp_name=None, **kwargs) -> TReturn:
        """
        Run HairFlash on the input images to transfer hair shape and color to the desired images.
        :param face_img:  face image in Tensor, PIL Image, array or file path format
        :param shape_img: shape image in Tensor, PIL Image, array or file path format
        :param color_img: color image in Tensor, PIL Image, array or file path format
        :param benchmark: starts counting the speed of the session
        :param align:     for arbitrary photos crops images to faces
        :param seed:      fixes seed for reproducibility, default 3407
        :param exp_name:  used as a folder name when 'save_all' model is enabled
        :return:          returns the final image as a Tensor
        """
        images: list[torch.Tensor] = []
        path_to_images: dict[TPath, torch.Tensor] = {}

        for img in (face_img, shape_img, color_img):
            if isinstance(img, (torch.Tensor, Image.Image, np.ndarray)):
                if not isinstance(img, torch.Tensor):
                    img = F.to_tensor(img)
            elif isinstance(img, (Path, str)):
                path_img = img
                if path_img not in path_to_images:
                    path_to_images[path_img] = read_image(str(path_img), mode=ImageReadMode.RGB)
                img = path_to_images[path_img]
            else:
                raise TypeError(f'Unsupported image format {type(img)}')

            images.append(img)

        if align:
            images = align_face(images)
        images = equal_replacer(images)

        final_image = self.__swap_from_tensors(*images, seed=seed, benchmark=benchmark, exp_name=exp_name, **kwargs)

        if align:
            return [final_image] + images
        return final_image

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)
    

class HairFlashv1ImgEmbeds:
    """
    HairFlash implementation with hairstyle transfer interface
    """

    def __init__(self, args):
        self.args = args
        self.net = Net(self.args)
        self.embed = Embedding(args, net=self.net)
        self.align = Alignment(args, net=self.net)
        self.blend = Blendingv1ImgEmbeds(args, net=self.net)

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor,
                            **kwargs) -> torch.Tensor:
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ('face', 'shape', 'color')):
            images_to_name[image].append(name)

        # Embedding stage
        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)

        # Alignment stage
        align_shape = self.align.align_images('face', 'shape', self.embed.get_e4e_embed, name_to_embed, **kwargs)

        # Shape Module stage for blending
        if shape is not color:
            align_color = self.align.shape_module('face', 'color', name_to_embed, **kwargs)
        else:
            align_color = align_shape

        # Blending and Post Process stage
        final_image = self.blend.blend_images(align_shape, align_color, name_to_embed, **kwargs)
        return final_image

    def swap(self, face_img: TImage | TPath, shape_img: TImage | TPath, color_img: TImage | TPath,
             benchmark=False, align=False, seed=None, exp_name=None, **kwargs) -> TReturn:
        """
        Run HairFlash on the input images to transfer hair shape and color to the desired images.
        :param face_img:  face image in Tensor, PIL Image, array or file path format
        :param shape_img: shape image in Tensor, PIL Image, array or file path format
        :param color_img: color image in Tensor, PIL Image, array or file path format
        :param benchmark: starts counting the speed of the session
        :param align:     for arbitrary photos crops images to faces
        :param seed:      fixes seed for reproducibility, default 3407
        :param exp_name:  used as a folder name when 'save_all' model is enabled
        :return:          returns the final image as a Tensor
        """
        images: list[torch.Tensor] = []
        path_to_images: dict[TPath, torch.Tensor] = {}

        for img in (face_img, shape_img, color_img):
            if isinstance(img, (torch.Tensor, Image.Image, np.ndarray)):
                if not isinstance(img, torch.Tensor):
                    img = F.to_tensor(img)
            elif isinstance(img, (Path, str)):
                path_img = img
                if path_img not in path_to_images:
                    path_to_images[path_img] = read_image(str(path_img), mode=ImageReadMode.RGB)
                img = path_to_images[path_img]
            else:
                raise TypeError(f'Unsupported image format {type(img)}')

            images.append(img)

        if align:
            images = align_face(images)
        images = equal_replacer(images)

        final_image = self.__swap_from_tensors(*images, seed=seed, benchmark=benchmark, exp_name=exp_name, **kwargs)

        if align:
            return [final_image] + images
        return final_image

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)
    

class HairFlashFinal:
    """
    HairFlash implementation with hairstyle transfer interface
    """

    def __init__(self, args):
        self.args = args
        self.net = Net(self.args)
        self.embed = Embedding(args, net=self.net)
        self.align = Alignment(args, net=self.net)
        self.blend = BlendingFinal(args, net=self.net)

    @seed_setter
    @bench_session
    def __swap_from_tensors(self, face: torch.Tensor, shape: torch.Tensor, color: torch.Tensor,
                            **kwargs) -> torch.Tensor:
        images_to_name = defaultdict(list)
        for image, name in zip((face, shape, color), ('face', 'shape', 'color')):
            images_to_name[image].append(name)

        # Embedding stage
        name_to_embed = self.embed.embedding_images(images_to_name, **kwargs)

        # Alignment stage
        align_shape = self.align.align_images('face', 'shape', self.embed.get_e4e_embed, name_to_embed, **kwargs)

        # Shape Module stage for blending
        # if shape is not color:
        #     align_color = self.align.shape_module('face', 'color', name_to_embed, **kwargs)
        # else:
        #     align_color = align_shape

        # Blending and Post Process stage
        final_image = self.blend.blend_images(align_shape, name_to_embed, **kwargs)
        return final_image

    def swap(self, face_img: TImage | TPath, shape_img: TImage | TPath, color_img: TImage | TPath,
             benchmark=False, align=False, seed=None, exp_name=None, **kwargs) -> TReturn:
        """
        Run HairFlash on the input images to transfer hair shape and color to the desired images.
        :param face_img:  face image in Tensor, PIL Image, array or file path format
        :param shape_img: shape image in Tensor, PIL Image, array or file path format
        :param color_img: color image in Tensor, PIL Image, array or file path format
        :param benchmark: starts counting the speed of the session
        :param align:     for arbitrary photos crops images to faces
        :param seed:      fixes seed for reproducibility, default 3407
        :param exp_name:  used as a folder name when 'save_all' model is enabled
        :return:          returns the final image as a Tensor
        """
        images: list[torch.Tensor] = []
        path_to_images: dict[TPath, torch.Tensor] = {}

        for img in (face_img, shape_img, color_img):
            if isinstance(img, (torch.Tensor, Image.Image, np.ndarray)):
                if not isinstance(img, torch.Tensor):
                    img = F.to_tensor(img)
            elif isinstance(img, (Path, str)):
                path_img = img
                if path_img not in path_to_images:
                    path_to_images[path_img] = read_image(str(path_img), mode=ImageReadMode.RGB)
                img = path_to_images[path_img]
            else:
                raise TypeError(f'Unsupported image format {type(img)}')

            images.append(img)

        if align:
            images = align_face(images)
        images = equal_replacer(images)

        final_image = self.__swap_from_tensors(*images, seed=seed, benchmark=benchmark, exp_name=exp_name, **kwargs)

        if align:
            return [final_image] + images
        return final_image

    @wraps(swap)
    def __call__(self, *args, **kwargs):
        return self.swap(*args, **kwargs)


def get_parser():
    parser = argparse.ArgumentParser(description='HairFlash')

    # I/O arguments
    parser.add_argument('--save_all_dir', type=Path, default=Path('output'),
                        help='the directory to save the latent codes and inversion images')

    # StyleGAN2 setting
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--ckpt', type=str, default="pretrained_models/StyleGAN/ffhq.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--n_mlp', type=int, default=8)

    # Arguments
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=3, help='batch size for encoding images')
    parser.add_argument('--save_all', action='store_true', help='save and print mode information')

    # HairFlash setting
    parser.add_argument('--mixing', type=float, default=0.95, help='hair blending in alignment')
    parser.add_argument('--smooth', type=int, default=5, help='dilation and erosion parameter')
    parser.add_argument('--rotate_checkpoint', type=str, default='pretrained_models/Rotate/rotate_best.pth')
    parser.add_argument('--blending_checkpoint', type=str, default='pretrained_models/Blending/clip_loss.pth')
    parser.add_argument('--pp_checkpoint', type=str, default='pretrained_models/PostProcess/pp_model.pth')
    return parser
