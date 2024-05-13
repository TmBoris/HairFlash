import os
import pickle
import shutil

import wandb
from PIL import Image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


class WandbLogger:
    def __init__(self, name='base-name', project='HairFlash'):
        self.name = name
        self.project = project

    def start_logging(self):
        wandb.login(key=os.environ['WANDB_KEY'], relogin=True)
        wandb.init(
            project=self.project,
            name=self.name
        )
        self.wandb = wandb
        self.run_dir = self.wandb.run.dir
        self.train_step = 0

    def log(self, scalar_name, scalar):
        self.wandb.log({scalar_name: scalar}, step=self.train_step, commit=False)

    def next_step(self):
        self.train_step += 1

    def save(self, file_path, save_online=True):
        file = os.path.basename(file_path)
        new_path = os.path.join(self.run_dir, file)
        shutil.copy2(file_path, new_path)
        if save_online:
            self.wandb.save(new_path)

    def __del__(self):
        self.wandb.finish()


def toggle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'dnnlib.tflib.network' and name == 'Network':
            return _TFNetworkStub
        return super().find_class(module, name)
