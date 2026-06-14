import torch
from klass.model import KlassModel
from klass.dataset import KlassDataset
from klass.config import Config

BATCH_SIZE = 2

cfg = Config.parse('C:/Projects/GitHub/ultimate/UrbanWatch/training/python/klass/configs/vcr.yml')

dataset = KlassDataset(cfg, 'train')
item = dataset.__getitem__(5)


model = KlassModel(cfg, training=False)
sample_args = (torch.randn(BATCH_SIZE, 3, cfg.model.imgH, cfg.model.imgW), )
torch.onnx.export(
        model.eval(),
        *sample_args,
        'klass.onnx',
        verbose=True,
        keep_initializers_as_inputs=True,
        opset_version=11, # requires old torch (e.g. 1.13.1), newest one (e.g. 2.9.0) will force us to use opset-18 which is not supported by jetpack 4.x
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}},
    )