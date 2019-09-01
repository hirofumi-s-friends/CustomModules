import sys
import re
import torch.nn as nn
import torchvision.models.densenet as densenet
from torchvision.models.densenet import DenseNet as TorchDenseNet, model_urls
from torch.hub import load_state_dict_from_url


# Override _densenet to support cached model file in model_path input port
def _densenet(arch, growth_rate, block_config, num_init_features, pretrained, progress, model_path,
              **kwargs):
    model = TorchDenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        def _load_state_dict(model, model_url, model_path, *args, **kwargs):
            # '.'s are no longer allowed in module names, but previous _DenseLayer
            # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
            # They are also in the checkpoints in model_urls. This pattern is used
            # to find such keys.
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

            state_dict = load_state_dict_from_url(model_url, model_dir=model_path)
            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
            model.load_state_dict(state_dict)

        _load_state_dict(model, model_urls[arch], model_path, progress)
    return model


class DenseNet(nn.Module):
    def __init__(self, model_type='densenet201', model_path='/home/root/chjinche/projects/img_cls/denseNet/pretrained',
                 pretrained=True, memory_efficient=False, classes=20):
        super(DenseNet, self).__init__()
        densenet._densenet = _densenet
        if not pretrained:
            model_type = "densenet201"
        densenet_func = getattr(densenet, model_type, None)
        if densenet_func is None:
            # todo: catch exception and throw AttributeError
            print(f"Error: No such pretrained model {model_type}")
            sys.exit()
        self.model1 = densenet_func(pretrained=pretrained, model_path=model_path)
        self.model2 = nn.Linear(1000, classes)

    def forward(self, input):
        output = self.model1(input)
        output = self.model2(output)
        return output


if __name__ == '__main__':
    net = DenseNet()
