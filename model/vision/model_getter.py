from os.path import expanduser, expandvars

import timm
import torch
import torchvision

from model.vision.voltron import VoltronVisual


def get_clip_vit_base_patch16(**kwargs):
    model = timm.create_model(
        "hf_hub:timm/vit_base_patch16_clip_224.openai", pretrained=True
    )
    return model


def get_resnet(name, weights=None, frozen=False, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()

    if frozen:
        resnet.requires_grad_(False)

    return resnet


def get_anzu_pretrained(name, checkpoint_path, frozen=False, **kwargs):
    assert name in ["resnet50", "resnet34", "resnet18"]

    model = get_resnet(name)

    # load weights from anzu pretrained
    checkpoint = torch.load(
        expandvars(expanduser(checkpoint_path)),
        map_location="cpu",
    )
    state_dict = checkpoint["state_dict"]
    d = {}
    for k, v in state_dict.items():
        if k.startswith("backbone."):
            d[k[len("backbone.") :]] = v
    model.load_state_dict(d)

    if frozen:
        model.requires_grad_(False)

    return model


def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m

    r3m.device = "cpu"
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to("cpu")
    return resnet_model


def get_voltron(weights, frozen=False):
    model = VoltronVisual(weights, frozen)
    return model
