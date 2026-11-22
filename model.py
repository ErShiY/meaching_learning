from models import (
    ChannelFusionResNet18,
    DenseNet121Baseline,
    DenseNet121Attention,
    DenseNet121Refine,
    ExperimentalDenseNet121,
    FourChannelResNet18,
)
from models.bestfitting_densenet import BestfittingDenseNet121
from models.se_resnext50 import SEResNeXt50


DEFAULT_MODEL_NAME = "baseline"

MODEL_REGISTRY = {
    "baseline": DenseNet121Baseline,
    "bestfitting_densenet121": BestfittingDenseNet121,
    "densenet_attention": DenseNet121Attention,
    "densenet_refine": DenseNet121Refine,
    "experimental": ExperimentalDenseNet121,
    "resnet18": FourChannelResNet18,
    "resnet18_fusion": ChannelFusionResNet18,
    "se_resnext50": SEResNeXt50,
}

DEFAULT_PRETRAINED_WEIGHTS = {
    "baseline": "load_checkpoint/densenet121-a639ec97.pth",
    "bestfitting_densenet121": "load_checkpoint/densenet121-a639ec97.pth",
    "densenet_attention": "load_checkpoint/densenet121-a639ec97.pth",
    "densenet_refine": "load_checkpoint/densenet121-a639ec97.pth",
    "experimental": "load_checkpoint/densenet121-a639ec97.pth",
    "resnet18": "load_checkpoint/resnet18-f37072fd.pth",
    "resnet18_fusion": "load_checkpoint/resnet18-f37072fd.pth",
    "se_resnext50": "",
}


def available_models():
    return tuple(MODEL_REGISTRY.keys())


def default_pretrained_weights(model_name):
    try:
        return DEFAULT_PRETRAINED_WEIGHTS[model_name]
    except KeyError as error:
        choices = ", ".join(available_models())
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {choices}"
        ) from error


def build_model(model_name, num_classes, pretrained_weights=""):
    try:
        model_class = MODEL_REGISTRY[model_name]
    except KeyError as error:
        choices = ", ".join(available_models())
        raise ValueError(
            f"Unknown model '{model_name}'. Available models: {choices}"
        ) from error

    return model_class(
        num_classes=num_classes,
        pretrained_weights=pretrained_weights,
    )
