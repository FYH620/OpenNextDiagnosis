import torch
import numpy as np
from torchvision.models import resnet50, inception_v3, densenet121
from config import MODEL_DIR


def load_pretrained_weights(model, model_path):
    model_dict = model.state_dict()
    model.load_state_dict(model_dict, strict=False)
    return model


def build_resnet50(num_classes):
    resnet = resnet50(num_classes=num_classes)
    resnet = load_pretrained_weights(resnet, MODEL_DIR["resnet"])
    return resnet


def build_inception_v3(num_classes):
    inception = inception_v3(aux_logits=False, num_classes=num_classes)
    inception = load_pretrained_weights(inception, MODEL_DIR["inception"])
    return inception


def build_densenet121(num_classes):
    densenet = densenet121(num_classes=num_classes)
    densenet = load_pretrained_weights(densenet, MODEL_DIR["densenet"])
    return densenet
