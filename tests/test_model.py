import pytest
import torch

from model import get_model


def test_get_model_returns_correct_output_shape():
    model = get_model(architecture="resnet18", num_classes=10, pretrained=False)
    model.eval()
    inputs = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        outputs = model(inputs)
    assert outputs.shape == (2, 10)


def test_get_model_respects_num_classes():
    model = get_model(architecture="resnet18", num_classes=5, pretrained=False)
    assert model.fc.out_features == 5


def test_get_model_adapts_first_conv_for_small_images():
    model = get_model(architecture="resnet18", num_classes=10, pretrained=False)
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)
    assert isinstance(model.maxpool, torch.nn.Identity)


def test_get_model_unsupported_architecture_raises():
    with pytest.raises(ValueError):
        get_model(architecture="not-a-real-model", num_classes=10)
