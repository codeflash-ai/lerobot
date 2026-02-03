#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for all  decorated functions/classes.

This test file verifies that all  decorated forward methods work correctly
with CUDA, fp32 precision, and simple input shapes.
"""

import pytest
import torch

# Import all modules with  decorated classes
from lerobot.datasets.transforms import RandomSubsetApply, ImageTransforms, ImageTransformsConfig
from lerobot.policies.vqbet.modeling_vqbet import (
    VQBeTPolicy,
    SpatialSoftmax,
    VQBeTModel,
    VQBeTHead,
    VQBeTRgbEncoder,
    FocalLoss,
)
from lerobot.policies.diffusion.modeling_diffusion import (
    DiffusionPolicy,
    SpatialSoftmax as DiffusionSpatialSoftmax,
    DiffusionRgbEncoder,
    DiffusionSinusoidalPosEmb,
    DiffusionConv1dBlock,
    DiffusionConditionalUnet1d,
    DiffusionConditionalResidualBlock1d,
)
from lerobot.policies.act.modeling_act import (
    ACTPolicy,
    ACT,
    ACTEncoder,
    ACTEncoderLayer,
    ACTDecoder,
    ACTDecoderLayer,
    ACTSinusoidalPositionEmbedding2d,
)
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


# Helper to check CUDA availability
def require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


class TestVQBeTCompile:
    """Test  decorated classes in VQBeT."""

    def test_spatial_softmax_forward(self):
        """Test SpatialSoftmax.forward() method."""
        require_cuda()
        device = torch.device("cuda")

        # Create instance
        input_shape = (512, 10, 12)
        num_kp = 32
        model = SpatialSoftmax(input_shape, num_kp=num_kp).to(device)

        # Create input
        batch_size = 4
        features = torch.randn(batch_size, *input_shape, dtype=torch.float32, device=device)

        # Test forward method
        output = model.forward(features)

        # Verify output shape
        assert output.shape == (batch_size, num_kp, 2)
        assert output.dtype == torch.float32
        assert output.device.type == "cuda"

    def test_focal_loss_forward(self):
        """Test FocalLoss.forward() method."""
        require_cuda()
        device = torch.device("cuda")

        # Create instance
        model = FocalLoss(gamma=2.0).to(device)

        # Create input (batch_size, num_classes)
        batch_size = 8
        num_classes = 10
        input_tensor = torch.randn(batch_size, num_classes, dtype=torch.float32, device=device)
        target = torch.randint(0, num_classes, (batch_size,), device=device)

        # Test forward method
        loss = model.forward(input_tensor, target)

        # Verify output
        assert loss.shape == ()
        assert loss.dtype == torch.float32
        assert loss.device.type == "cuda"


class TestDiffusionCompile:
    """Test  decorated classes in Diffusion Policy."""

    def test_diffusion_spatial_softmax_forward(self):
        """Test DiffusionSpatialSoftmax.forward() method."""
        require_cuda()
        device = torch.device("cuda")

        # Create instance
        input_shape = (512, 10, 12)
        num_kp = 32
        model = DiffusionSpatialSoftmax(input_shape, num_kp=num_kp).to(device)

        # Create input
        batch_size = 4
        features = torch.randn(batch_size, *input_shape, dtype=torch.float32, device=device)

        # Test forward method
        output = model.forward(features)

        # Verify output shape
        assert output.shape == (batch_size, num_kp, 2)
        assert output.dtype == torch.float32
        assert output.device.type == "cuda"

    def test_diffusion_sinusoidal_pos_emb_forward(self):
        """Test DiffusionSinusoidalPosEmb.forward() method."""
        require_cuda()
        device = torch.device("cuda")

        # Create instance
        dim = 128
        model = DiffusionSinusoidalPosEmb(dim).to(device)

        # Create input (timesteps)
        batch_size = 8
        x = torch.randint(0, 1000, (batch_size,), dtype=torch.int64, device=device).float()

        # Test forward method
        output = model.forward(x)

        # Verify output shape
        assert output.shape == (batch_size, dim)
        assert output.dtype == torch.float32
        assert output.device.type == "cuda"

    def test_diffusion_conv1d_block_forward(self):
        """Test DiffusionConv1dBlock.forward() method."""
        require_cuda()
        device = torch.device("cuda")

        # Create instance
        inp_channels = 32
        out_channels = 64
        kernel_size = 3
        model = DiffusionConv1dBlock(inp_channels, out_channels, kernel_size).to(device)

        # Create input (batch, channels, time)
        batch_size = 4
        seq_len = 16
        x = torch.randn(batch_size, inp_channels, seq_len, dtype=torch.float32, device=device)

        # Test forward method
        output = model.forward(x)

        # Verify output shape
        assert output.shape == (batch_size, out_channels, seq_len)
        assert output.dtype == torch.float32
        assert output.device.type == "cuda"

    def test_diffusion_conditional_residual_block1d_forward(self):
        """Test DiffusionConditionalResidualBlock1d.forward() method."""
        require_cuda()
        device = torch.device("cuda")

        # Create instance
        in_channels = 32
        out_channels = 64
        cond_dim = 128
        model = DiffusionConditionalResidualBlock1d(
            in_channels, out_channels, cond_dim
        ).to(device)

        # Create inputs
        batch_size = 4
        seq_len = 16
        x = torch.randn(batch_size, in_channels, seq_len, dtype=torch.float32, device=device)
        cond = torch.randn(batch_size, cond_dim, dtype=torch.float32, device=device)

        # Test forward method
        output = model.forward(x, cond)

        # Verify output shape
        assert output.shape == (batch_size, out_channels, seq_len)
        assert output.dtype == torch.float32
        assert output.device.type == "cuda"


class TestACTCompile:
    """Test  decorated classes in ACT Policy."""

    def test_act_encoder_layer_forward(self):
        """Test ACTEncoderLayer.forward() method."""
        require_cuda()
        device = torch.device("cuda")

        # Create config and instance
        config = ACTConfig()
        model = ACTEncoderLayer(config).to(device)

        # Create input (seq, batch, dim)
        seq_len = 10
        batch_size = 4
        x = torch.randn(seq_len, batch_size, config.dim_model, dtype=torch.float32, device=device)
        pos_embed = torch.randn(seq_len, batch_size, config.dim_model, dtype=torch.float32, device=device)

        # Test forward method
        output = model.forward(x, pos_embed=pos_embed)

        # Verify output shape
        assert output.shape == (seq_len, batch_size, config.dim_model)
        assert output.dtype == torch.float32
        assert output.device.type == "cuda"

    def test_act_encoder_forward(self):
        """Test ACTEncoder.forward() method."""
        require_cuda()
        device = torch.device("cuda")

        # Create config and instance
        config = ACTConfig()
        model = ACTEncoder(config).to(device)

        # Create input (seq, batch, dim)
        seq_len = 10
        batch_size = 4
        x = torch.randn(seq_len, batch_size, config.dim_model, dtype=torch.float32, device=device)
        pos_embed = torch.randn(seq_len, batch_size, config.dim_model, dtype=torch.float32, device=device)

        # Test forward method
        output = model.forward(x, pos_embed=pos_embed)

        # Verify output shape
        assert output.shape == (seq_len, batch_size, config.dim_model)
        assert output.dtype == torch.float32
        assert output.device.type == "cuda"

    def test_act_decoder_layer_forward(self):
        """Test ACTDecoderLayer.forward() method."""
        require_cuda()
        device = torch.device("cuda")

        # Create config and instance
        config = ACTConfig()
        model = ACTDecoderLayer(config).to(device)

        # Create inputs (seq, batch, dim)
        dec_seq_len = 8
        enc_seq_len = 10
        batch_size = 4
        x = torch.randn(dec_seq_len, batch_size, config.dim_model, dtype=torch.float32, device=device)
        encoder_out = torch.randn(enc_seq_len, batch_size, config.dim_model, dtype=torch.float32, device=device)
        decoder_pos_embed = torch.randn(dec_seq_len, batch_size, config.dim_model, dtype=torch.float32, device=device)
        encoder_pos_embed = torch.randn(enc_seq_len, batch_size, config.dim_model, dtype=torch.float32, device=device)

        # Test forward method
        output = model.forward(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)

        # Verify output shape
        assert output.shape == (dec_seq_len, batch_size, config.dim_model)
        assert output.dtype == torch.float32
        assert output.device.type == "cuda"

    def test_act_decoder_forward(self):
        """Test ACTDecoder.forward() method."""
        require_cuda()
        device = torch.device("cuda")

        # Create config and instance
        config = ACTConfig()
        model = ACTDecoder(config).to(device)

        # Create inputs (seq, batch, dim)
        dec_seq_len = 8
        enc_seq_len = 10
        batch_size = 4
        x = torch.randn(dec_seq_len, batch_size, config.dim_model, dtype=torch.float32, device=device)
        encoder_out = torch.randn(enc_seq_len, batch_size, config.dim_model, dtype=torch.float32, device=device)
        decoder_pos_embed = torch.randn(dec_seq_len, batch_size, config.dim_model, dtype=torch.float32, device=device)
        encoder_pos_embed = torch.randn(enc_seq_len, batch_size, config.dim_model, dtype=torch.float32, device=device)

        # Test forward method
        output = model.forward(x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed)

        # Verify output shape
        assert output.shape == (dec_seq_len, batch_size, config.dim_model)
        assert output.dtype == torch.float32
        assert output.device.type == "cuda"

    def test_act_sinusoidal_position_embedding2d_forward(self):
        """Test ACTSinusoidalPositionEmbedding2d.forward() method."""
        require_cuda()
        device = torch.device("cuda")

        # Create instance
        dimension = 128
        model = ACTSinusoidalPositionEmbedding2d(dimension).to(device)

        # Create input (batch, channels, height, width)
        batch_size = 4
        channels = 256
        height = 7
        width = 7
        x = torch.randn(batch_size, channels, height, width, dtype=torch.float32, device=device)

        # Test forward method
        output = model.forward(x)

        # Verify output shape
        assert output.shape == (1, dimension * 2, height, width)
        assert output.dtype == torch.float32
        assert output.device.type == "cuda"


class TestTransformsCompile:
    """Test  decorated classes in transforms."""

    def test_random_subset_apply_forward(self):
        """Test RandomSubsetApply.forward() method."""
        require_cuda()
        device = torch.device("cuda")

        # Create simple transforms
        from torchvision.transforms import v2
        transforms = [v2.Identity(), v2.Identity(), v2.Identity()]

        # Create instance
        model = RandomSubsetApply(transforms, n_subset=2)

        # Create input
        batch_size = 4
        channels = 3
        height = 84
        width = 84
        x = torch.randn(batch_size, channels, height, width, dtype=torch.float32, device=device)

        # Test forward method
        output = model.forward(x)

        # Verify output shape (Identity transforms should preserve shape)
        assert output.shape == (batch_size, channels, height, width)
        assert output.dtype == torch.float32

    def test_image_transforms_forward(self):
        """Test ImageTransforms.forward() method."""
        require_cuda()
        device = torch.device("cuda")

        # Create config with transforms disabled for simple test
        config = ImageTransformsConfig(enable=False)
        model = ImageTransforms(config)

        # Create input
        batch_size = 4
        channels = 3
        height = 84
        width = 84
        x = torch.randn(batch_size, channels, height, width, dtype=torch.float32, device=device)

        # Test forward method
        output = model.forward(x)

        # Verify output shape
        assert output.shape == (batch_size, channels, height, width)
        assert output.dtype == torch.float32


# Note: Full policy tests are covered in test_policies.py
# This file focuses on unit testing individual  decorated modules


class TestComparisonForwardVsCall:
    """Test that .forward() and () produce same results (with torch.allclose)."""

    def test_spatial_softmax_forward_vs_call(self):
        """Test that SpatialSoftmax.forward() matches SpatialSoftmax()."""
        require_cuda()
        device = torch.device("cuda")

        # Create instance
        input_shape = (512, 10, 12)
        num_kp = 32
        model = SpatialSoftmax(input_shape, num_kp=num_kp).to(device)
        model.eval()  # Set to eval mode for deterministic behavior

        # Create input
        batch_size = 4
        torch.manual_seed(42)
        features = torch.randn(batch_size, *input_shape, dtype=torch.float32, device=device)

        # Test both methods with no_grad to avoid autograd graph differences
        with torch.no_grad():
            output_forward = model.forward(features)
            output_call = model(features)

        # Verify they match with default torch.allclose parameters
        assert torch.allclose(output_forward, output_call)

    def test_focal_loss_forward_vs_call(self):
        """Test that FocalLoss.forward() matches FocalLoss()."""
        require_cuda()
        device = torch.device("cuda")

        # Create instance
        model = FocalLoss(gamma=2.0).to(device)
        model.eval()

        # Create input
        batch_size = 8
        num_classes = 10
        torch.manual_seed(42)
        input_tensor = torch.randn(batch_size, num_classes, dtype=torch.float32, device=device)
        target = torch.randint(0, num_classes, (batch_size,), device=device)

        # Test both methods
        with torch.no_grad():
            loss_forward = model.forward(input_tensor, target)
            loss_call = model(input_tensor, target)

        # Verify they match with default torch.allclose parameters
        assert torch.allclose(loss_forward, loss_call)

    def test_diffusion_conv1d_block_forward_vs_call(self):
        """Test that DiffusionConv1dBlock.forward() matches ()."""
        require_cuda()
        device = torch.device("cuda")

        # Create instance
        inp_channels = 32
        out_channels = 64
        kernel_size = 3
        model = DiffusionConv1dBlock(inp_channels, out_channels, kernel_size).to(device)
        model.eval()

        # Create input
        batch_size = 4
        seq_len = 16
        torch.manual_seed(42)
        x = torch.randn(batch_size, inp_channels, seq_len, dtype=torch.float32, device=device)

        # Test both methods
        with torch.no_grad():
            output_forward = model.forward(x)
            output_call = model(x)

        # Verify they match with default torch.allclose parameters
        assert torch.allclose(output_forward, output_call)

    def test_act_encoder_layer_forward_vs_call(self):
        """Test that ACTEncoderLayer.forward() matches ()."""
        require_cuda()
        device = torch.device("cuda")

        # Create config and instance
        config = ACTConfig()
        model = ACTEncoderLayer(config).to(device)
        model.eval()

        # Create input
        seq_len = 10
        batch_size = 4
        torch.manual_seed(42)
        x = torch.randn(seq_len, batch_size, config.dim_model, dtype=torch.float32, device=device)
        pos_embed = torch.randn(seq_len, batch_size, config.dim_model, dtype=torch.float32, device=device)

        # Test both methods
        with torch.no_grad():
            output_forward = model.forward(x, pos_embed=pos_embed)
            output_call = model(x, pos_embed=pos_embed)

        # Verify they match with default torch.allclose parameters
        assert torch.allclose(output_forward, output_call)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
