# Copyright (c) OpenMMLab. All rights reserved.
"""Convert KITTI 3-class checkpoint to SiT 2-class checkpoint.

This script converts a PointPillars checkpoint trained on KITTI 3-class dataset
(Pedestrian, Cyclist, Car) to a 2-class checkpoint for SiT dataset (Pedestrian, Car).

The conversion:
- Maps classification head weights: [Pedestrian, Cyclist, Car] -> [Pedestrian, Car]
- Updates checkpoint metadata to reflect 2-class configuration
- Preserves all other weights (backbone, neck, etc.)

Usage:
    python tools/model_converters/convert_kitti3class_to_sit2class.py \
        --input checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20220301_150306-37dc2420.pth \
        --output checkpoints/hv_pointpillars_kitti3class_to_sit2class.pth
"""

import argparse

import torch


def convert_checkpoint(input_ckpt: str, output_ckpt: str):
    """Convert 3-class to 2-class checkpoint.
    
    Args:
        input_ckpt (str): Path to input 3-class checkpoint.
        output_ckpt (str): Path to output 2-class checkpoint.
    """
    print(f"Loading checkpoint from {input_ckpt}...")
    checkpoint = torch.load(input_ckpt, map_location='cpu')
    
    if 'state_dict' not in checkpoint:
        raise ValueError("Checkpoint must contain 'state_dict' key")
    
    state_dict = checkpoint['state_dict']
    
    # Keys to modify: classification head weights and biases
    cls_weight_key = 'bbox_head.conv_cls.weight'
    cls_bias_key = 'bbox_head.conv_cls.bias'

    modified = False

    # Convert classification head weights
    if cls_weight_key in state_dict:
        old_weight = state_dict[cls_weight_key]  # Shape: [A * C_old, in_ch, 1, 1]
        print(f"Original classification weight shape: {old_weight.shape}")

        C_old = 3  # Pedestrian, Cyclist, Car in the KITTI checkpoint
        out_channels, in_ch, k1, k2 = old_weight.shape
        if out_channels % C_old != 0:
            raise ValueError(
                f"Classification out_channels {out_channels} not divisible by {C_old}; "
                "unexpected head layout."
            )
        num_anchors = out_channels // C_old
        C_new = 2  # Pedestrian, Car
        out_channels_new = num_anchors * C_new

        # Reshape to [anchors, C_old, in_ch, 1, 1], select classes 0 and 2, reshape back
        w = old_weight.view(num_anchors, C_old, in_ch, k1, k2)
        new_w = torch.stack([w[:, 0], w[:, 2]], dim=1)  # [anchors, 2, in_ch, 1, 1]
        new_w = new_w.reshape(out_channels_new, in_ch, k1, k2)

        state_dict[cls_weight_key] = new_w
        print(f"Converted classification weight shape: {new_w.shape} (anchors={num_anchors}, classes={C_new})")
        modified = True

    # Convert classification head bias
    if cls_bias_key in state_dict:
        old_bias = state_dict[cls_bias_key]  # Shape: [A * C_old]
        print(f"Original classification bias shape: {old_bias.shape}")

        C_old = 3
        out_channels = old_bias.shape[0]
        if out_channels % C_old != 0:
            raise ValueError(
                f"Classification bias len {out_channels} not divisible by {C_old}; unexpected head layout."
            )
        num_anchors = out_channels // C_old
        C_new = 2
        out_channels_new = num_anchors * C_new

        b = old_bias.view(num_anchors, C_old)
        new_b = torch.stack([b[:, 0], b[:, 2]], dim=1).reshape(out_channels_new)

        state_dict[cls_bias_key] = new_b
        print(f"Converted classification bias shape: {new_b.shape} (anchors={num_anchors}, classes={C_new})")
        modified = True

    if not modified:
        print("Warning: No classification head weights found. Checkpoint may not be compatible.")
        return
    
    # Update checkpoint state dict
    checkpoint['state_dict'] = state_dict
    
    # Update metadata
    if 'meta' not in checkpoint:
        checkpoint['meta'] = {}
    
    # Update classes in metadata
    checkpoint['meta']['classes'] = ['Pedestrian', 'Car']
    
    # Update dataset_meta if present
    if 'dataset_meta' in checkpoint['meta']:
        checkpoint['meta']['dataset_meta']['classes'] = ['Pedestrian', 'Car']
    else:
        checkpoint['meta']['dataset_meta'] = {'classes': ['Pedestrian', 'Car']}
    
    # Add conversion note
    if 'note' not in checkpoint['meta']:
        checkpoint['meta']['note'] = ''
    checkpoint['meta']['note'] += (
        '\nConverted from KITTI 3-class (Pedestrian, Cyclist, Car) '
        'to SiT 2-class (Pedestrian, Car) using convert_kitti3class_to_sit2class.py'
    )
    
    # Save converted checkpoint
    print(f"Saving converted checkpoint to {output_ckpt}...")
    torch.save(checkpoint, output_ckpt)
    print(f"✅ Successfully converted checkpoint!")
    print(f"   Input:  {input_ckpt}")
    print(f"   Output: {output_ckpt}")
    print(f"   Classes: 3 (Pedestrian, Cyclist, Car) -> 2 (Pedestrian, Car)")


def main():
    parser = argparse.ArgumentParser(
        description='Convert KITTI 3-class checkpoint to SiT 2-class checkpoint'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input 3-class checkpoint'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output 2-class checkpoint'
    )
    
    args = parser.parse_args()
    
    convert_checkpoint(args.input, args.output)


if __name__ == '__main__':
    main()

