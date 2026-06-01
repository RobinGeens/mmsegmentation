# Simba-L sequence lengths per crop size

Effective strides are 4x, 8x, 16x, 32x relative to input.
Embed dims are [96, 192, 384, 512] for stages 1-4.

## Tokens per forward pass

| Crop (HxW)   | Stage 1 (÷4)          | Stage 2 (÷8)         | Stage 3 (÷16)        | Stage 4 (÷32)       |
|---------------|-----------------------|----------------------|----------------------|---------------------|
| 512x512       | 128x128 = 16,384      | 64x64 = 4,096        | 32x32 = 1,024        | 16x16 = 256         |
| 512x1024      | 128x256 = 32,768      | 64x128 = 8,192       | 32x64 = 2,048        | 16x32 = 512         |
| 1024x1024     | 256x256 = 65,536      | 128x128 = 16,384     | 64x64 = 4,096        | 32x32 = 1,024       |
| 1024x2048     | 256x512 = 131,072     | 128x256 = 32,768     | 64x128 = 8,192       | 32x64 = 2,048       |

## Slide-window inference on Cityscapes (1024x2048)

Grid formula from mmseg `slide_inference`: `h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1`.

| Crop (HxW)   | No overlap | 25% overlap (current) | 50% overlap |
|---------------|------------|-----------------------|-------------|
| 512x512       | 2x4 = 8    | 3x5 = 15              | 3x7 = 21    |
| 512x1024      | 2x2 = 4    | 3x3 = 9               | 3x3 = 9     |
| 1024x1024     | 1x2 = 2    | 1x3 = 3               | 1x3 = 3     |
| 1024x2048     | 1x1 = 1    | 1x1 = 1               | 1x1 = 1     |

## Total stage-1 tokens per image (windows x L_stage1)

This is proportional to total backbone compute, since stage 1 dominates.

| Crop (HxW)   | No overlap  | 25% overlap (current) | 50% overlap  |
|---------------|-------------|-----------------------|--------------|
| 512x512       | 131,072     | 245,760                | 344,064      |
| 512x1024      | 131,072     | 294,912                | 294,912      |
| 1024x1024     | 131,072     | 196,608                | 196,608      |
| 1024x2048     | 131,072     | 131,072                | 131,072      |

Note: with no overlap, total stage-1 tokens = 131,072 for all crops (the full image is tiled exactly once).
With overlap, smaller crops pay a larger overhead because border regions are re-processed more often.
