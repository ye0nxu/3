# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Triton kernel for euclidean distance transform (EDT)"""

import sys

import torch

# Windows에서는 Triton DLL 로드 실패(0xC06D007F)로 프로세스가 종료되므로
# Windows이거나 Triton import에 실패하면 CPU fallback을 사용합니다.
_HAS_TRITON = False
if sys.platform != "win32":
    try:
        import triton
        import triton.language as tl
        _HAS_TRITON = True
    except Exception:
        pass


if _HAS_TRITON:
    @triton.jit
    def edt_kernel(inputs_ptr, outputs_ptr, v, z, height, width, horizontal: tl.constexpr):
        batch_id = tl.program_id(axis=0)
        if horizontal:
            row_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + row_id * width
            length = width
            stride = 1
        else:
            col_id = tl.program_id(axis=1)
            block_start = (batch_id * height * width) + col_id
            length = height
            stride = width

        k = 0
        for q in range(1, length):
            cur_input = tl.load(inputs_ptr + block_start + (q * stride))
            r = tl.load(v + block_start + (k * stride))
            z_k = tl.load(z + block_start + (k * stride))
            previous_input = tl.load(inputs_ptr + block_start + (r * stride))
            s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            while s <= z_k and k - 1 >= 0:
                k = k - 1
                r = tl.load(v + block_start + (k * stride))
                z_k = tl.load(z + block_start + (k * stride))
                previous_input = tl.load(inputs_ptr + block_start + (r * stride))
                s = (cur_input - previous_input + q * q - r * r) / (q - r) / 2

            k = k + 1
            tl.store(v + block_start + (k * stride), q)
            tl.store(z + block_start + (k * stride), s)
            if k + 1 < length:
                tl.store(z + block_start + ((k + 1) * stride), 1e9)

        k = 0
        for q in range(length):
            while (
                k + 1 < length
                and tl.load(
                    z + block_start + ((k + 1) * stride), mask=(k + 1) < length, other=q
                )
                < q
            ):
                k += 1
            r = tl.load(v + block_start + (k * stride))
            d = q - r
            old_value = tl.load(inputs_ptr + block_start + (r * stride))
            tl.store(outputs_ptr + block_start + (q * stride), old_value + d * d)

    def edt_triton(data: torch.Tensor):
        """
        Computes the Euclidean Distance Transform (EDT) of a batch of binary images.

        Args:
            data: A tensor of shape (B, H, W) representing a batch of binary images.

        Returns:
            A tensor of the same shape as data containing the EDT.
            It should be equivalent to a batched version of cv2.distanceTransform(input, cv2.DIST_L2, 0)
        """
        assert data.dim() == 3
        assert data.is_cuda
        B, H, W = data.shape
        data = data.contiguous()

        output = torch.where(data, 1e18, 0.0)
        assert output.is_contiguous()

        parabola_loc = torch.zeros(B, H, W, dtype=torch.uint32, device=data.device)
        parabola_inter = torch.empty(B, H, W, dtype=torch.float, device=data.device)
        parabola_inter[:, :, 0] = -1e18
        parabola_inter[:, :, 1] = 1e18

        grid = (B, H)
        edt_kernel[grid](
            output.clone(),
            output,
            parabola_loc,
            parabola_inter,
            H,
            W,
            horizontal=True,
        )

        parabola_loc.zero_()
        parabola_inter[:, :, 0] = -1e18
        parabola_inter[:, :, 1] = 1e18

        grid = (B, W)
        edt_kernel[grid](
            output.clone(),
            output,
            parabola_loc,
            parabola_inter,
            H,
            W,
            horizontal=False,
        )
        return output.sqrt()

else:
    ## =====================================
    ## 함수 기능 : Triton 미사용 시 cv2 기반 CPU EDT 폴백
    ## 매개 변수 : data(torch.Tensor) - (B, H, W) 바이너리 텐서
    ## 반환 결과 : EDT 결과 텐서 (data와 동일한 device/shape)
    ## =====================================
    def edt_triton(data: torch.Tensor) -> torch.Tensor:
        """
        CPU fallback for edt_triton using cv2.distanceTransform.
        Equivalent to a batched cv2.distanceTransform(input, cv2.DIST_L2, 0).
        """
        import cv2
        import numpy as np

        device = data.device
        uint8_np = data.bool().cpu().numpy().astype(np.uint8)  # (B, H, W)
        result = np.zeros(uint8_np.shape, dtype=np.float32)
        for b in range(uint8_np.shape[0]):
            result[b] = cv2.distanceTransform(uint8_np[b], cv2.DIST_L2, 0)
        return torch.from_numpy(result).to(device=device)
