## 2024-05-15 - PyTorch Memory Allocations in Inference
**Learning:** Found that `TransformerNet` uses standard `ReLU()` and separate `ReflectionPad2d` + `Conv2d` layers. In PyTorch, using `ReLU(inplace=True)` and `Conv2d(..., padding_mode='reflect')` prevents intermediate tensor allocations, saving significant VRAM and reducing memory bandwidth overhead during inference, especially on high-resolution images.
**Action:** Always check for inplace activations and fused padding modes in PyTorch models during inference optimization to reduce memory bandwidth bottlenecks.

## 2024-05-16 - PCIe Bandwidth Optimization with GPU Casting
**Learning:** Found that the app was transferring float32 output tensors to the CPU before clamping and converting to uint8. By applying `clamp(0, 255).to(torch.uint8)` on the GPU *before* `.cpu()`, we reduce the PCIe data transfer size by 75% (4 bytes down to 1 byte per value) and skip an unnecessary `.clone()` on the CPU.
**Action:** Always perform final terminal casting (like float to uint8 for images) on the GPU before transferring data back to the CPU to minimize PCIe memory bandwidth bottlenecks.
