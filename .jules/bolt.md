## 2024-05-15 - PyTorch Memory Allocations in Inference
**Learning:** Found that `TransformerNet` uses standard `ReLU()` and separate `ReflectionPad2d` + `Conv2d` layers. In PyTorch, using `ReLU(inplace=True)` and `Conv2d(..., padding_mode='reflect')` prevents intermediate tensor allocations, saving significant VRAM and reducing memory bandwidth overhead during inference, especially on high-resolution images.
**Action:** Always check for inplace activations and fused padding modes in PyTorch models during inference optimization to reduce memory bandwidth bottlenecks.
## 2024-05-15 - PyTorch to Pillow Image Conversion
**Learning:** Found a memory bottleneck in converting PyTorch tensors to Pillow Images: using `.clone().clamp().numpy().astype("uint8")` creates redundant memory allocations and relies on NumPy type casting.
**Action:** Always avoid `.clone()` if not modifying the source tensor structurally, and use PyTorch's native `.byte()` (or `.to(torch.uint8)`) combined with `.permute()` before converting to NumPy. This saves intermediate allocations and reduces memory usage during post-processing.
