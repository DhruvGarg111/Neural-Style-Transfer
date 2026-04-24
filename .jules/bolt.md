## 2024-05-15 - PyTorch Memory Allocations in Inference
**Learning:** Found that `TransformerNet` uses standard `ReLU()` and separate `ReflectionPad2d` + `Conv2d` layers. In PyTorch, using `ReLU(inplace=True)` and `Conv2d(..., padding_mode='reflect')` prevents intermediate tensor allocations, saving significant VRAM and reducing memory bandwidth overhead during inference, especially on high-resolution images.
**Action:** Always check for inplace activations and fused padding modes in PyTorch models during inference optimization to reduce memory bandwidth bottlenecks.
