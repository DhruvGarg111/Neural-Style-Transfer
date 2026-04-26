## 2024-04-23 - Helper Text for Opaque API Values
**Learning:** Sometimes API contracts or backward compatibility requires keeping non-descriptive option labels (e.g., "Style 1", "Style 2"). This presents a poor UX as users don't know what they represent without external documentation.
**Action:** Use helper text or `info` parameters on form controls to provide descriptive context mapping the opaque values to their actual visual meanings (e.g., "Vibrant impressionist", "Dark aesthetic"), bridging the gap between API requirements and user comprehension.
## 2024-04-25 - Directly labeling options instead of helper mapping
**Learning:** For Gradio Radio buttons (and similar select components) where API values must be retained (e.g., "Style 1") but are non-descriptive to users, putting a mapping key in the `info` parameter still creates cognitive load. Users have to read the helper text and map it to the unhelpful button labels.
**Action:** Use `(display_label, api_value)` tuples for `choices`. This keeps the required API values intact for the backend while rendering clear, helpful names directly on the interactive elements, reducing cognitive load and saving the `info` param for actual helpful context.
## 2024-05-10 - Blank Canvas Friction
**Learning:** Users often bounce from generative AI tools if they have to find and upload their own image before they can see how the tool works. A "blank canvas" is intimidating.
**Action:** Always provide one-click example inputs (like Gradio's `examples` feature) for image/audio/text inputs. This allows users to test the functionality instantly, reducing initial friction and demonstrating the tool's capabilities immediately.
