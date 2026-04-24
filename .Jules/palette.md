## 2024-04-23 - Helper Text for Opaque API Values
**Learning:** Sometimes API contracts or backward compatibility requires keeping non-descriptive option labels (e.g., "Style 1", "Style 2"). This presents a poor UX as users don't know what they represent without external documentation.
**Action:** Use helper text or `info` parameters on form controls to provide descriptive context mapping the opaque values to their actual visual meanings (e.g., "Vibrant impressionist", "Dark aesthetic"), bridging the gap between API requirements and user comprehension.
## 2025-02-12 - Replaced abstract choices with display names in Gradio Radio
**Learning:** In Gradio applications, using abstract choices (like 'Style 1') with a separate 'info' string creates unnecessary cognitive load for the user, as they have to mentally map the info description to the option. Gradio's `gr.Radio` supports tuple choices `('Display Name', 'Internal Value')`, which allows displaying descriptive names directly on the UI options while maintaining internal API compatibility.
**Action:** Use tuple choices in Gradio selection components to provide clear, immediate descriptions directly on the selectable options.
