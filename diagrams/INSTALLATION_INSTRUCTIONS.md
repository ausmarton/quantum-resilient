# Diagram Conversion Instructions

## Option 1: Using mermaid-cli (Recommended for automated conversion)

### Installation
```bash
npm install -g @mermaid-js/mermaid-cli
```

### Conversion
```bash
cd diagrams
./convert_diagrams.sh
```

This will generate SVG and PNG files in the `figures/` directory.

## Option 2: Using HTML Renderer (No installation required)

1. Open `diagrams/render_diagrams.html` in a web browser
2. The diagrams will render automatically
3. To export:
   - **For PNG**: Right-click on each diagram → "Inspect" → Right-click on the SVG element → "Copy image" or take a screenshot
   - **For PDF**: Use browser Print function (Ctrl+P / Cmd+P) → Save as PDF
   - **For SVG**: Right-click on diagram → "Inspect" → Copy the SVG element → Save as .svg file

## Option 3: Online Mermaid Editor

1. Go to https://mermaid.live/
2. Copy the contents of each `.mmd` file
3. Paste into the editor
4. Export as PNG or SVG using the download button

## Option 4: Using Python (if you have graphviz)

```bash
# Install dependencies
pip install pygraphviz  # May require graphviz system package

# Or use a simpler approach with matplotlib
pip install matplotlib networkx
```

Then use a Python script to render (we can create this if needed).

---

## Recommended Approach

For Word or other document formats, we recommend:
1. Use Option 1 (mermaid-cli) if you can install npm packages
2. Use Option 2 (HTML renderer) if you prefer no installation - just open in browser and screenshot
3. Use Option 3 (online editor) for quick one-off conversions

The HTML renderer (`render_diagrams.html`) is ready to use - just open it in any modern browser.
