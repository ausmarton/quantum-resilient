#!/bin/bash
# Convert Mermaid diagrams to SVG and PNG
# Requires: npm install -g @mermaid-js/mermaid-cli

set -e

DIAGRAM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIGURES_DIR="$DIAGRAM_DIR/../figures"

# Check if mmdc is available
if ! command -v mmdc &> /dev/null; then
    echo "Error: mermaid-cli (mmdc) not found."
    echo "Install with: npm install -g @mermaid-js/mermaid-cli"
    exit 1
fi

echo "Converting Mermaid diagrams to SVG and PNG..."

for mmd_file in "$DIAGRAM_DIR"/*.mmd; do
    if [ -f "$mmd_file" ]; then
        basename=$(basename "$mmd_file" .mmd)
        echo "  Converting $basename..."
        
        # Convert to SVG
        mmdc -i "$mmd_file" -o "$FIGURES_DIR/${basename}.svg" -b transparent
        
        # Convert to PNG (high resolution for Word doc)
        mmdc -i "$mmd_file" -o "$FIGURES_DIR/${basename}.png" -b transparent -w 2400 -H 1800
        
        echo "    ✓ Created ${basename}.svg and ${basename}.png"
    fi
done

echo "Done! Diagrams are in $FIGURES_DIR"
