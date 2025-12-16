#!/bin/bash
# Verify Chapter 3 structure and figure references

set -e

DISS_FILE="FERNANDES_Dissertation.md"
FIGURES_DIR="figures"

echo "=== Chapter 3 Structure Verification ==="
echo ""

# Check for required sections
echo "Checking section structure..."
sections=(
    "3.1.1 Research Methodology Overview"
    "3.1.2 Framework Architecture Overview"
    "3.1.3 Data Collection Overview"
    "3.1.4 Analysis Approach Overview"
    "3.2.1 Methodology Alignment with Research Objectives"
    "3.2.2 Framework Representation of Live Production Systems"
    "3.2.3 Justification for Experimental Method"
    "3.2.4 Exclusion of Alternative Methods"
    "3.3.1 Methods and techniques"
    "3.3.2 Data collection and analysis"
    "3.3.3 Framework Implementation"
    "3.3.4 Framework Validation"
)

missing_sections=()
for section in "${sections[@]}"; do
    if ! grep -q "$section" "$DISS_FILE"; then
        missing_sections+=("$section")
    fi
done

if [ ${#missing_sections[@]} -eq 0 ]; then
    echo "✓ All required sections found"
else
    echo "✗ Missing sections:"
    printf '  - %s\n' "${missing_sections[@]}"
fi

echo ""

# Check for figure references
echo "Checking figure references..."
figures=(
    "Figure 3.0"
    "Figure 3.1"
    "Figure 3.2"
)

missing_figures=()
for fig in "${figures[@]}"; do
    if ! grep -q "$fig" "$DISS_FILE"; then
        missing_figures+=("$fig")
    fi
done

if [ ${#missing_figures[@]} -eq 0 ]; then
    echo "✓ All figure references found"
else
    echo "✗ Missing figure references:"
    printf '  - %s\n' "${missing_figures[@]}"
fi

echo ""

# Check for figure files (if they exist)
echo "Checking figure files..."
figure_files=(
    "$FIGURES_DIR/high-level-overview.png"
    "$FIGURES_DIR/framework-architecture.png"
    "$FIGURES_DIR/live-system-comparison.png"
)

missing_files=()
for file in "${figure_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    echo "✓ All figure files exist"
else
    echo "⚠ Figure files not yet converted (expected if diagrams not converted yet):"
    printf '  - %s\n' "${missing_files[@]}"
    echo "  Run: ./diagrams/convert_diagrams.sh or use diagrams/render_diagrams.html"
fi

echo ""

# Check for code references that should be removed
echo "Checking for code references..."
code_refs=(
    "Instant::now()"
    "getrusage()"
    "/proc filesystem"
)

found_refs=()
for ref in "${code_refs[@]}"; do
    if grep -q "$ref" "$DISS_FILE"; then
        found_refs+=("$ref")
    fi
done

if [ ${#found_refs[@]} -eq 0 ]; then
    echo "✓ No problematic code references found"
else
    echo "⚠ Found code references that may need review:"
    printf '  - %s\n' "${found_refs[@]}"
fi

echo ""
echo "=== Verification Complete ==="
