#!/usr/bin/env python3
"""
Analyze how figures are used in the dissertation narrative.
Check if all figures are:
1. Referenced in the text
2. Integrated with proper academic language
3. Contributing to the narrative
"""

import re
from collections import defaultdict

def analyze_figures(doc_path):
    with open(doc_path, 'r') as f:
        lines = f.readlines()
    
    # Find all figure definitions
    figures = {}
    for i, line in enumerate(lines, 1):
        # Match figure captions
        fig_match = re.search(r'(?:^\s*\*+\s*|^\s*)Figure\s+([0-9]+\.[0-9]+[a-z]?|[0-9]+[a-z]?)\s*[:*]', line, re.IGNORECASE)
        if fig_match:
            fig_num = fig_match.group(1)
            figures[fig_num] = {
                'line': i,
                'caption': line.strip()[:100],
                'references': []
            }
    
    # Find all figure references in text (not in captions)
    for i, line in enumerate(lines, 1):
        # Skip caption lines
        if re.match(r'^\s*\*+\s*Figure', line, re.IGNORECASE):
            continue
        
        # Find references
        for fig_num in figures.keys():
            # Match "Figure X.Y" or "(Figure X.Y)" patterns
            pattern = rf'(?:^|[^a-zA-Z])Figure\s+{re.escape(fig_num)}(?:[^a-zA-Z]|$)'
            if re.search(pattern, line, re.IGNORECASE):
                figures[fig_num]['references'].append({
                    'line': i,
                    'context': line.strip()[:100]
                })
    
    return figures, lines

def assess_integration_quality(fig_num, references, lines):
    """Assess how well a figure is integrated into the narrative."""
    issues = []
    strengths = []
    
    if not references:
        issues.append("NOT REFERENCED in text")
        return issues, strengths
    
    for ref in references:
        context = ref['context'].lower()
        line_num = ref['line']
        
        # Check for weak integration patterns
        weak_patterns = [
            r'figure\s+\d+\.\d+\s+indicates',
            r'figure\s+\d+\.\d+\s+shows',
            r'figure\s+\d+\.\d+\s+reveals',
            r'as\s+shown\s+in\s+figure',
            r'see\s+figure',
        ]
        
        # Check for strong integration patterns
        strong_patterns = [
            r'\(figure\s+\d+\.\d+\)',  # Parenthetical reference
            r'consistent\s+with\s+figure',
            r'supported\s+by\s+figure',
            r'figure\s+\d+\.\d+\s+demonstrates',
            r'figure\s+\d+\.\d+\s+illustrates',
            r'as\s+visualised\s+in\s+figure',
        ]
        
        is_weak = any(re.search(pattern, context) for pattern in weak_patterns)
        is_strong = any(re.search(pattern, context) for pattern in strong_patterns)
        
        if is_weak:
            issues.append(f"Line {line_num}: Weak integration - stating what figure shows rather than using it to support argument")
        elif is_strong:
            strengths.append(f"Line {line_num}: Strong integration - figure used to support argument")
        else:
            # Check if it's in an "Observed Result" or "Interpretation" section
            # Look backwards for context
            context_lines = lines[max(0, line_num-5):line_num]
            context_text = ' '.join(context_lines).lower()
            if 'observed result' in context_text or 'interpretation' in context_text:
                strengths.append(f"Line {line_num}: Integrated in structured analysis section")
            else:
                issues.append(f"Line {line_num}: Neutral integration - could be strengthened")
    
    return issues, strengths

def main():
    doc_path = 'FERNANDES_H2807295_F87_dissertation (2).md'
    
    figures, lines = analyze_figures(doc_path)
    
    print("=" * 80)
    print("FIGURE USAGE ANALYSIS")
    print("=" * 80)
    print()
    
    for fig_num in sorted(figures.keys(), key=lambda x: (float(x.split('.')[0]) if '.' in x else float(x), x)):
        fig_info = figures[fig_num]
        refs = fig_info['references']
        
        print(f"Figure {fig_num} (line {fig_info['line']}):")
        print(f"  Caption: {fig_info['caption'][:70]}...")
        print(f"  References: {len(refs)}")
        
        if refs:
            issues, strengths = assess_integration_quality(fig_num, refs, lines)
            
            if strengths:
                print("  ✓ Strengths:")
                for strength in strengths:
                    print(f"    - {strength}")
            
            if issues:
                print("  ⚠ Issues:")
                for issue in issues:
                    print(f"    - {issue}")
            
            print("  References:")
            for ref in refs:
                print(f"    Line {ref['line']}: {ref['context']}")
        else:
            print("  ✗ NOT REFERENCED IN TEXT")
        
        print()

if __name__ == '__main__':
    main()
