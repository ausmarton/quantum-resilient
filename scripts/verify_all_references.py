#!/usr/bin/env python3
"""
Comprehensive verification of ALL references in the dissertation:
- Section references
- Figure references  
- Table references

Also verifies that section numbering matches the actual document structure.
"""

import re
from collections import defaultdict

def extract_all_references(doc_path):
    """Extract all references (sections, figures, tables) from the document."""
    with open(doc_path, 'r') as f:
        lines = f.readlines()
    
    references = {
        'sections': [],
        'figures': [],
        'tables': []
    }
    
    for i, line in enumerate(lines, 1):
        # Section references
        for match in re.finditer(r'Section\s+([0-9]+\.[0-9]+(?:\.[0-9]+)?)', line, re.IGNORECASE):
            references['sections'].append((i, match.group(1), line.strip()[:80]))
        
        # Figure references (in text, not captions)
        for match in re.finditer(r'(?:^|[^a-zA-Z])Figure\s+([0-9]+\.[0-9]+[a-z]?|[0-9]+[a-z]?)(?:[^a-zA-Z]|$)', line, re.IGNORECASE):
            fig_ref = match.group(1)
            # Skip if it's in a caption (starts with *** or *)
            if not re.match(r'^\s*\*+\s*Figure', line, re.IGNORECASE):
                references['figures'].append((i, fig_ref, line.strip()[:80]))
        
        # Table references
        for match in re.finditer(r'(?:^|[^a-zA-Z])Table\s+([0-9]+\.[0-9]+[a-z]?|[0-9]+[a-z]?)(?:[^a-zA-Z]|$)', line, re.IGNORECASE):
            table_ref = match.group(1)
            # Skip if it's in a caption
            if not re.match(r'^\s*\*+\s*Table', line, re.IGNORECASE):
                references['tables'].append((i, table_ref, line.strip()[:80]))
    
    return references, lines

def extract_defined_items(lines):
    """Extract all defined figures and tables from the document."""
    figures = {}
    tables = {}
    
    for i, line in enumerate(lines, 1):
        # Figure definitions (captions) - handle various formats:
        # *Figure 3.1:, **Figure 3.2**:, ***Figure 4.1**:, Figure 4.2a:
        fig_match = re.search(r'(?:^\s*\*+\s*|^\s*)Figure\s+([0-9]+\.[0-9]+[a-z]?|[0-9]+[a-z]?)\s*[:*]', line, re.IGNORECASE)
        if fig_match:
            fig_num = fig_match.group(1)
            figures[fig_num] = i
        
        # Table definitions - handle various formats:
        # **Table 2.1**:, **Table 3.2**:, **Table 4.6**
        table_match = re.search(r'(?:^\s*\*+\s*|^\s*)Table\s+([0-9]+\.[0-9]+[a-z]?|[0-9]+[a-z]?)\s*[:*]', line, re.IGNORECASE)
        if table_match:
            table_num = table_match.group(1)
            tables[table_num] = i
    
    return figures, tables

def verify_section_structure(lines):
    """Verify that section numbering matches the actual document structure."""
    issues = []
    current_chapter = None
    chapter_num = None
    section_counter = 0
    subsection_counter = defaultdict(int)
    
    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()
        
        # Detect chapters
        if re.match(r'^#\s+(Chapter\s+[0-9]|Analysis and interpretation|Conclusions)', line_stripped, re.IGNORECASE):
            match = re.match(r'^#\s+(.+?)(?:\s*\{#|$)', line_stripped)
            if match:
                current_chapter = match.group(1).strip()
                if 'Chapter 1' in current_chapter:
                    chapter_num = '1'
                elif 'Chapter 2' in current_chapter:
                    chapter_num = '2'
                    section_counter = 0
                elif 'Chapter 3' in current_chapter:
                    chapter_num = '3'
                    section_counter = 0
                elif 'Analysis and interpretation' in current_chapter:
                    chapter_num = '4'
                    section_counter = 0
                elif 'Conclusions' in current_chapter:
                    chapter_num = '5'
                    section_counter = 0
        
        # Detect major sections (##)
        elif line_stripped.startswith('## ') and chapter_num:
            match = re.match(r'^##\s+(.+?)(?:\s*\{#|$)', line_stripped)
            if match:
                text = match.group(1).strip()
                # Check if heading has explicit number
                num_match = re.search(r'^([0-9]+)\s+(.+)', text)
                if num_match:
                    explicit_num = num_match.group(1)
                    section_counter += 1
                    expected = f"{chapter_num}.{section_counter}"
                    actual = f"{chapter_num}.{explicit_num}"
                    if expected != actual:
                        issues.append(f"Line {i}: Section numbering mismatch. Expected {expected}, found {actual} in heading")
                else:
                    section_counter += 1
                    subsection_counter[section_counter] = 0
        
        # Detect subsections (###)
        elif line_stripped.startswith('### ') and chapter_num:
            match = re.match(r'^###\s+(.+?)(?:\s*\{#|$)', line_stripped)
            if match:
                text = match.group(1).strip()
                text = re.sub(r'\*\*', '', text)
                num_match = re.search(r'^([0-9]+\.[0-9]+(?:\.[0-9]+)?)\s+(.+)', text)
                if num_match:
                    explicit_num = num_match.group(1)
                    # Verify it matches expected structure
                    parts = explicit_num.split('.')
                    if len(parts) == 3:
                        ch, sec, sub = parts
                        if ch != chapter_num:
                            issues.append(f"Line {i}: Subsection {explicit_num} in wrong chapter (Chapter {chapter_num})")
                    elif len(parts) == 2:
                        ch, sec = parts
                        if ch != chapter_num:
                            issues.append(f"Line {i}: Subsection {explicit_num} in wrong chapter (Chapter {chapter_num})")
                else:
                    if section_counter > 0:
                        subsection_counter[section_counter] += 1
    
    return issues

def main():
    doc_path = 'FERNANDES_H2807295_F87_dissertation (2).md'
    
    print("=" * 80)
    print("COMPREHENSIVE REFERENCE VERIFICATION")
    print("=" * 80)
    
    references, lines = extract_all_references(doc_path)
    figures, tables = extract_defined_items(lines)
    
    print(f"\n1. SECTION REFERENCES: {len(references['sections'])} found")
    print("-" * 80)
    # Use the existing verification script logic
    from verify_section_references import map_sections
    sections = map_sections(doc_path)
    
    section_issues = []
    for line_num, ref_num, context in references['sections']:
        if ref_num not in sections:
            section_issues.append((line_num, ref_num, context))
    
    if section_issues:
        print(f"✗ {len(section_issues)} invalid section references:")
        for line_num, ref_num, context in section_issues[:5]:
            print(f"  Line {line_num}: Section {ref_num} - {context}")
        if len(section_issues) > 5:
            print(f"  ... and {len(section_issues) - 5} more")
    else:
        print(f"✓ All {len(references['sections'])} section references are valid")
    
    print(f"\n2. FIGURE REFERENCES: {len(references['figures'])} found")
    print("-" * 80)
    figure_issues = []
    for line_num, fig_ref, context in references['figures']:
        if fig_ref not in figures:
            figure_issues.append((line_num, fig_ref, context))
    
    if figure_issues:
        print(f"✗ {len(figure_issues)} invalid figure references:")
        for line_num, fig_ref, context in figure_issues:
            print(f"  Line {line_num}: Figure {fig_ref} - {context}")
    else:
        print(f"✓ All {len(references['figures'])} figure references are valid")
    
    print(f"\n3. TABLE REFERENCES: {len(references['tables'])} found")
    print("-" * 80)
    table_issues = []
    for line_num, table_ref, context in references['tables']:
        if table_ref not in tables:
            table_issues.append((line_num, table_ref, context))
    
    if table_issues:
        print(f"✗ {len(table_issues)} invalid table references:")
        for line_num, table_ref, context in table_issues:
            print(f"  Line {line_num}: Table {table_ref} - {context}")
    else:
        print(f"✓ All {len(references['tables'])} table references are valid")
    
    print(f"\n4. DEFINED ITEMS:")
    print("-" * 80)
    print(f"  Figures defined: {sorted(figures.keys(), key=lambda x: (float(x.split('.')[0]) if '.' in x else float(x), x))}")
    print(f"  Tables defined: {sorted(tables.keys(), key=lambda x: (float(x.split('.')[0]) if '.' in x else float(x), x))}")
    
    print(f"\n5. SECTION STRUCTURE VERIFICATION:")
    print("-" * 80)
    structure_issues = verify_section_structure(lines)
    if structure_issues:
        print(f"✗ {len(structure_issues)} section structure issues:")
        for issue in structure_issues[:5]:
            print(f"  {issue}")
        if len(structure_issues) > 5:
            print(f"  ... and {len(structure_issues) - 5} more")
    else:
        print("✓ Section numbering structure appears consistent")
    
    print("\n" + "=" * 80)
    total_issues = len(section_issues) + len(figure_issues) + len(table_issues) + len(structure_issues)
    if total_issues == 0:
        print("✓ ALL REFERENCES AND STRUCTURE VERIFIED - NO ISSUES FOUND")
    else:
        print(f"✗ TOTAL ISSUES FOUND: {total_issues}")
    print("=" * 80)

if __name__ == '__main__':
    main()
