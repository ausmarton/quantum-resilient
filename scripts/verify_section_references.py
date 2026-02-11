#!/usr/bin/env python3
"""
Verify all section references in the dissertation document.

Maps the actual section structure and identifies:
1. Missing section references
2. Incorrect section references
3. Dead references that need to be fixed
"""

import re
from pathlib import Path
from collections import defaultdict

def map_sections(doc_path):
    """Map all sections in the document."""
    with open(doc_path, 'r') as f:
        lines = f.readlines()
    
    sections = {}  # section_num -> (line_num, chapter, title, level)
    current_chapter = None
    chapter_num = None
    
    # Track section numbering within each chapter
    chapter_sections = defaultdict(lambda: {'major': 0, 'subsections': defaultdict(int)})
    
    # Manual mapping for Chapter 4 (sections don't have explicit numbers in headings)
    chapter4_manual_map = {
        553: '4.1',  # "## Summary of data collected"
        555: '4.1.1',  # "### **Experimental Scope and Scale**"
        561: '4.1.2',  # "### **Data Completeness and Coverage**"
        576: '4.1.3',  # "### **Data Processing and Aggregation**"
        580: '4.2',  # "## Data Analysis"
        584: '4.2.1',  # "### **Algorithm Performance Comparison**"
        651: '4.2.2',  # "### **Statistical Hypothesis Testing**"
        673: '4.2.3',  # "### **Environment Comparison**"
        727: '4.2.4',  # "### **Payload Size and Workload Rate Impact**"
        772: '4.2.5',  # "### **Resource Utilisation Analysis**"
        806: '4.3',  # "## Interpretation in relation to the objectives"
        808: '4.3.1',  # "### **Objective 1**"
        814: '4.3.2',  # "### **Objective 2**"
        820: '4.3.3',  # "### **Objective 3**"
        826: '4.3.4',  # "### **Objective 4**"
        832: '4.3.5',  # "### **Objective 5**"
        840: '4.4',  # "## Interpretation in relation to the research aim"
    }
    
    for i, line in enumerate(lines, 1):
        line_stripped = line.strip()
        
        # Check manual Chapter 4 mapping first
        if i in chapter4_manual_map:
            section_num = chapter4_manual_map[i]
            title = line_stripped.replace('###', '').replace('##', '').replace('**', '').strip()
            sections[section_num] = (i, 'Analysis and interpretation', title, 2 if '##' in line_stripped else 3)
            continue
        
        # Detect chapters (# heading)
        if re.match(r'^#\s+(Chapter\s+[0-9]|Analysis and interpretation|Conclusions)', line_stripped, re.IGNORECASE):
            match = re.match(r'^#\s+(.+?)(?:\s*\{#|$)', line_stripped)
            if match:
                current_chapter = match.group(1).strip()
                # Extract chapter number
                if 'Chapter 1' in current_chapter:
                    chapter_num = '1'
                elif 'Chapter 2' in current_chapter:
                    chapter_num = '2'
                elif 'Chapter 3' in current_chapter:
                    chapter_num = '3'
                elif 'Analysis and interpretation' in current_chapter:
                    chapter_num = '4'
                elif 'Conclusions' in current_chapter:
                    chapter_num = '5'
                else:
                    chapter_num = None
        
        # Detect major sections (## heading) - skip if already in manual map
        elif line_stripped.startswith('## ') and i not in chapter4_manual_map:
            match = re.match(r'^##\s+(.+?)(?:\s*\{#|$)', line_stripped)
            if match:
                text = match.group(1).strip()
                # Check if it has an explicit number
                num_match = re.search(r'^([0-9]+)\s+(.+)', text)
                if num_match and chapter_num:
                    section_num = num_match.group(1)
                    title = num_match.group(2)
                    full_section = f"{chapter_num}.{section_num}"
                    sections[full_section] = (i, current_chapter, title, 2)
                    chapter_sections[chapter_num]['major'] = int(section_num)
                elif chapter_num:
                    # No explicit number - infer from order
                    chapter_sections[chapter_num]['major'] += 1
                    section_num = chapter_sections[chapter_num]['major']
                    full_section = f"{chapter_num}.{section_num}"
                    sections[full_section] = (i, current_chapter, text, 2)
        
        # Detect subsections (### heading) - skip if already in manual map
        elif line_stripped.startswith('### ') and i not in chapter4_manual_map:
            match = re.match(r'^###\s+(.+?)(?:\s*\{#|$)', line_stripped)
            if match:
                text = match.group(1).strip()
                text = re.sub(r'\*\*', '', text)  # Remove bold markers
                # Check for explicit subsection number
                num_match = re.search(r'^([0-9]+\.[0-9]+(?:\.[0-9]+)?)\s+(.+)', text)
                if num_match:
                    subsection_num = num_match.group(1)
                    title = num_match.group(2)
                    sections[subsection_num] = (i, current_chapter, title, 3)
                elif chapter_num and chapter_sections[chapter_num]['major'] > 0:
                    # Infer subsection number
                    major = chapter_sections[chapter_num]['major']
                    chapter_sections[chapter_num]['subsections'][major] += 1
                    minor = chapter_sections[chapter_num]['subsections'][major]
                    subsection_num = f"{chapter_num}.{major}.{minor}"
                    sections[subsection_num] = (i, current_chapter, text, 3)
    
    return sections

def find_references(doc_path):
    """Find all section references in the document."""
    with open(doc_path, 'r') as f:
        lines = f.readlines()
    
    references = []
    for i, line in enumerate(lines, 1):
        matches = re.finditer(r'Section\s+([0-9]+\.[0-9]+(?:\.[0-9]+)?)', line, re.IGNORECASE)
        for match in matches:
            ref_num = match.group(1)
            context = line.strip()[:100]
            references.append((i, ref_num, context))
    
    return references

def main():
    doc_path = Path('FERNANDES_H2807295_F87_dissertation (2).md')
    
    print("Mapping section structure...")
    sections = map_sections(doc_path)
    
    print(f"\nFound {len(sections)} sections:")
    print("=" * 80)
    for section_num in sorted(sections.keys(), key=lambda x: tuple(map(int, x.split('.')))):
        line_num, chapter, title, level = sections[section_num]
        indent = "  " * (level - 2) if level > 1 else ""
        print(f"{indent}Section {section_num}: {title[:60]}")
    
    print(f"\n\nFinding section references...")
    references = find_references(doc_path)
    
    # Categorize references
    valid = []
    missing = []
    
    for line_num, ref_num, context in references:
        if ref_num in sections:
            valid.append((line_num, ref_num, context))
        else:
            missing.append((line_num, ref_num, context))
    
    print(f"\n✓ VALID REFERENCES: {len(valid)}")
    print(f"✗ MISSING REFERENCES: {len(missing)}")
    
    if missing:
        print("\n\nMISSING SECTION REFERENCES:")
        print("=" * 80)
        missing_by_section = defaultdict(list)
        for line_num, ref_num, context in missing:
            missing_by_section[ref_num].append((line_num, context))
        
        for ref_num in sorted(missing_by_section.keys(), key=lambda x: tuple(map(int, x.split('.')))):
            print(f"\nSection {ref_num} (referenced {len(missing_by_section[ref_num])} times):")
            for line_num, context in missing_by_section[ref_num]:
                print(f"  Line {line_num}: {context}")

if __name__ == '__main__':
    main()
