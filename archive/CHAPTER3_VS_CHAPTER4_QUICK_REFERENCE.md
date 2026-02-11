# Chapter 3 vs Chapter 4: Quick Reference Guide

**Quick Decision Tool**: Ask yourself: "Is this about HOW I did it, or WHAT I found?"

---

## ONE-SENTENCE SUMMARY

- **Chapter 3**: Describes HOW the research was designed and conducted (methodology, framework, procedures, validation)
- **Chapter 4**: Presents WHAT was discovered (data, results, analysis, findings, interpretation)

---

## DECISION TREE

```
Is this content about...
│
├─ HOW research was designed/conducted?
│  └─ YES → Chapter 3
│
├─ WHAT was found/discovered?
│  └─ YES → Chapter 4
│
├─ Framework design/architecture?
│  └─ YES → Chapter 3 (with methodological justification)
│
└─ Actual data/results/measurements?
   └─ YES → Chapter 4
```

---

## CONTENT CHECKLIST

### ✅ BELONGS IN CHAPTER 3

- [ ] Research methodology overview
- [ ] Framework architecture (with methodological justification)
- [ ] Data collection procedures (HOW data will be collected)
- [ ] Statistical methods (WHICH methods, WHY chosen)
- [ ] Measurement methodology (HOW measurements taken)
- [ ] Validation activities (HOW validation conducted)
- [ ] Threats to validity (HOW validity ensured)
- [ ] Justification for method selection
- [ ] Mapping of objectives to methodology
- [ ] Ethical considerations

**Language**: "The framework enables...", "Data collection will employ...", "Statistical analysis will utilize..."

---

### ✅ BELONGS IN CHAPTER 4

- [ ] Actual performance measurements
- [ ] Statistical test results (p-values, test statistics)
- [ ] Effect sizes (Cohen's d, confidence intervals)
- [ ] Visualizations (CDFs, charts, tables with actual data)
- [ ] Data interpretation and findings
- [ ] Patterns and trends in data
- [ ] Comparative analysis results
- [ ] Findings addressing research objectives
- [ ] Context and limitations of findings

**Language**: "The results show...", "Analysis revealed...", "Statistical tests found...", "Measurements indicate..."

---

### ❌ DOES NOT BELONG IN EITHER

- [ ] Implementation details (function names, file formats) → Appendix or remove
- [ ] Detailed code descriptions → Appendix
- [ ] Directory structures → Remove or Appendix
- [ ] Literature review content → Chapter 2
- [ ] Future work → Chapter 5 or Conclusion

---

## COMMON MISTAKES

### Mistake 1: Results in Chapter 3
❌ "The framework measured Kyber-512 latency at 2.3μs..."  
✅ "The framework enables measurement of cryptographic operation latency with nanosecond precision..."

### Mistake 2: Methodology in Chapter 4
❌ "We designed the framework with five layers..."  
✅ "The comprehensive telemetry collection enabled detailed performance characterization..."

### Mistake 3: Implementation Details in Chapter 4
❌ "Latency measurements employ Rust's `Instant::now()` function..."  
✅ "Latency measurements were captured with nanosecond precision..."

### Mistake 4: Statistical Methods vs Results
❌ (Chapter 4) "We used t-tests to compare algorithms..."  
✅ (Chapter 3) "Statistical analysis employs independent samples t-tests..."  
✅ (Chapter 4) "Independent samples t-tests revealed statistically significant differences (t=4.32, p<0.01)..."

---

## TEMPORAL DISTINCTION

| Aspect | Chapter 3 | Chapter 4 |
|--------|-----------|-----------|
| **Timeframe** | Before/during research design | During/after data collection |
| **Tense** | Future/present (planning) | Past/present (completed) |
| **Focus** | "How will we do it?" | "What did we find?" |

---

## VALIDATION QUESTIONS

### Chapter 3 Validation
- [ ] Could another researcher replicate your research using this chapter?
- [ ] Does it describe HOW, not WHAT?
- [ ] Does it avoid presenting actual results?
- [ ] Does it justify methodological choices?

### Chapter 4 Validation
- [ ] Could someone understand your findings using this chapter?
- [ ] Does it present WHAT was found, not HOW?
- [ ] Does it include actual data/results?
- [ ] Does it interpret findings in context?

---

## QUICK REFERENCE TABLE

| Topic | Chapter 3 | Chapter 4 |
|-------|-----------|-----------|
| Framework architecture | ✅ Detailed with justification | ❌ Brief reference only |
| Data collection | ✅ How data will be collected | ✅ What data was collected |
| Statistical methods | ✅ Which methods, why | ✅ Actual test results |
| Performance measurements | ✅ How measurements taken | ✅ Actual measured values |
| Validation | ✅ How validation conducted | ❌ Results only if needed |
| Visualizations | ❌ What types will be used | ✅ Actual figures with data |
| Findings | ❌ None | ✅ All findings |

---

## LANGUAGE PATTERNS

### Chapter 3 Language
- "The framework enables..."
- "Data collection will employ..."
- "Statistical analysis will utilize..."
- "The methodology controls for..."
- "The framework is designed to..."

### Chapter 4 Language
- "The results show..."
- "Analysis revealed..."
- "Statistical tests found..."
- "Measurements indicate..."
- "Findings suggest..."

---

## FINAL CHECK

**Before submitting, ask:**

1. **Chapter 3**: Does this describe the research design and procedures? ✅
2. **Chapter 4**: Does this present actual findings and results? ✅
3. **No overlap**: Are methodology details in Chapter 3, not Chapter 4? ✅
4. **No gaps**: Are all findings in Chapter 4, not Chapter 3? ✅

---

**Remember**: Chapter 3 = HOW | Chapter 4 = WHAT

