# Objective Traceability Guidelines

**Principle**: Maintain clear connections between objectives, methodology, and results without using meta-language.

---

## How to Show Objective-Methodology-Results Connections

### ✅ Good Approaches (Direct, Not Meta)

#### 1. **Integrate into Description**
Instead of: "This component addresses Objective 1 by establishing criteria..."
Use: "Systematic literature analysis established criteria for algorithm selection based on NIST standardisation status, implementation maturity, and documented empirical performance."

#### 2. **Use Section References**
Instead of: "This addresses Objective 1..."
Use: "Algorithm selection criteria (Section 2.2.3) were validated through framework integration (Section 3.3.4), confirming that selected algorithms could be successfully integrated within real-time streaming contexts."

#### 3. **Show Through Results**
Instead of: "This objective was achieved by..."
Use: "The selected algorithms, Kyber-512 and Dilithium-2, demonstrated performance characteristics that align with real-time system requirements (Section 4.2.1), validating the selection criteria established in Section 2.2.3."

#### 4. **Use Table Structure**
Tables can show connections without meta-language:
- Table 3.1: Objective → Methodology → Metrics → Statistical Method
- This is structural, not meta-commentary

#### 5. **Cross-Reference in Results**
In Chapter 4, when presenting results:
- "Performance evaluation (Objective 3) revealed..."
- "Comparative analysis (Objective 4) demonstrated..."
- These are factual statements about what was done, not meta-commentary

---

## ❌ Avoid: Meta-Statements

- ❌ "This component addresses Objective X by..."
- ❌ "This ensures that Objective Y is met..."
- ❌ "The methodology was designed to address..."
- ❌ "This explicit mapping ensures traceability..."

---

## ✅ Use Instead: Factual Statements

- ✅ "Systematic literature analysis established criteria..."
- ✅ "Framework validation confirmed successful integration..."
- ✅ "Performance evaluation revealed..."
- ✅ "Statistical analysis demonstrated..."

---

## Traceability Through Structure

### Chapter 3 Structure:
- **Section 3.1.5**: Table mapping objectives to methodology (structural, not meta)
- **Section 3.2.1**: Describe what was done for each objective (factual)
- **Section 3.3**: Detailed procedures (factual descriptions)

### Chapter 4 Structure:
- **Section 4.3**: "Interpretation in relation to the objectives" - This section explicitly discusses objectives (appropriate place for objective references)
- Results sections: Present findings, then in Section 4.3 connect to objectives

---

## Example: Good Objective Traceability

**In Section 3.2.1:**
"Systematic literature analysis (Section 3.1.1) identified candidate algorithms based on NIST standardisation status, implementation maturity, and documented empirical performance. Framework validation (Section 3.3.4) confirmed successful integration of selected algorithms within real-time streaming contexts."

**In Section 4.3.1 (Objective 1):**
"The selected algorithms, Kyber-512 and Dilithium-2, demonstrated performance characteristics that align with real-time system requirements (Section 4.2.1). Both algorithms are NIST-standardised (ML-KEM/Kyber and ML-DSA/Dilithium), providing security strength equivalent to classical algorithms, and were successfully integrated into the experimental framework (Section 3.3.3), demonstrating implementation maturity and industry readiness."

**Key**: The connection is shown through:
1. What was done (factual)
2. Section references (structural)
3. Results that validate (factual)
4. No meta-commentary about "addressing objectives"

---

## Checklist for Objective Traceability

- [ ] Each objective has corresponding methodology described (Section 3.1, 3.2, 3.3)
- [ ] Each objective has corresponding results presented (Chapter 4)
- [ ] Section 4.3 explicitly discusses each objective (appropriate place for objective references)
- [ ] Cross-references use section numbers, not meta-statements
- [ ] Table 3.1 provides structural mapping (no meta-commentary needed)
- [ ] No statements like "This addresses/ensures/achieves objective X"

