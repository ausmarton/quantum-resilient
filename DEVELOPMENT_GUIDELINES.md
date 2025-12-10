# Development Guidelines

**Date**: 2025-12-10  
**Status**: Active  
**Purpose**: Ground rules for making changes and managing TODO items safely and reliably

---

## Overview

This document establishes ground rules for development work to ensure changes are made safely, reliably, and without breaking existing functionality. All developers must follow these guidelines when implementing changes or adding TODO items.

---

## Core Principles

### 1. TODO Item Requirements

**Every TODO item must include:**

- **Clear Title**: Descriptive, specific title
- **Status**: Current status (pending, in_progress, completed, cancelled)
- **Priority**: Critical, High, Medium, Low
- **Context**: Sufficient detail to pick up work later without lost context:
  - **Problem Statement**: What problem does this solve?
  - **Current State**: What exists now?
  - **Expected Outcome**: What should the result be?
  - **Dependencies**: What blocks this or what does this block?
  - **Related Files**: All files that will be modified or created
  - **Testing Requirements**: How to verify the change works
  - **Acceptance Criteria**: What defines "done"?
- **Implementation Notes**: Any technical details, approaches, or considerations
- **Risk Assessment**: What could break? What needs special attention?

**Example Template:**
```markdown
### X. [Title]

**Status**: 🟡 **IN PROGRESS**  
**Priority**: High  
**Blocks**: Item #Y  
**Depends on**: Item #Z

**Problem Statement**: 
[Clear description of the problem]

**Current State**: 
[What exists now, what's missing, what's broken]

**Expected Outcome**: 
[What should exist after completion]

**Implementation Plan**:
[Step-by-step approach]

**Related Files**:
- `path/to/file1` - [what will change]
- `path/to/file2` - [what will change]

**Testing Requirements**:
- [ ] Test case 1
- [ ] Test case 2
- [ ] Integration test

**Acceptance Criteria**:
- [ ] Criterion 1
- [ ] Criterion 2

**Risk Assessment**:
- **High Risk**: [What could break]
- **Mitigation**: [How to prevent issues]
```

---

### 2. Testing Requirements

**Before marking any TODO item as completed:**

1. **Unit Tests**: If applicable, add/update unit tests
2. **Integration Tests**: Test with real data/workflows
3. **Regression Tests**: Verify existing functionality still works
4. **Edge Cases**: Test boundary conditions and error cases
5. **Cross-Environment**: If applicable, test on native, minikube, GCP
6. **Documentation**: Update relevant documentation

**Testing Checklist:**
- [ ] Code compiles/builds without errors
- [ ] Existing tests pass
- [ ] New tests added (if applicable)
- [ ] Manual testing completed
- [ ] Edge cases tested
- [ ] Error handling tested
- [ ] Documentation updated
- [ ] No regressions introduced

**Test Evidence**: Include test results, logs, or screenshots in the TODO item when marking complete.

---

### 3. Requirements Compliance

**All changes must comply with `docs/REQUIREMENTS_SPECIFICATION.md`:**

1. **Check Requirements First**: Before implementing, verify the change aligns with requirements
2. **Update Requirements**: If requirements need updating, update `REQUIREMENTS_SPECIFICATION.md` first
3. **Traceability**: Link TODO items to specific requirements (FR#, NFR#, etc.)
4. **Impact Assessment**: Document how the change affects requirements coverage

**Requirements Checklist:**
- [ ] Change aligns with existing requirements
- [ ] No requirements violated
- [ ] Requirements updated if needed
- [ ] Requirements traceability documented
- [ ] Impact on requirements coverage assessed

---

### 4. Change Safety Practices

**Additional practices to ensure safe, reliable changes:**

#### A. Impact Analysis

Before making changes:
- **Identify Affected Areas**: List all components/files that will be affected
- **Dependency Check**: Understand what depends on the code being changed
- **Backward Compatibility**: Ensure changes don't break existing functionality
- **Data Migration**: If data structures change, plan migration path

#### B. Incremental Changes

- **Small, Focused Changes**: Prefer multiple small changes over one large change
- **Atomic Commits**: Each commit should be a complete, testable unit
- **Feature Flags**: Use feature flags for major changes to enable rollback
- **Progressive Enhancement**: Add new features without breaking existing ones

#### C. Code Review Process

- **Self-Review**: Review your own changes before marking complete
- **Documentation Review**: Ensure documentation matches implementation
- **Test Review**: Verify tests are comprehensive
- **Requirements Review**: Confirm requirements compliance

#### D. Rollback Plan

- **Reversible Changes**: Design changes to be reversible
- **Backup Strategy**: Backup data/config before major changes
- **Rollback Steps**: Document how to rollback if issues occur
- **Version Control**: Use branches for experimental changes

#### E. Documentation Updates

- **Code Comments**: Add/update code comments for complex logic
- **API Documentation**: Update API docs if interfaces change
- **User Documentation**: Update user-facing docs if workflows change
- **Changelog**: Document changes in appropriate changelog

#### F. Validation & Verification

- **Pre-commit Checks**: Run linters, formatters, tests before committing
- **Post-commit Verification**: Verify changes work in target environment
- **Smoke Tests**: Run quick smoke tests after changes
- **Integration Verification**: Verify integration with other components

---

## Workflow

### When to Add a TODO Item

**MANDATORY**: Add a TODO item to `TODO.md` whenever you encounter:

1. **Issues During Work**:
   - Bugs or errors discovered during implementation
   - Unexpected behavior or edge cases
   - Missing functionality or incomplete features
   - Performance problems or optimization opportunities
   - Security concerns or vulnerabilities
   - Data quality issues or validation failures

2. **Additional Work Identified**:
   - Improvements that would enhance the codebase
   - Refactoring opportunities discovered
   - Documentation gaps or unclear areas
   - Test coverage gaps
   - Missing error handling or edge case coverage
   - Technical debt or code quality issues
   - Requirements gaps or missing capabilities

3. **Dependencies or Blockers**:
   - Work that blocks other items
   - Prerequisites needed before starting work
   - External dependencies or constraints

**DO NOT**:
- ❌ Fix issues ad-hoc without tracking them
- ❌ Leave work items undocumented
- ❌ Assume you'll remember to do it later
- ❌ Skip adding TODOs because "it's small" or "quick"
- ❌ Make changes to unrelated code while investigating something else

**Rationale**: 
- Ensures nothing gets lost or forgotten
- Enables proper prioritization and planning
- Provides visibility into all outstanding work
- Allows tracking of work over time
- Enables others to pick up work if needed
- Prevents scope creep and maintains focus on current work

### Special Case: Issues Discovered During Investigation

**CRITICAL**: When investigating or working on a TODO item and you notice something else that needs to be fixed:

**✅ CORRECT APPROACH**:
1. **Document First**: Immediately add a new TODO item to `TODO.md` with full context
2. **Continue Current Work**: Complete the investigation/work you were doing
3. **Prioritize Later**: The new TODO can be prioritized and addressed separately
4. **Maintain Focus**: Don't switch context mid-investigation

**❌ INCORRECT APPROACH**:
- Making ad-hoc fixes to unrelated code during investigation
- Fixing issues "while you're at it" without tracking
- Assuming you'll remember to fix it later
- Expanding scope of current work to include unrelated fixes

**Why This Matters**:
- **Prevents Lost Context**: Issues identified during investigation are documented before context is lost
- **Maintains Focus**: Keeps current work focused and prevents scope creep
- **Enables Prioritization**: Documented issues can be properly prioritized against other work
- **Prevents Rushed Fixes**: Ensures fixes are done properly with full context, not as quick side-fixes
- **Traceability**: Creates a record of when/why issues were identified

**Example Scenario**:

**Context**: Investigating why Item #18 (smoke test) is failing

**❌ WRONG**:
```
While investigating smoke test failure, notice a typo in another file.
→ Fix the typo immediately
→ Continue smoke test investigation
→ Forget to document the typo fix
```

**✅ CORRECT**:
```
While investigating smoke test failure, notice a typo in another file.
→ Add TODO item: "Item #20: Fix typo in X file (discovered during Item #18 investigation)"
→ Continue smoke test investigation (maintain focus)
→ Later: Prioritize Item #20 and fix it properly with full context
```

**Key Principle**: 
> **Document first, fix later**. The goal is not to prevent fixes, but to ensure nothing identified gets lost and all work is properly prioritized and tracked.

**This applies especially to AI assistants and automated tools**:
- When investigating a TODO item and discovering unrelated issues, **document them immediately** in `TODO.md`
- Don't make ad-hoc fixes to unrelated code during investigation
- Don't expand the scope of current work to include unrelated fixes
- The primary concern is **not losing sight of identified issues** - once documented, they can be prioritized and fixed properly later
- Maintaining focus on current work prevents scope creep and ensures quality

### Adding a TODO Item

**Process**:

1. **Check Existing TODOs**: Search `TODO.md` to ensure it's not already tracked
   - Use grep or search to check for similar items
   - If similar item exists, consider updating it rather than creating duplicate

2. **Gather Context**: Collect all relevant information before writing
   - What exactly is the issue/work?
   - Where does it occur? (files, functions, environments)
   - What's the impact? (critical, high, medium, low)
   - What's needed to fix/implement it?
   - Any related issues or dependencies?

3. **Check Requirements**: Verify alignment with `docs/REQUIREMENTS_SPECIFICATION.md`
   - Does this relate to a functional requirement (FR#)?
   - Does this relate to a non-functional requirement (NFR#)?
   - Should requirements be updated first?

4. **Write TODO**: Use the template below, include all required sections
   - Follow the format in "TODO Item Requirements" section
   - Assign next available item number
   - Include sufficient context for future pick-up

5. **Link Dependencies**: Document what blocks/is blocked by this item
   - Check if other TODO items depend on this
   - Check if this depends on other items
   - Update dependency chains in "Work Order & Dependencies" section

6. **Set Priority**: Based on impact and urgency
   - **Critical**: Blocks dissertation or critical functionality
   - **High**: Important for quality or functionality
   - **Medium**: Recommended improvement
   - **Low**: Nice-to-have or optional

7. **Update TODO.md Header**: Add to "Recent Additions" section
   - Include item number, title, and date
   - Example: `- Item #20: [Title] (YYYY-MM-DD)`

### Implementing a TODO Item

**During Implementation**:

1. **Read TODO**: Understand full context and requirements
2. **Check Requirements**: Verify compliance with `REQUIREMENTS_SPECIFICATION.md`
3. **Plan Implementation**: Break down into steps
4. **Impact Analysis**: Identify what could break
5. **Implement**: Make changes incrementally
   - **⚠️ IMPORTANT**: If you encounter issues or identify additional work during implementation, **STOP** and add them to `TODO.md` before continuing
   - **⚠️ CRITICAL**: If investigating or working on something and you notice unrelated issues, **document them in TODO.md immediately** - don't fix them ad-hoc
   - Don't create "quick fixes" that aren't tracked
   - Don't skip proper tracking because you're "almost done"
   - Don't expand scope to include unrelated fixes - document them for later prioritization
6. **Test**: Follow testing checklist
7. **Document**: Update code, tests, and documentation
8. **Verify**: Ensure requirements compliance
9. **Mark Complete**: Update TODO with completion details and test evidence

### Marking a TODO Complete

1. **Verify Testing**: All tests pass, manual testing complete
2. **Verify Requirements**: Requirements compliance confirmed
3. **Update Status**: Change status to ✅ **COMPLETED**
4. **Add Completion Date**: Record when completed
5. **Document Results**: Include test results, evidence
6. **Update Related Docs**: Update `REQUIREMENTS_SPECIFICATION.md` if needed
7. **Check Dependencies**: Update any items that depended on this

---

## File Naming & Organization

- **TODO Items**: Tracked in `TODO.md` (renamed from `OUTSTANDING_WORK.md` on 2025-12-10)
- **Requirements**: Defined in `docs/REQUIREMENTS_SPECIFICATION.md`
- **Guidelines**: This document (`DEVELOPMENT_GUIDELINES.md`)

**Note**: All references to `OUTSTANDING_WORK.md` have been updated to `TODO.md` throughout the codebase.

---

## Examples

### Good TODO Item

```markdown
### 4. Generate Missing Summary Files

**Status**: 🟡 **READY TO EXECUTE**  
**Priority**: Medium  
**Blocked by**: pandas installation (or containerization - Item #11)

**Problem Statement**: 
14 experiments have raw data but are missing `summary.json` files, preventing complete analysis.

**Current State**: 
- Script exists: `scripts/generate_missing_summaries.sh`
- Script can identify missing summaries (no pandas required)
- Script requires pandas to generate summaries
- Blocked: Need pandas installation or containerization

**Expected Outcome**: 
- All 14 missing summary files generated
- Complete data for aggregation and analysis
- All figures include complete data

**Implementation Plan**:
1. Install pandas or use containerization (Item #11)
2. Run `./scripts/generate_missing_summaries.sh`
3. Verify all summaries generated
4. Re-run aggregation

**Related Files**:
- `scripts/generate_missing_summaries.sh` - Script to generate summaries
- `analysis/aggregate_results.py` - Aggregation script
- `final-results/index.json` - Experiment index

**Testing Requirements**:
- [ ] Verify script identifies missing summaries correctly
- [ ] Run script and verify summaries generated
- [ ] Verify summary.json files are valid JSON
- [ ] Verify aggregation includes new summaries
- [ ] Check that all experiments now have summaries

**Acceptance Criteria**:
- [ ] All 14 missing summaries generated
- [ ] All summaries are valid JSON
- [ ] Aggregation includes new summaries
- [ ] No errors in generation process

**Risk Assessment**:
- **Low Risk**: Script already exists and tested
- **Mitigation**: Verify summaries before aggregation
```

### Bad TODO Item

```markdown
### 4. Fix summaries

**Status**: pending

**Issue**: Some summaries missing

**Fix**: Generate them
```

**Problems**: No context, no testing requirements, no acceptance criteria, no risk assessment.

---

## Enforcement

- **Self-Enforcement**: Developers are responsible for following these guidelines
- **Mandatory TODO Tracking**: All issues and additional work **MUST** be added to `TODO.md`
- **Review Process**: TODO items should be reviewed for completeness
- **Periodic Review**: Regularly review `TODO.md` to ensure all discovered work is tracked
- **Continuous Improvement**: Guidelines should be updated based on lessons learned

**Checking TODO Coverage**:

Before marking work complete, verify:
- [ ] All issues encountered during work are tracked in `TODO.md`
- [ ] All additional work identified is tracked in `TODO.md`
- [ ] No ad-hoc fixes were made without tracking
- [ ] All dependencies are documented

**Example Scenarios**:

**Scenario 1: Bug Found During Implementation**
```
❌ WRONG: Fix the bug immediately, don't track it
✅ CORRECT: Add TODO item for the bug, then fix it (or fix it and add TODO for verification)
```

**Scenario 2: Missing Feature Discovered**
```
❌ WRONG: Note it mentally, implement later without tracking
✅ CORRECT: Add TODO item with full context, prioritize, implement when appropriate
```

**Scenario 3: Code Quality Issue**
```
❌ WRONG: Fix it inline without tracking
✅ CORRECT: Add TODO item, then fix it (or add TODO for future refactoring)
```

**Scenario 4: Documentation Gap**
```
❌ WRONG: Fix documentation without tracking
✅ CORRECT: Add TODO item, then update documentation (or add TODO for documentation update)
```

**Scenario 5: Unrelated Issue Found During Investigation** ⭐ **CRITICAL**
```
❌ WRONG: 
  - Investigating Item #18 (smoke test)
  - Notice typo in unrelated file
  - Fix typo immediately "while you're at it"
  - Continue smoke test investigation
  - Typo fix is undocumented and untracked

✅ CORRECT:
  - Investigating Item #18 (smoke test)
  - Notice typo in unrelated file
  - STOP: Add TODO item immediately: "Item #20: Fix typo in X file (discovered during Item #18 investigation)"
  - Continue smoke test investigation (maintain focus)
  - Later: Prioritize Item #20 and fix it properly with full context
```

**Key Principle for Scenario 5**:
> **Document first, fix later**. The primary goal is to ensure nothing identified gets lost. Once documented, issues can be properly prioritized and fixed with appropriate context, rather than being rushed as side-fixes during unrelated work.

---

## Related Documentation

- **[TODO.md](TODO.md)** - All outstanding work items
- **[Requirements Specification](docs/REQUIREMENTS_SPECIFICATION.md)** - Requirements and capabilities
- **[Test Coverage](docs/reference/test-coverage.md)** - Testing strategy and coverage

---

---

## Quick Reference: TODO Tracking Checklist

**Before Starting Work**:
- [ ] Check `TODO.md` for existing related items
- [ ] Review `DEVELOPMENT_GUIDELINES.md` for process

**During Work**:
- [ ] If issue encountered → Add to `TODO.md` immediately
- [ ] If additional work identified → Add to `TODO.md` immediately
- [ ] If unrelated issue noticed during investigation → Add to `TODO.md` immediately (don't fix ad-hoc)
- [ ] Don't create untracked "quick fixes"
- [ ] Don't expand scope to include unrelated fixes - document them for later

**Before Completing Work**:
- [ ] Verify all discovered issues/work are in `TODO.md`
- [ ] Update TODO item status appropriately
- [ ] Link dependencies if needed

**After Completing Work**:
- [ ] Mark TODO as completed with evidence
- [ ] Update `REQUIREMENTS_SPECIFICATION.md` if needed
- [ ] Update related documentation

---

**Last Updated**: 2025-12-10  
**Maintainer**: Update when guidelines change or new practices are established
