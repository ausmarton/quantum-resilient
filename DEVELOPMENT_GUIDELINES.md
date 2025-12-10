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

### Adding a TODO Item

1. **Check Existing TODOs**: Ensure it's not already tracked
2. **Gather Context**: Collect all relevant information
3. **Check Requirements**: Verify alignment with requirements
4. **Write TODO**: Use template, include all required sections
5. **Link Dependencies**: Document what blocks/is blocked by this item
6. **Set Priority**: Based on impact and urgency

### Implementing a TODO Item

1. **Read TODO**: Understand full context and requirements
2. **Check Requirements**: Verify compliance with `REQUIREMENTS_SPECIFICATION.md`
3. **Plan Implementation**: Break down into steps
4. **Impact Analysis**: Identify what could break
5. **Implement**: Make changes incrementally
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
- **Review Process**: TODO items should be reviewed for completeness
- **Continuous Improvement**: Guidelines should be updated based on lessons learned

---

## Related Documentation

- **[TODO.md](TODO.md)** - All outstanding work items
- **[Requirements Specification](docs/REQUIREMENTS_SPECIFICATION.md)** - Requirements and capabilities
- **[Test Coverage](docs/reference/test-coverage.md)** - Testing strategy and coverage

---

**Last Updated**: 2025-12-10  
**Maintainer**: Update when guidelines change or new practices are established
