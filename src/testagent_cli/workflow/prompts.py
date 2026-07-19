"""
Shared prompt components for workflow stages.
Ensures consistency across all stages.
"""

# Workflow overview that all stages should include
WORKFLOW_OVERVIEW = """
## 🔄 Complete 7-Stage Test Generation Workflow

You are an AI agent in a **structured workflow** for generating tests for Ascend 910B operators.

### All Stages & Artifacts (in order)

1. **Understand Function** → Generates `function_doc.md`
   - Explore project, read operator code, document function details
   
2. **Generate Requirements** → Generates `requirements.md`
   - Read `function_doc.md`, define test requirements
   
3. **Design Test Plan** → Generates `test_plan.md`
   - Read `function_doc.md` + `requirements.md`, design test cases
   
4. **Generate Code** → Generates test code file at `{test_file_path}`
   - Read all previous artifacts, write C++ test code
   
5. **Execute Tests** → Generates execution logs
   - Build and run tests, capture results
   
6. **Analyze Results** → Generates `analysis.md`
   - Analyze coverage and results
   
7. **Generate Report** → Generates `final_report.md`
   - Summarize entire process

---
"""

def get_stage_header(stage_number: int, stage_name: str, inputs: list, output: str) -> str:
    """Generate standardized header for each stage."""
    inputs_str = ", ".join(f"`{i}`" for i in inputs) if inputs else "None"
    
    return f"""
### 📍 Your Current Stage: Stage {stage_number} - {stage_name}

**Input Artifacts**: {inputs_str}
**Output Artifact**: `{output}`
**Operator**: {{op}}
**Architecture**: {{arch}}

"""

# Standard artifact structure requirements
ARTIFACT_STRUCTURES = {
    "function_doc.md": """
## Required Structure for function_doc.md

Your output MUST follow this structure:

```markdown
# {Operator} Operator - Function Documentation

## 1. Basic Information
- **Operator Name**: {op}
- **Implementation File**: `path/to/file.cpp`
- **Header Files**: List all relevant headers
- **Existing Test File**: `path/to/existing/test.cpp` (if any)

## 2. Function Purpose
Brief description of what this operator does.

## 3. API Signature
```cpp
// Main function signature
returnType functionName(params...);
```

## 4. Key Parameters
- **param1**: Description, type, constraints
- **param2**: Description, type, constraints

## 5. Key Logic Summary
- Logic point 1
- Logic point 2
- Critical implementation details

## 6. Dependencies
- Required headers
- External libraries
- Build requirements
```

**IMPORTANT**: This structure ensures Stage 2 (Requirements) has all needed information.
""",
    
    "requirements.md": """
## Required Structure for requirements.md

```markdown
# Test Requirements for {Operator}

## Reference
- **Based on**: `function_doc.md`
- **Operator**: {op}
- **Platform**: {arch}

## 1. Functional Requirements
FR-01: [Requirement description]
FR-02: [Requirement description]

## 2. Input/Output Requirements
- Valid inputs to test
- Expected outputs
- Edge cases

## 3. Coverage Requirements
- Code coverage goals
- Scenario coverage

## 4. Performance Requirements
(if applicable)
```
""",
    
    "test_plan.md": """
## Required Structure for test_plan.md

```markdown
# Test Plan for {Operator}

## Reference
- **Based on**: `function_doc.md`, `requirements.md`
- **Operator**: {op}

## 1. Test Strategy
Overview of testing approach

## 2. Test Cases

### TC-01: [Test Case Name]
- **Objective**: What this tests
- **Input**: Specific test data
- **Expected Output**: Expected result
- **Priority**: High/Medium/Low

### TC-02: [Test Case Name]
...

## 3. Test Data
Specific datasets to use
```
"""
}


def get_artifact_structure_requirement(artifact_name: str) -> str:
    """Get the structure requirement for a specific artifact."""
    return ARTIFACT_STRUCTURES.get(artifact_name, "")
