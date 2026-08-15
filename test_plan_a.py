#!/usr/bin/env python3
"""
Test Plan A implementation - verify that _find_source_test_file and 
_ensure_skeleton correctly reuse existing source files instead of creating 
new ones with LLM-fabricated headers.
"""
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, '/mnt/fangcr/ATTest/src')

from attest_cli.workflow.ascend_stages import AscendCodeGenStage, AscendGenerationAgentLoopStage
from attest_cli.workflow.state import WorkflowState
from attest_cli.llm import LLMClient
from attest_cli.tools import ToolRunner, ToolContext, build_default_registry, ToolRegistry
import json

def test_find_source_test_file():
    """Test that _find_source_test_file correctly identifies source files"""
    print("🧪 Test 1: _find_source_test_file locates source files")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)
        
        # Setup: create math/sign/tests/ut/op_api directory
        op_api_dir = project_root / "math" / "sign" / "tests" / "ut" / "op_api"
        op_api_dir.mkdir(parents=True, exist_ok=True)
        
        # Create source test file
        source_file = op_api_dir / "test_aclnn_sign.cpp"
        source_file.write_text(
            "#include \"gtest/gtest.h\"\n"
            "#include \"aclnn/aclnn_sign.h\"\n"
            "\n"
            "class SignTest : public testing::Test {\n"
            "  void SetUp() override {}\n"
            "};\n"
            "\n"
            "TEST_F(SignTest, Basic) {\n"
            "  // existing test\n"
            "}\n"
        )
        
        # Create target attest path (doesn't exist yet)
        attest_path = op_api_dir / "test_aclnn_sign_attest.cpp"
        
        # Mock LLM and tool runner
        llm = LLMClient()
        tool_registry = build_default_registry()
        tool_runner = ToolRunner(tool_registry)
        
        # Create stage instance (Plan A logic is in AscendGenerationAgentLoopStage)
        stage = AscendGenerationAgentLoopStage(llm, tool_runner)
        
        # Test finding source file
        source = stage._find_source_test_file(project_root, attest_path, "sign")
        
        assert source is not None, "Should find source file"
        assert source == source_file, f"Should find test_aclnn_sign.cpp, got {source}"
        print("  ✓ Found source file correctly")
        
        # Test with .bak file
        bak_file = op_api_dir / "test_aclnn_sign.cpp.bak"
        bak_file.write_text("// backup")
        
        source2 = stage._find_source_test_file(project_root, attest_path, "sign")
        # Should still find original .cpp (preferred)
        assert source2 == source_file, "Should prefer original .cpp over .bak"
        print("  ✓ Prefers original .cpp over .bak")
        
        # Test with no source file
        empty_dir = project_root / "math" / "empty" / "tests" / "ut" / "op_api"
        empty_dir.mkdir(parents=True, exist_ok=True)
        empty_attest = empty_dir / "test_empty_attest.cpp"
        
        source3 = stage._find_source_test_file(project_root, empty_attest, "empty")
        assert source3 is None, "Should return None when no source file exists"
        print("  ✓ Returns None when no source file exists")
    
    print("✅ Test 1 passed\n")

def test_ensure_skeleton_reuse():
    """Test that _ensure_skeleton copies source file when attest doesn't exist"""
    print("🧪 Test 2: _ensure_skeleton reuses source files")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)
        
        # Setup directory structure
        op_api_dir = project_root / "math" / "sign" / "tests" / "ut" / "op_api"
        op_api_dir.mkdir(parents=True, exist_ok=True)
        
        # Create source file
        source_file = op_api_dir / "test_aclnn_sign.cpp"
        source_content = (
            "/* Original source file */\n"
            "#include \"gtest/gtest.h\"\n"
            "#include \"aclnn/aclnn_sign.h\"\n"
            "\n"
            "class SignTest : public testing::Test {\n"
            "  void SetUp() override {}\n"
            "};\n"
            "\n"
            "TEST_F(SignTest, ExistingCase1) {\n"
            "  EXPECT_EQ(1, 1);\n"
            "}\n"
        )
        source_file.write_text(source_content)
        
        # Mock LLM and tool runner
        llm = LLMClient()
        tool_registry = build_default_registry()
        tool_runner = ToolRunner(tool_registry)
        
        stage = AscendGenerationAgentLoopStage(llm, tool_runner)
        
        # Create file entry for attest file
        file_entry = {
            "file_id": "FILE_SIGN_OP_API",
            "path": "math/sign/tests/ut/op_api/test_aclnn_sign_attest.cpp",
            "kind": "test_file",
            "layer_id": "op_api",
            "op_name": "sign"
        }
        
        # Define new test cases to add
        file_cases = [
            {"block_id": "CASE_NEW_1", "description": "New test case 1"},
            {"block_id": "CASE_NEW_2", "description": "New test case 2"}
        ]
        
        result = stage._ensure_skeleton(project_root, file_entry, file_cases)
        
        attest_path = project_root / file_entry["path"]
        assert attest_path.exists(), f"Should create {attest_path}"
        print(f"  ✓ Created {attest_path.name}")
        
        content = attest_path.read_text()
        
        # Verify it contains markers
        assert "// ==== BLOCK:HEADER START ====" in content, "Should have HEADER START marker"
        assert "// ==== BLOCK:HEADER END ====" in content, "Should have HEADER END marker"
        print("  ✓ Contains HEADER markers")
        
        # Verify source content is preserved
        assert "#include \"gtest/gtest.h\"" in content, "Should preserve include"
        assert "class SignTest" in content, "Should preserve class declaration"
        assert "TEST_F(SignTest, ExistingCase1)" in content, "Should preserve existing TEST_F"
        print("  ✓ Source file content preserved")
        
        # Verify new case placeholders added
        assert "// ==== BLOCK:CASE_NEW_1 ====" in content, "Should add CASE_NEW_1 placeholder"
        assert "// ==== BLOCK:CASE_NEW_2 ====" in content, "Should add CASE_NEW_2 placeholder"
        print("  ✓ New CASE placeholders added")
        
        # Verify FOOTER block present
        assert "// ==== BLOCK:FOOTER START ====" in content, "Should have FOOTER START"
        assert "// ==== BLOCK:FOOTER END ====" in content, "Should have FOOTER END"
        print("  ✓ FOOTER block added")
    
    print("✅ Test 2 passed\n")

def test_ensure_skeleton_no_source():
    """Test that _ensure_skeleton falls back to minimal skeleton when no source"""
    print("🧪 Test 3: _ensure_skeleton fallback when no source file exists")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)
        
        # Setup directory WITHOUT source file
        op_api_dir = project_root / "math" / "empty" / "tests" / "ut" / "op_api"
        op_api_dir.mkdir(parents=True, exist_ok=True)
        
        llm = LLMClient()
        tool_registry = build_default_registry()
        tool_runner = ToolRunner(tool_registry)
        stage = AscendGenerationAgentLoopStage(llm, tool_runner)
        
        file_entry = {
            "file_id": "FILE_EMPTY_OP_API",
            "path": "math/empty/tests/ut/op_api/test_aclnn_empty_attest.cpp",
            "kind": "test_file",
            "layer_id": "op_api",
            "op_name": "empty"
        }
        
        file_cases = [
            {"block_id": "CASE_1", "description": "Test case 1"}
        ]
        
        result = stage._ensure_skeleton(project_root, file_entry, file_cases)
        # Should still succeed (return True or None - need to check actual return)
        
        attest_path = project_root / file_entry["path"]
        assert attest_path.exists(), "Should create attest file even without source"
        
        content = attest_path.read_text()
        
        # Should have BLOCK markers
        assert "// ==== BLOCK:HEADER START ====" in content
        assert "// ==== BLOCK:HEADER END ====" in content
        assert "// ==== BLOCK:CASE_1 ====" in content
        assert "// ==== BLOCK:FOOTER START ====" in content
        assert "// ==== BLOCK:FOOTER END ====" in content
        
        # Plan B: should include C++ boilerplate in HEADER block
        assert "#include" in content, "Plan B should include C++ boilerplate"
        assert "class" in content, "Plan B should include test class definition"
        assert "testing::Test" in content, "Plan B should include gtest class"
        
        print("  ✓ Created skeleton with Plan B C++ boilerplate when no source exists")
    
    print("✅ Test 3 passed\n")

def test_ensure_skeleton_wrap_cmake():
    """Test that _ensure_skeleton adds BLOCK markers (including FOOTER) to existing CMakeLists.txt.
    
    This is required so that _ensure_cmake_attest_registration can insert the
    *_attest.cpp file registration into the FOOTER block.
    """
    print("🧪 Test 4: _ensure_skeleton wraps existing CMakeLists.txt with BLOCK markers")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        project_root = Path(tmpdir)
        
        op_api_dir = project_root / "math" / "sign" / "tests" / "ut" / "op_api"
        op_api_dir.mkdir(parents=True, exist_ok=True)
        
        cmake_path = op_api_dir / "CMakeLists.txt"
        original_content = (
            "# Original CMakeLists content\n"
            "message(STATUS \"Test CMake\")\n"
        )
        cmake_path.write_text(original_content)
        
        llm = LLMClient()
        tool_registry = build_default_registry()
        tool_runner = ToolRunner(tool_registry)
        stage = AscendGenerationAgentLoopStage(llm, tool_runner)
        
        file_entry = {
            "file_id": "FILE_SIGN_CMAKE_OP_API",
            "path": "math/sign/tests/ut/op_api/CMakeLists.txt",
            "kind": "cmake",
            "layer_id": "op_api",
            "comment_style": "#",
        }
        
        result = stage._ensure_skeleton(project_root, file_entry, [])
        
        # Should return True (modified) - BLOCK markers added
        assert result is True, "Should wrap existing CMakeLists.txt with BLOCK markers"
        
        new_content = cmake_path.read_text()
        assert "# ==== BLOCK:HEADER START ====" in new_content, "Should have HEADER START marker"
        assert "# ==== BLOCK:HEADER END ====" in new_content, "Should have HEADER END marker"
        assert "# ==== BLOCK:FOOTER START ====" in new_content, "Should have FOOTER START marker (needed for _attest.cpp registration)"
        assert "# ==== BLOCK:FOOTER END ====" in new_content, "Should have FOOTER END marker"
        assert original_content.rstrip() in new_content, "Should preserve original content inside HEADER block"
        print("  ✓ Wrapped existing CMakeLists.txt with BLOCK markers (HEADER + FOOTER)")
    
    print("✅ Test 4 passed\n")

if __name__ == '__main__':
    print("=" * 60)
    print("Plan A Implementation Tests")
    print("=" * 60 + "\n")
    
    test_find_source_test_file()
    test_ensure_skeleton_reuse()
    test_ensure_skeleton_no_source()
    test_ensure_skeleton_wrap_cmake()
    
    print("=" * 60)
    print("✅ All Plan A tests passed!")
    print("=" * 60)
