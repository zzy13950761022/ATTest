# ATTest Usage Guide

## 🎉 Installation Complete

ATTest has been installed successfully in your environment. You can now use the `attest` command directly.

## 📖 Quick Start

### 1. Show Help

```bash
attest --help
```

### 2. View Configuration

```bash
attest config list
```

The default configuration is:
- **Model**: `deepseek-chat`
- **API Base**: `https://api.deepseek.com/v1`
- **API Key**: configured
- **Temperature**: `0.2`
- **Max Tokens**: `4096`

### 3. Start Chat Mode

```bash
# Start with the welcome screen, then enter chat mode
attest

# Start in the current directory
attest chat

# Specify a workspace
attest chat --workspace /path/to/your/project

# Automatically approve tool calls
attest chat --auto-approve

# Equivalent short form
attest --workspace /path/to/your/project --auto-approve
```

## 💬 Chat Mode Example

After launch, you can interact with the agent like this:

```text
[Chat Mode] Session: a1b2c3d4
[Chat Mode] Workspace: /Users/mac/Desktop/test
[Chat Mode] Type 'exit' or Ctrl+C to quit

You: List the files in the current directory
[Tool] Calling list_files with {'path': '/Users/mac/Desktop/test'}
[Tool] ✓ file1.txt
file2.py
README.md
Assistant: The current directory contains 3 files: file1.txt, file2.py, and README.md

You: Read the contents of README.md
[Tool] Calling read_file with {'path': 'README.md'}
[Tool] ✓ # My Project
This is a test project...
Assistant: README.md is a project description file...

You: exit
[Chat Mode] Goodbye!
```

## 🔧 Available Tools

The agent can call the following tools automatically:

| Tool | Description | Approval Required |
|------|------|----------|
| `list_files` | List directory contents | ❌ |
| `read_file` | Read a full file | ❌ |
| `part_read` | Read a specific line range from a file | ❌ |
| `search` | Search file contents with `ripgrep` | ❌ |
| `inspect_python` | Inspect Python objects (signature, docs, source) | ❌ |
| `write_file` | Write a file with automatic backup | ✅ |
| `replace_in_file` | Replace text in a file | ✅ |
| `exec_command` | Run a shell command | ✅ |

> Tools marked as requiring approval will ask for confirmation before execution, unless you use `--auto-approve`.

## 📋 Configuration Management

```bash
# Set configuration values
attest config set api.model deepseek-chat
attest config set api.temperature 0.5

# Get a configuration value
attest config get api.model

# List all configuration values
attest config list
```

## 📂 Session Management

```bash
# List all sessions
attest sessions list

# Clear a specific session
attest sessions clear <session_id>
```

Session history is stored in `~/.attest_cli/sessions/`.

## 🚀 Advanced Usage

### Common Workflows

1. **Code Review**

```bash
attest chat --workspace ~/my-project
> Review the code quality of src/main.py
```

2. **Search and Edit Files**

```bash
attest chat
> Search for all TODO comments in the current project
> Add a new configuration entry to config.py
```

3. **Batch Processing**

```bash
attest chat --auto-approve
> Rename all .txt files to .md
```

## 🛠️ Troubleshooting

### Issue: API Call Fails

Check whether the API key is configured correctly:

```bash
attest config get api.api_key
```

### Issue: Command Not Found

Make sure the package is installed:

```bash
pip install -e /Users/mac/Desktop/cliagent/ATTest
```

### Issue: Tool Execution Fails

Some tools, such as `search`, depend on external commands like `ripgrep`. Make sure they are installed:

```bash
brew install ripgrep  # macOS
```

## 📝 Tips

1. **Be specific**: “Refactor this function” is more effective than “Optimize the code.”
2. **Use the tools**: The agent can read files directly, so you usually do not need to paste code manually.
3. **Iterate**: Complex tasks are often best solved across multiple turns.
4. **Use `--auto-approve` carefully**: It can speed up trusted operations.

## 🎯 Next Steps

Now that you know the basics, you can:
- Try chat mode in a real project
- Explore more complex tool combinations
- Tune configuration values for your workflow

Enjoy using ATTest.
