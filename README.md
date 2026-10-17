# AI Note Assistant (Learning AI Agents)

A practical project exploring and building a simple AI Agent with Tool/Function Calling capabilities using **LangChain** and **Google Gemini** (`gemini-3.8-flash`).

---

## 📌 Overview

The goal of this project is to demonstrate how a Large Language Model (LLM) can receive natural language requests from users (e.g., *"Remind me to go to the supermarket to buy vegetables this afternoon"*), analyze the intent, and decide to call an appropriate note-taking tool to save tasks into a `todo.txt` file.

### Project Structure

```text
learning_ai_agents_note_assistant/
│
├── brain.py              # The AI "brain" - Configures the Gemini model, binds tools, and processes messages
├── settings.py           # Reads GOOGLE_API_KEY and GEMINI_MODEL from environment variables
├── quota_errors.py       # Detects and formats Gemini quota/billing (402/429 RESOURCE_EXHAUSTED) errors
├── tools.py              # Defines tools (@tool) such as them_ghi_chu to append notes to todo.txt
├── todo.txt              # File storing notes and tasks (auto-created when the tool executes)
├── requirements.txt      # Project runtime dependency specifications
├── requirements-dev.txt  # Development and test dependency specifications (includes pytest)
├── tests/                # Unit tests for settings, quota error detection, and brain entrypoint
│   ├── test_settings.py
│   ├── test_quota_errors.py
│   └── test_brain.py
└── README.md             # Project documentation and guide
```

---

## ⚡ Project Initialization (Quick Start)

To quickly initialize the project and install all required libraries on Windows using PowerShell (`pwsh.exe`), run:

```powershell
py -m pip install langchain-google-genai langchain-core
```

---

## 💻 Detailed Setup Guide for Windows (PowerShell / `pwsh.exe`)

On Windows OS and PowerShell (`pwsh.exe`), it is recommended to use the Python Launcher (`py`) with the `-m` flag to manage and install packages accurately.

### 1. Check Python & pip Environment

Open **PowerShell (`pwsh.exe`)** in the project directory and run:

```powershell
# Check Python version
py --version

# Check pip version
py -m pip --version
```

### 2. Create and Activate a Virtual Environment (Recommended)

To isolate dependencies and avoid package conflicts across projects:

```powershell
# Create a virtual environment named .venv
py -m venv .venv

# Activate the virtual environment in PowerShell
.\.venv\Scripts\Activate.ps1
```

> **Tip**: If you encounter an `ExecutionPolicy` error when running the activation script in PowerShell, grant permission with:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Install Dependencies

Install the packages directly or via `requirements.txt`:

```powershell
# Upgrade pip (optional)
py -m pip install --upgrade pip

# Install project libraries directly
py -m pip install langchain-google-genai langchain-core

# Or install from requirements.txt
py -m pip install -r requirements.txt
```

---

## 🔑 API Key Configuration (Environment Setup)

For security, avoid hardcoding the API key directly in source files (`brain.py`). Instead, configure it as an environment variable in PowerShell:

```powershell
$env:GOOGLE_API_KEY = "your_gemini_api_key_here"
```

Optionally, you can override the Gemini model name via `GEMINI_MODEL` (defaults to `gemini-3.8-flash` when unset or blank):

```powershell
$env:GEMINI_MODEL = "gemini-3.8-flash"
```

If `GOOGLE_API_KEY` is missing or blank, `brain.py` prints a setup hint to `stderr` and exits with code `2`.

---

## 🚀 Running the Project (Execution)

Run `brain.py`:

```powershell
py brain.py
```

### Expected Output:
The model will analyze the prompt and output the structured tool call decision (`tool_calls`):
```python
[{'name': 'them_ghi_chu', 'args': {'noi_dung': 'chiều nay đi siêu thị mua rau'}, 'id': '...'}]
```

---

## 🧪 Running Tests

Install development dependencies and run the test suite from the repository root:

```powershell
py -m pip install -r requirements-dev.txt
py -m pytest -q
```

All tests run offline and do not require `GOOGLE_API_KEY` to be set.

---

## 🩺 Troubleshooting

### `402` / `429` `RESOURCE_EXHAUSTED` (Exit Code `3`)
If `py brain.py` exits with code `3` and reports that Gemini API quota or prepaid credits are exhausted:
- The Google Cloud / AI Studio project behind your `GOOGLE_API_KEY` has depleted its prepaid credits or rate quota.
- **Fix**: Top up credits or manage billing at <https://ai.studio/projects> (see [Gemini API Billing Docs](https://ai.google.dev/gemini-api/docs/billing#prepay)), or set `$env:GOOGLE_API_KEY` to a key from a project with available quota.

---

## 🛠️ Technical Notes & Tips

1. **Importing the Tool in `brain.py`**:
   Make sure to import `them_ghi_chu` from `tools.py` at the top of `brain.py`:
   ```python
   from tools import them_ghi_chu
   ```

2. **Executing Tool Calls**:
   Currently, `brain.py` prints the agent's tool call decision (`phan_hoi.tool_calls`). To automatically invoke the tool and persist the note into `todo.txt`, you can iterate through the tool calls and invoke the function with the model-provided arguments:
   ```python
   for tool_call in phan_hoi.tool_calls:
       if tool_call["name"] == "them_ghi_chu":
           result = them_ghi_chu.invoke(tool_call["args"])
           print(result)
   ```
