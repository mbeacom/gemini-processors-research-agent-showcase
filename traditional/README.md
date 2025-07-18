# Traditional Research Agent

This directory contains the traditional implementation of the AI research agent, built using manual control flow, explicit error handling, and the Google Gemini Python SDK (`google-genai`).

## 🎯 Purpose

This implementation serves as the "before" picture to demonstrate the complexity and boilerplate code required when building AI workflows without a proper framework like Gemini Processors.

## 🚀 Quick Start

```bash
# Ensure you're in the traditional directory
cd traditional

# Set your API key
export GEMINI_API_KEY="your-gemini-api-key"

# Install dependencies and run
uv sync --all-groups
uv run python agent.py
```

## 📁 Files

- **`agent.py`** - The traditional implementation (~800+ lines)
- **`pyproject.toml`** - Dependencies and project configuration
- **`newsletter.md`** - Generated output report

*Note: Input data is located in `../assets/sources.json` (shared with processors implementation)*

## ⚙️ Key Characteristics

- **Manual Control Flow**: Uses explicit for loops, if/else statements, and try/catch blocks
- **Explicit Error Handling**: Every step requires manual error checking and state management
- **Boilerplate Code**: Lots of repetitive code for validation, fetching, and processing
- **Complex State Management**: Manual tracking of success/failure states throughout the pipeline
- **Direct API Usage**: Uses the `google-genai` SDK directly with manual client management

## 🔧 Installation

1. Make sure you have Python 3.13+ installed
2. Install dependencies using uv:

   ```bash
   cd traditional
   uv sync --all-groups
   ```

## ⚙️ Configuration

Set your Gemini API key:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

## 🏃‍♂️ Usage

Run the traditional agent:

```bash
uv run python agent.py
```

This will:

1. Read tasks from `../assets/sources.json`
2. Validate each task manually
3. Fetch content from valid URLs
4. Generate summaries using Gemini AI
5. Create a `newsletter.md` report

## 🧪 Testing

Test that imports work correctly:

```bash
uv run python -c "import agent; print('✅ Imports working')"
```

## 📊 Expected Output

The agent processes `../assets/sources.json` and generates `newsletter.md` with the same output as the processors implementation, but using traditional manual control flow.

## 📂 Input Format

The agent expects a `../assets/sources.json` file with the following format:

```json
[
  {
    "topic": "Topic description",
    "source_url": "https://example.com/article"
  }
]
```

## 📄 Output

The agent generates a `newsletter.md` file containing summaries of all successfully processed articles.

## 🔍 Compare with Processors

See the [Gemini Processors implementation](../processors/) to understand the dramatic difference in complexity and development experience.

## 🛠️ Dependencies

- **google-genai** - Google Gemini API client
- **pydantic** - Data validation
- **httpx** - HTTP client for URL fetching

## 🏗️ Implementation Highlights

This traditional implementation showcases:

- **Manual Validation Loop**: Explicit iteration through tasks with try/catch for each validation
- **Manual HTTP Handling**: Custom security checks, timeout handling, and status code processing
- **Manual State Tracking**: Explicit counters for success/failure states
- **Manual Error Recovery**: Individual try/catch blocks for each operation
- **Manual Resource Management**: Explicit HTTP client lifecycle management
- **Manual Template Generation**: String manipulation for report formatting
- **Direct Gemini API Usage**: Using `client.models.generate_content()` with manual error handling

## 💡 API Usage Examples

The implementation demonstrates traditional patterns like:

```python
# Manual client creation
from google import genai
client = genai.Client(api_key=api_key)

# Manual API call with error handling
try:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    if response and response.text:
        summary = response.text.strip()
        return summary
except Exception as e:
    print(f"Error: {e}")
    return None
```

Compare this complexity with the Gemini Processors implementation to see the dramatic difference in code clarity and maintainability.
