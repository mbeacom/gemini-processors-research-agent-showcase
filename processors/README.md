# Gemini Processors Implementation

This directory contains the **Gemini Processors** implementation of the research agent, demonstrating the power of declarative pipeline composition.

## 🚀 Quick Start

```bash
# Ensure you're in the processors directory
cd processors

# Set your API key
export GEMINI_API_KEY="your-gemini-api-key"

# Install dependencies and run
uv sync --all-groups
uv run python agent.py
```

## 📁 Files

- **`agent.py`** - The main Gemini Processors implementation (~400 lines)
- **`pyproject.toml`** - Dependencies and project configuration
- **`newsletter.md`** - Generated output report

*Note: Input data is located in `../assets/sources.json` (shared with traditional implementation)*

## 🔧 Key Features Demonstrated

### Declarative Pipeline Composition

```python
research_pipeline = (
    load_tasks_from_json
    + validate_task
    + fetch_url_content
    + filter_successful_fetches
    + add_summary_prompt
    + genai_model.GenaiModel(
        api_key=api_key,
        model_name="gemini-2.5-flash"
    )
    + collect_summaries_for_report
)
```

### Processor Functions

Each step is a composable `@processor_function`:

- **`load_tasks_from_json`** - Loads and parses JSON input
- **`validate_task`** - Pydantic validation with metadata
- **`fetch_url_content`** - Secure URL fetching
- **`filter_successful_fetches`** - Automatic error filtering
- **`add_summary_prompt`** - Prompt preparation
- **`collect_summaries_for_report`** - Output aggregation

### Automatic Error Handling

Errors flow through metadata automatically:

```python
yield content_api.ProcessorPart(
    value=content,
    metadata={
        "validation_status": "success",
        "url_fetch_status": "success"
    }
)
```

## 🎯 Advantages Over Traditional Approach

1. **No Manual Error Handling** - Errors flow through metadata
2. **Composable Architecture** - Mix and match processors
3. **Streaming Data Flow** - Automatic parallel processing
4. **Clean Separation** - Each processor has single responsibility
5. **Testable Components** - Easy to unit test individual processors

## 📊 Expected Output

The agent processes `../assets/sources.json` and generates `newsletter.md` with:

- ✅ 2 successful summaries (from valid URLs)
- ❌ 3 failed tasks (missing URL, invalid URL, localhost URL)
- 📝 Professional Markdown newsletter format

## 🔍 Compare with Traditional

See the [traditional implementation](../traditional/) to understand the dramatic difference in complexity and development experience.

## 🛠️ Dependencies

- **genai-processors** - Core framework
- **genai_processors.core** - GenAI model integration
- **google-genai** - Google Gemini API client
- **pydantic** - Data validation
- **httpx** - Async HTTP client

## 🎓 Learning Points

This implementation showcases:

- How declarative pipelines eliminate boilerplate
- The power of metadata-driven error handling
- Composable processor architecture patterns
- Streaming async data processing
- Clean separation of concerns in AI workflows
