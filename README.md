# Gemini Processors Research Agent Showcase

A side-by-side comparison demonstrating the power and elegance of **Gemini Processors** versus traditional Python development for building AI research agents.

## 🎯 What This Showcases

This project implements the **exact same AI research agent** using two completely different approaches:

1. **Traditional Python** (`traditional/`) - Manual control flow, boilerplate code, complex error handling
2. **Gemini Processors** (`processors/`) - Declarative pipelines, automatic error handling, composable architecture

**The functionality is identical - the development experience is dramatically different.**

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for dependency management
- Google Gemini API key

### Setup

```bash
# Clone the repository
git clone https://github.com/mbeacom/gemini-processors-research-agent-showcase.git
cd gemini-processors-research-agent-showcase

# Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"
```

### Run Traditional Implementation

```bash
cd traditional
uv sync --all-groups
uv run python agent.py
```

### Run Gemini Processors Implementation

```bash
cd processors
uv sync --all-groups
uv run python agent.py
```## 📊 The Dramatic Difference

| Aspect | Traditional Python | Gemini Processors |
|--------|-------------------|-------------------|
| **Lines of Code** | ~450+ | ~370+ |
| **Error Handling** | Manual try/catch everywhere | Automatic through metadata |
| **Composability** | Monolithic functions | Reusable `@processor_function` |
| **Pipeline Definition** | Nested function calls | Declarative `+` composition |
| **Testability** | Complex mocking required | Easy processor unit testing |

## 🔧 What Both Agents Do

1. **Load Tasks**: Read research topics from `sources.json`
2. **Validate**: Use Pydantic models to ensure data quality
3. **Fetch Content**: Securely retrieve content from URLs
4. **Generate Summaries**: Use Google Gemini AI for intelligent summaries
5. **Create Report**: Generate a professional Markdown newsletter

## 💡 Key Insights

### Traditional Approach Pain Points
- ❌ **Boilerplate Everywhere**: Repetitive error handling, logging, state management
- ❌ **Monolithic Functions**: Large functions mixing multiple concerns
- ❌ **Manual State Passing**: Complex parameter threading between steps
- ❌ **Error Prone**: Easy to miss edge cases and error conditions

### Gemini Processors Advantages
- ✅ **Declarative Pipelines**: `pipeline = processor1 + processor2 + processor3`
- ✅ **Automatic Error Handling**: Metadata flows errors through the pipeline
- ✅ **Single Responsibility**: Each processor has one clear purpose
- ✅ **Composable & Reusable**: Mix and match processors for different workflows
- ✅ **Built-in Concurrency**: Automatic parallel processing where possible

## 🔍 Code Comparison

### Traditional: Manual Everything
```python
async def process_tasks(self):
    """Traditional approach - manual control flow"""
    try:
        # Manual loading with explicit error handling
        raw_tasks = await self._load_tasks()

        # Manual validation with state tracking
        validated_tasks = []
        for task in raw_tasks:
            try:
                validated = self._validate_task(task)
                validated_tasks.append(validated)
            except Exception as e:
                self._handle_validation_error(task, e)

        # Manual fetching with progress tracking
        fetched_tasks = []
        for i, task in enumerate(validated_tasks):
            try:
                content = await self._fetch_content(task)
                fetched_tasks.append(content)
                self._log_progress(i, len(validated_tasks))
            except Exception as e:
                self._handle_fetch_error(task, e)

        # Manual summary generation...
        # More boilerplate code continues...
    except Exception as e:
        self._handle_pipeline_error(e)
```

### Gemini Processors: Declarative Elegance

```python
# Gemini Processors approach - compose with + operator
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

# Execute the entire pipeline
async for result in research_pipeline(input_stream):
    if result.metadata.get("status") == "completed":
        print("Pipeline completed successfully!")
```

## 📁 Project Structure

```bash
gemini-processors-research-agent-showcase/
├── assets               # Project assets (input data)
│   └── sources.json     # Test data (valid & invalid entries)
├── LICENSE              # The project license file
├── processors           # Gemini Processors implementation
│   ├── agent.py         # Processors-based agent
│   ├── pyproject.toml   # Dependencies & tooling
│   ├── README.md        # Processors approach docs
│   └── uv.lock          # Dependencies & tooling
├── README.md            # This file
├── SHOWCASE.md          # Detailed comparison analysis
└── traditional          # Manual Python implementation
    ├── agent.py         # Traditional research agent
    ├── pyproject.toml   # Dependencies & tooling
    ├── README.md        # Traditional approach docs
    └── uv.lock          # Dependencies & tooling
```

## 🎓 Learning Outcomes

After exploring this showcase, you'll understand:

- **Why declarative pipelines** are superior to imperative control flow
- **How automatic error handling** eliminates boilerplate code
- **The power of composable processors** for building complex workflows
- **When to choose** Gemini Processors vs traditional approaches
- **Migration strategies** for existing Python codebases

## 🔗 Learn More

- **[Blog Post: Gemini Processors Announcement](https://www.markbeacom.com/blog/architecture/genai/processors-announcement)** - Comprehensive overview and announcement
- **[Detailed Comparison](SHOWCASE.md)** - In-depth analysis of both approaches
- **[Gemini Processors Documentation](https://github.com/google-gemini/genai-processors)** - Official framework docs
- **[Design Document](DESIGN_DOCUMENT.md)** - Technical architecture decisions

## 🤝 Contributing

Found an issue or want to improve the showcase? Contributions welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

**Ready to see the future of AI application development?**

Start with the [Traditional Implementation](traditional/) to experience the pain points, then discover the elegance of [Gemini Processors](processors/) - you'll never want to go back to manual control flow! 🚀
