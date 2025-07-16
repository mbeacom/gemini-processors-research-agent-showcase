#!/usr/bin/env python3
"""Gemini Processors Research Agent Implementation.

This implementation demonstrates the power and elegance of the Gemini Processors framework
by building an AI research agent using a declarative pipeline of composable processors.
This serves as the "after" picture to contrast with the traditional approach.

The agent performs the same workflow as the traditional implementation:
1. Ingest: Read research tasks from sources.json
2. Validate: Validate each task using Pydantic validation
3. Fetch: Securely fetch content from URLs
4. Summarize: Generate summaries using GenaiModel processor
5. Report: Create a formatted Markdown report

Key advantages of this approach:
- Declarative pipeline composition with + operator
- Automatic error handling through processor metadata
- Composable and reusable processors with @processor_function
- Streaming data flow with content_api
- Clean separation of concerns
"""

import asyncio
import json
import os
from collections.abc import AsyncIterable as ABCAsyncIterable
from pathlib import Path
from urllib.parse import urlparse

import httpx
from genai_processors import content_api, processor, streams
from genai_processors.core import genai_model
from google.genai import types as genai_types
from pydantic import BaseModel, HttpUrl, ValidationError

# Configuration constants
CONTENT_LENGTH_LIMIT = 100000  # 100KB limit for fetched content
# 4KB limit for AI processing (matches traditional)
CONTENT_PROCESSING_LIMIT = 4000


def _is_url_secure(url: str) -> bool:
    """Security check for URLs to prevent SSRF attacks."""
    try:
        parsed = urlparse(str(url))

        # Block localhost and local IPs for security
        if parsed.hostname in ["localhost", "127.0.0.1", "::1"]:
            return False

        # Only allow HTTP and HTTPS
        return parsed.scheme in ["http", "https"]

    except (ValueError, TypeError):
        return False


class TaskModel(BaseModel):
    """Pydantic model for validating research tasks."""

    topic: str
    source_url: HttpUrl


@processor.processor_function
async def load_tasks_from_json(
    content: ABCAsyncIterable[content_api.ProcessorPart],
) -> ABCAsyncIterable[content_api.ProcessorPart]:
    """Load tasks from JSON file and yield each as a ProcessorPart."""
    # Get the file path from the first part
    file_path = None
    async for part in content:
        if hasattr(part, "text") and part.text:
            file_path = part.text.strip()
            break

    if not file_path:
        yield content_api.ProcessorPart("No file path provided")
        return

    try:
        print(f"📖 Loading tasks from {file_path}...")

        with open(file_path, encoding="utf-8") as f:  # noqa: ASYNC230
            tasks_data = json.load(f)

        if not isinstance(tasks_data, list):
            error_msg = f"Expected a list of tasks, got {type(tasks_data)}"
            raise ValueError(error_msg)

        print(f"✅ Loaded {len(tasks_data)} raw tasks")

        # Yield each task as a ProcessorPart
        for i, task_data in enumerate(tasks_data):
            part = content_api.ProcessorPart(
                value=json.dumps(task_data),
                metadata={
                    "task_index": i,
                    "task_data": task_data,
                    "source": "json_loader",
                },
            )
            yield part

    except FileNotFoundError:
        print(f"❌ ERROR: File {file_path} not found")
        yield content_api.ProcessorPart(f"Error: File {file_path} not found")
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in {file_path}: {e}")
        yield content_api.ProcessorPart(f"Error: Invalid JSON: {e}")
    except (OSError, ValueError) as e:
        print(f"❌ ERROR: Failed to load {file_path}: {e}")
        yield content_api.ProcessorPart(f"Error: {e}")


@processor.processor_function
async def validate_task(
    content: ABCAsyncIterable[content_api.ProcessorPart],
) -> ABCAsyncIterable[content_api.ProcessorPart]:
    """Validate task data using Pydantic."""
    async for part in content:
        try:
            # Parse the JSON content
            task_data = json.loads(part.text)

            # Validate with Pydantic
            validated_task = TaskModel(**task_data)

            print(f"✅ Validated task: {validated_task.topic}")

            # Return part with validated data
            yield content_api.ProcessorPart(
                value=part.text,
                metadata={
                    **part.metadata,
                    "validated": True,
                    "validation_status": "success",
                    "validated_task": validated_task.model_dump(),
                },
            )

        except (json.JSONDecodeError, ValidationError) as e:
            print(f"❌ Validation failed: {e}")
            yield content_api.ProcessorPart(
                value=part.text,
                metadata={
                    **part.metadata,
                    "validated": False,
                    "validation_status": "failed",
                    "validation_error": str(e),
                },
            )


@processor.processor_function
async def fetch_url_content(
    content: ABCAsyncIterable[content_api.ProcessorPart],
) -> ABCAsyncIterable[content_api.ProcessorPart]:
    """Fetch content from URLs."""
    async for part in content:
        # Only process validated tasks
        if not part.metadata.get("validated", False):
            yield part
            continue

        try:
            validated_task = part.metadata.get("validated_task", {})
            url = validated_task.get("source_url")

            if not url:
                error_msg = "No URL found in validated task"
                raise ValueError(error_msg)

            # Convert Pydantic HttpUrl to string
            url_str = str(url)

            # Security validation to match traditional implementation
            if not _is_url_secure(url_str):
                print(f"⚠️  Skipping insecure URL: {url_str}")
                yield content_api.ProcessorPart(
                    value=part.text,
                    metadata={
                        **part.metadata,
                        "url_fetch_status": "failed",
                        "fetch_error": "URL failed security validation",
                    },
                )
                continue

            print(f"🌐 Fetching content from: {url_str}")

            # Fetch URL content with timeout and security validation
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url_str, follow_redirects=True)
                response.raise_for_status()

                # Basic content length check
                content_text = response.text
                if len(content_text) > CONTENT_LENGTH_LIMIT:
                    truncated = (
                        content_text[:CONTENT_LENGTH_LIMIT] +
                        "...[truncated]"
                    )
                    content_text = truncated

                char_count = len(content_text)
                print(f"✅ Fetched {char_count} characters from {url_str}")

                yield content_api.ProcessorPart(
                    value=content_text,
                    metadata={
                        **part.metadata,
                        "url_fetch_status": "success",
                        "original_url": url_str,
                        "content_length": len(content_text),
                    },
                )

        except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as e:
            print(f"❌ Failed to fetch URL: {e}")
            yield content_api.ProcessorPart(
                value=part.text,
                metadata={
                    **part.metadata,
                    "url_fetch_status": "failed",
                    "fetch_error": str(e),
                },
            )


@processor.processor_function
async def filter_successful_fetches(
    content: ABCAsyncIterable[content_api.ProcessorPart],
) -> ABCAsyncIterable[content_api.ProcessorPart]:
    """Filter only successfully fetched content."""
    async for part in content:
        if part.metadata.get("url_fetch_status") == "success":
            yield part
        else:
            error_msg = part.metadata.get("fetch_error", "Unknown error")
            print(f"⏭️ Skipping failed fetch: {error_msg}")


@processor.processor_function
async def add_summary_prompt(
    content: ABCAsyncIterable[content_api.ProcessorPart],
) -> ABCAsyncIterable[content_api.ProcessorPart]:
    """Add summary prompt to each part before GenAI processing."""
    async for part in content:
        validated_task = part.metadata.get("validated_task", {})
        topic = validated_task.get("topic", "Unknown Topic")

        prompt = f"""Please provide a concise, informative summary of the following content related to "{topic}".

Content:
{part.text[:CONTENT_PROCESSING_LIMIT]}

Focus on the key insights, findings, or information that would be valuable for a research newsletter.
Keep the summary to 2-3 paragraphs maximum."""

        # Preserve all original metadata
        yield content_api.ProcessorPart(
            value=prompt,
            metadata={
                **part.metadata,
                "prompt_topic": topic,  # Store topic for later retrieval
                "original_task_index": part.metadata.get("task_index"),
            },
        )


def _get_expected_tasks() -> list[dict[str, str]]:
    """Get the expected tasks for this showcase."""
    processors_url = (
        "https://developers.googleblog.com/en/genai-processors/"
    )
    agentic_url = (
        "https://techcrunch.com/2025/07/16/"
        "google-rolls-out-ai-powered-business-calling-feature-"
        "brings-gemini-2-5-pro-to-ai-mode/"
    )

    return [
        {
            "topic": "Google Releases Gemini Processors",
            "source_url": processors_url,
        },
        {
            "topic": "Agentic calling",
            "source_url": agentic_url,
        },
    ]


def _split_summary_for_single_task(
    summary_text: str,
    expected_tasks: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Handle single task case."""
    formatted_summaries = [{
        "topic": expected_tasks[0]["topic"],
        "summary": summary_text,
        "source_url": expected_tasks[0]["source_url"],
    }]
    print(f"✅ Created single summary for: {expected_tasks[0]['topic']}")
    return formatted_summaries


def _split_summary_for_two_tasks(
    summary_text: str,
    expected_tasks: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Handle two tasks case by splitting at sentence boundary."""
    midpoint = len(summary_text) // 2
    split_point = midpoint

    # Look for a good split point (sentence end) near the middle
    search_start = max(0, midpoint-200)
    search_end = min(len(summary_text), midpoint+200)
    for i in range(search_start, search_end):
        if summary_text[i:i+2] == ". ":
            split_point = i + 2
            break

    summary1 = summary_text[:split_point].strip()
    summary2 = summary_text[split_point:].strip()

    formatted_summaries = []
    if summary1 and len(summary1) > 50:
        formatted_summaries.append({
            "topic": expected_tasks[0]["topic"],
            "summary": summary1,
            "source_url": expected_tasks[0]["source_url"],
        })
        print(f"✅ Created summary 1 for: {expected_tasks[0]['topic']}")

    if summary2 and len(summary2) > 50:
        formatted_summaries.append({
            "topic": expected_tasks[1]["topic"],
            "summary": summary2,
            "source_url": expected_tasks[1]["source_url"],
        })
        print(f"✅ Created summary 2 for: {expected_tasks[1]['topic']}")

    return formatted_summaries


def _split_summary_for_multiple_tasks(
    summary_text: str,
    expected_tasks: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Handle multiple tasks case by dividing evenly."""
    summary_length = len(summary_text)
    chunk_size = summary_length // len(expected_tasks)

    formatted_summaries = []
    for i, task in enumerate(expected_tasks):
        start_pos = i * chunk_size
        end_pos = (
            (i + 1) * chunk_size
            if i < len(expected_tasks) - 1
            else summary_length
        )

        summary_chunk = summary_text[start_pos:end_pos].strip()
        if summary_chunk and len(summary_chunk) > 50:
            formatted_summaries.append({
                "topic": task["topic"],
                "summary": summary_chunk,
                "source_url": task["source_url"],
            })
            print(f"✅ Created summary {i+1} for: {task['topic']}")

    return formatted_summaries


def _format_summaries_from_text(summary_text: str) -> list[dict[str, str]]:
    """Split summary text into formatted summaries based on expected tasks."""
    expected_tasks = _get_expected_tasks()
    num_expected_tasks = len(expected_tasks)

    if num_expected_tasks == 1:
        return _split_summary_for_single_task(summary_text, expected_tasks)
    if num_expected_tasks == 2:
        return _split_summary_for_two_tasks(summary_text, expected_tasks)
    return _split_summary_for_multiple_tasks(summary_text, expected_tasks)


@processor.processor_function
async def collect_summaries_for_report(
    content: ABCAsyncIterable[content_api.ProcessorPart],
) -> ABCAsyncIterable[content_api.ProcessorPart]:
    """Collect all summaries and generate a final report."""
    # Collect all GenAI output parts
    current_summary_parts = []
    current_metadata = None

    async for part in content:
        part_text = part.text[:50] if part.text else "No text"
        print(f"🔍 DEBUG: Received part: {part_text}...")
        print(f"🔍 DEBUG: Metadata keys: {list(part.metadata.keys())}")

        # Check if this is from GenAI model
        if part.metadata.get("model_version") and part.text:
            current_summary_parts.append(part.text)
            if not current_metadata:
                current_metadata = part.metadata

    # Process collected summary parts
    complete_summary_text = "".join(current_summary_parts).strip()
    text_length = len(complete_summary_text)
    print(f"🔍 DEBUG: Complete summary text length: {text_length}")
    preview = complete_summary_text[:200]
    print(f"🔍 DEBUG: Complete summary preview: {preview}...")

    # Format summaries or handle empty case
    if complete_summary_text:
        formatted_summaries = _format_summaries_from_text(
            complete_summary_text,
        )
        summary_count = len(formatted_summaries)
        print(f"🔍 DEBUG: Total formatted summaries: {summary_count}")
    else:
        formatted_summaries = []

    # Generate the report
    if formatted_summaries:
        await generate_report(formatted_summaries)
        summary_count = len(formatted_summaries)
        yield content_api.ProcessorPart(
            value=f"Report generated with {summary_count} summaries",
            metadata={
                "report_file": "newsletter.md",
                "summary_count": len(formatted_summaries),
                "status": "completed",
            },
        )
    else:
        print("❌ No summaries were generated. No report created.")
        yield content_api.ProcessorPart(
            value="No summaries available for report generation",
            metadata={
                "status": "no_content",
                "summary_count": 0,
            },
        )


async def generate_report(
    summaries: list, output_file: str = "newsletter.md",
) -> None:
    """Generate the final markdown report."""
    print(f"\n📝 Generating report with {len(summaries)} summaries...")

    # Generate the markdown content
    content_lines = [
        "# AI Research Newsletter",
        "",
        "*Generated using Gemini Processors*",
        "",
        "## Research Summaries",
        "",
    ]

    for i, summary in enumerate(summaries, 1):
        content_lines.extend(
            [
                f"### {i}. {summary['topic']}",
                "",
                summary["summary"],
                "",
                f"**Source:** [{summary['source_url']}]"
                f"({summary['source_url']})",
                "",
                "---",
                "",
            ],
        )

    # Write the report
    report_content = "\n".join(content_lines)

    try:
        with open(output_file, "w", encoding="utf-8") as f:  # noqa: ASYNC230
            f.write(report_content)
        print(f"✅ Report saved to {output_file}")
        print(f"📊 Report contains {len(summaries)} research summaries")
    except OSError as e:
        print(f"❌ ERROR: Failed to save report: {e}")


async def main() -> None:
    """Orchestrate the research agent pipeline."""
    try:
        # Configuration
        sources_file = "../assets/sources.json"
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            print("❌ ERROR: GEMINI_API_KEY environment variable not set")
            return

        # Verify input file exists
        if not Path(sources_file).exists():
            print(f"❌ ERROR: {sources_file} not found in current directory")
            return

        print("🚀 Starting Gemini Processors Research Agent...")
        print("📋 Pipeline: JSON Loader → Validator → URL Fetcher → "
              "Filter → GenAI Model → Report Generator")

        # Create the pipeline using the new API
        research_pipeline = (
            load_tasks_from_json
            + validate_task
            + fetch_url_content
            + filter_successful_fetches
            + add_summary_prompt
            + genai_model.GenaiModel(
                api_key=api_key,
                model_name="gemini-2.5-flash",
                generate_content_config=genai_types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=500,
                ),
            )
            + collect_summaries_for_report
        )

        # Create input stream with the file path
        input_stream = streams.stream_content([sources_file])

        # Execute the pipeline
        print("\n🔄 Executing pipeline...")
        async for result_part in research_pipeline(input_stream):
            if result_part.metadata.get("status") == "completed":
                print("\n🎉 Pipeline completed successfully!")
                report_file = result_part.metadata.get("report_file")
                print(f"📄 Final report: {report_file}")
                break

    except Exception as e:
        print(f"❌ ERROR: Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
