#!/usr/bin/env python3
"""Traditional Research Agent Implementation.

This implementation demonstrates a traditional approach to building an AI research agent
using manual control flow, explicit error handling, and standard Python libraries.
This serves as the "before" picture to contrast with the Gemini Processors approach.

The agent performs the following workflow:
1. Ingest: Read research tasks from sources.json
2. Validate: Validate each task using Pydantic
3. Fetch: Securely fetch content from URLs
4. Summarize: Generate summaries using Gemini AI
5. Report: Create a formatted Markdown report
"""

import contextlib
import json
import os
import sys
from typing import Any
from urllib.parse import urlparse

import httpx
from google import genai
from pydantic import BaseModel, HttpUrl, ValidationError


class TaskModel(BaseModel):
    """Pydantic model for validating research tasks."""

    topic: str
    source_url: HttpUrl


class FetchedContent(BaseModel):
    """Model for content that was successfully fetched."""

    task: TaskModel
    content: str
    url: str


class Summary(BaseModel):
    """Model for generated summaries."""

    topic: str
    summary: str
    source_url: str


class TraditionalResearchAgent:
    """A traditional implementation of a research agent using manual control flow.

    This implementation deliberately showcases the boilerplate and complexity
    that arises when building AI workflows without a proper framework.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the agent with Gemini API configuration."""
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            error_msg = "GEMINI_API_KEY environment variable must be set or api_key must be provided"
            raise ValueError(error_msg)

        # Configure the Gemini client using the new SDK
        self.client = genai.Client(api_key=api_key)

        # HTTP client for fetching URLs
        self.http_client = httpx.Client(
            timeout=30.0,
            headers={
                "User-Agent": "Traditional-Research-Agent/1.0",
            },
        )

    def load_tasks_from_json(self, file_path: str) -> list[dict[str, Any]]:
        """Step 1: Ingest - Load tasks from JSON file.

        This is the first manual step where we need explicit error handling
        for file operations.
        """
        print(f"📖 Loading tasks from {file_path}...")

        try:
            with open(file_path, encoding="utf-8") as f:
                tasks_data = json.load(f)
        except FileNotFoundError:
            print(f"❌ ERROR: File {file_path} not found")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Invalid JSON in {file_path}: {e}")
            return []
        except (OSError, ValueError) as e:
            print(f"❌ ERROR: Failed to load {file_path}: {e}")
            return []

        if not isinstance(tasks_data, list):
            print(f"❌ ERROR: Expected a list of tasks, got {type(tasks_data)}")
            return []

        print(f"✅ Loaded {len(tasks_data)} raw tasks")
        return tasks_data

    def validate_tasks(self, tasks_data: list[dict[str, Any]]) -> list[TaskModel]:
        """Step 2: Validate - Manually validate each task using Pydantic.

        This demonstrates the manual iteration and error handling required
        for validation without a framework.
        """
        print("🔍 Validating tasks...")

        valid_tasks = []
        invalid_count = 0

        for i, task_data in enumerate(tasks_data):
            try:
                # Manual validation using Pydantic
                validated_task = TaskModel(**task_data)
                valid_tasks.append(validated_task)
                print(f"✅ Task {i+1}: '{validated_task.topic}' - Valid")

            except ValidationError as e:
                invalid_count += 1
                print(f"❌ Task {i+1}: Validation failed - {e}")
                # In a traditional approach, we manually log and skip
                continue
            except (TypeError, ValueError) as e:
                invalid_count += 1
                print(f"❌ Task {i+1}: Unexpected error during validation - {e}")
                continue

        print(f"✅ Validation complete: {len(valid_tasks)} valid, {invalid_count} invalid")
        return valid_tasks

    def is_url_secure(self, url: str) -> bool:
        """Manual security check for URLs.

        In a traditional approach, we need to manually implement
        security checks and edge case handling.
        """
        try:
            parsed = urlparse(str(url))

            # Block localhost and local IPs for security
            if parsed.hostname in ["localhost", "127.0.0.1", "::1"]:
                return False

            # Only allow HTTP and HTTPS
            return parsed.scheme in ["http", "https"]

        except (ValueError, TypeError):
            return False

    def fetch_url_content(self, url: str) -> str | None:
        """Manual URL fetching with security checks and error handling.

        This demonstrates the complexity of manual HTTP handling
        and the various edge cases that need to be considered.
        """
        try:
            # Manual security validation
            if not self.is_url_secure(url):
                print(f"⚠️  Skipping insecure URL: {url}")
                return None

            print(f"🌐 Fetching content from: {url}")

            # Manual HTTP request with error handling
            response = self.http_client.get(url)

            # Manual status code checking
            if response.status_code == 200:
                content = response.text
                print(f"✅ Successfully fetched {len(content)} characters")
                return content
            print(f"❌ HTTP {response.status_code} error for {url}")
            return None

        except httpx.TimeoutException:
            print(f"❌ Timeout error fetching {url}")
            return None
        except httpx.RequestError as e:
            print(f"❌ Request error fetching {url}: {e}")
            return None
        except (httpx.HTTPStatusError, ValueError) as e:
            print(f"❌ Unexpected error fetching {url}: {e}")
            return None

    def fetch_content_for_tasks(self, valid_tasks: list[TaskModel]) -> list[FetchedContent]:
        """Step 3: Fetch - Manually iterate through tasks and fetch content.

        This shows the manual iteration and state management required
        when working without a streaming framework.
        """
        print("📥 Fetching content for valid tasks...")

        fetched_content = []
        failed_count = 0

        for i, task in enumerate(valid_tasks):
            print(f"\n--- Fetching {i+1}/{len(valid_tasks)}: {task.topic} ---")

            content = self.fetch_url_content(str(task.source_url))

            if content is not None:
                # Manual object creation and state management
                fetched = FetchedContent(
                    task=task,
                    content=content,
                    url=str(task.source_url),
                )
                fetched_content.append(fetched)
                print(f"✅ Content fetched for: {task.topic}")
            else:
                failed_count += 1
                print(f"❌ Failed to fetch content for: {task.topic}")
                # Manual error counting and logging
                continue

        print(f"\n✅ Fetch complete: {len(fetched_content)} successful, {failed_count} failed")
        return fetched_content

    def generate_summary(self, topic: str, content: str, source_url: str) -> str | None:
        """Manual summary generation using Gemini AI.

        This demonstrates the manual API interaction and error handling
        required when working directly with the Gemini client.
        """
        try:
            # Manual prompt construction
            prompt = f"""
Please create a concise summary of the following content about "{topic}".

The summary should be:
- 2-3 paragraphs maximum
- Focus on key insights and main points
- Written in a professional, newsletter-style tone
- Suitable for a research newsletter

Source URL: {source_url}

Content to summarize:
{content[:5000]}"""

            print(f"🤖 Generating summary for: {topic}")

            # Manual API call with error handling using new SDK
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )

            if response and response.text:
                summary = response.text.strip()
                print(f"✅ Summary generated ({len(summary)} characters)")
                return summary
            print(f"❌ Empty response from Gemini for: {topic}")
            return None

        except (ValueError, AttributeError) as e:
            print(f"❌ Error generating summary for {topic}: {e}")
            return None

    def generate_summaries(self, fetched_content: list[FetchedContent]) -> list[Summary]:
        """Step 4: Summarize - Manually generate summaries for each piece of content.

        This shows the manual iteration and error handling required
        for AI generation tasks.
        """
        print("\n🤖 Generating summaries...")

        summaries = []
        failed_count = 0

        for i, content_item in enumerate(fetched_content):
            print(f"\n--- Summarizing {i+1}/{len(fetched_content)}: {content_item.task.topic} ---")

            summary_text = self.generate_summary(
                content_item.task.topic,
                content_item.content,
                content_item.url,
            )

            if summary_text is not None:
                # Manual object creation
                summary = Summary(
                    topic=content_item.task.topic,
                    summary=summary_text,
                    source_url=content_item.url,
                )
                summaries.append(summary)
                print(f"✅ Summary created for: {content_item.task.topic}")
            else:
                failed_count += 1
                print(f"❌ Failed to generate summary for: {content_item.task.topic}")
                # Manual error tracking
                continue

        print(f"\n✅ Summary generation complete: {len(summaries)} successful, {failed_count} failed")
        return summaries

    def generate_markdown_report(
        self,
        summaries: list[Summary],
        output_file: str = "newsletter.md",
    ) -> None:
        """Step 5: Report - Manually generate a formatted Markdown report.

        This demonstrates manual template construction and file operations
        that would be handled automatically by a framework.
        """
        print(f"\n📝 Generating report: {output_file}")

        try:
            # Manual report template construction
            report_lines = [
                "# AI Research Newsletter",
                "",
                "*Generated by Traditional Research Agent*",
                f"*Total articles summarized: {len(summaries)}*",
                "",
            ]

            # Manual iteration through summaries
            for i, summary in enumerate(summaries):
                report_lines.extend(
                    [
                        f"## {summary.topic}",
                        "",
                        f"**Source:** {summary.source_url}",
                        "",
                        summary.summary,
                        "",
                    ],
                )

                # Manual separator logic
                if i < len(summaries) - 1:
                    report_lines.append("---")
                    report_lines.append("")

            # Manual file writing with error handling
            report_content = "\n".join(report_lines)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_content)

            print(f"✅ Report saved to {output_file}")
            print(f"📊 Report contains {len(summaries)} summaries")

        except OSError as e:
            print(f"❌ Error generating report: {e}")

    def run(
        self,
        sources_file: str = "../assets/sources.json",
        output_file: str = "newsletter.md",
    ) -> None:
        """Orchestrate the entire workflow.

        This demonstrates the manual orchestration and state management
        required when building workflows without a framework.
        """
        print("🚀 Starting Traditional Research Agent")
        print("=" * 50)

        try:
            # Step 1: Manual ingestion
            tasks_data = self.load_tasks_from_json(sources_file)
            if not tasks_data:
                print("❌ No tasks to process. Exiting.")
                return

            # Step 2: Manual validation
            valid_tasks = self.validate_tasks(tasks_data)
            if not valid_tasks:
                print("❌ No valid tasks found. Exiting.")
                return

            # Step 3: Manual fetching
            fetched_content = self.fetch_content_for_tasks(valid_tasks)
            if not fetched_content:
                print("❌ No content was successfully fetched. Exiting.")
                return

            # Step 4: Manual summarization
            summaries = self.generate_summaries(fetched_content)
            if not summaries:
                print("❌ No summaries were generated. Exiting.")
                return

            # Step 5: Manual report generation
            self.generate_markdown_report(summaries, output_file)

            print("\n" + "=" * 50)
            print("🎉 Traditional Research Agent completed successfully!")
            print(f"📄 Final report saved to: {output_file}")

        except KeyboardInterrupt:
            print("\n❌ Process interrupted by user")
        except (ValueError, OSError) as e:
            print(f"\n❌ Unexpected error in main workflow: {e}")
        finally:
            # Manual cleanup
            self.http_client.close()

    def __del__(self) -> None:
        """Manual cleanup of resources."""
        if hasattr(self, "http_client"):
            with contextlib.suppress(AttributeError, RuntimeError):
                self.http_client.close()


def main() -> None:
    """Entry point for the traditional research agent.

    This demonstrates the manual setup and configuration
    required for a traditional Python application.
    """
    print("Traditional Research Agent")
    print("This implementation showcases manual control flow and boilerplate code")
    print("that will be contrasted with the Gemini Processors approach.\n")

    # Manual argument handling
    sources_file = "../assets/sources.json"
    output_file = "newsletter.md"

    if len(sys.argv) > 1:
        sources_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    # Manual environment variable checking
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ ERROR: GEMINI_API_KEY environment variable is required")
        print("Please set it with: export GEMINI_API_KEY='your-api-key'")
        sys.exit(1)

    try:
        # Manual agent instantiation and execution
        agent = TraditionalResearchAgent()
        agent.run(sources_file, output_file)

    except (ValueError, OSError) as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
