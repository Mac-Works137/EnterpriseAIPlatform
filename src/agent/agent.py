
"""Gemini-powered agent for the Enterprise Knowledge Platform."""

import logging
import os
from pathlib import Path

from time import perf_counter

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.callbacks import BaseCallbackHandler

from src.tools.retrieval_tool import retrieve_documents
from src.tools.web_search_tool import web_search_tool
from src.tools.web_search_tool import reset_search_budget
from src.tools.sql_tool import (
    DB_FILE,
    initialize_database,
    query_sales_database,
)
from src.tools.python_tool import calculate_statistics


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = "gemini-3.6-flash"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ToolTimingCallback(BaseCallbackHandler):
    """Log the duration and outcome of each tool execution."""

    def __init__(self):
        self.tool_start_times = {}

    def on_tool_start(self, serialized, input_str, **kwargs):
        run_id = kwargs.get("run_id")
        tool_name = serialized.get("name", "unknown")

        if run_id is not None:
            self.tool_start_times[run_id] = (
                tool_name,
                perf_counter(),
            )

        logger.info("Tool started: %s", tool_name)

    def on_tool_end(self, output, **kwargs):
        run_id = kwargs.get("run_id")
        timing = self.tool_start_times.pop(run_id, None)

        if timing:
            tool_name, start_time = timing
            elapsed = perf_counter() - start_time

            logger.info(
                "Tool completed: %s | Duration: %.3f seconds",
                tool_name,
                elapsed,
            )

    def on_tool_error(self, error, **kwargs):
        run_id = kwargs.get("run_id")
        timing = self.tool_start_times.pop(run_id, None)

        if timing:
            tool_name, start_time = timing
            elapsed = perf_counter() - start_time

            logger.error(
                "Tool failed: %s | Duration: %.3f seconds | Error: %s",
                tool_name,
                elapsed,
                error,
            )

# --------------------------------------------------
# Gemini setup
# --------------------------------------------------

def get_api_key() -> str:
    """Read the Gemini API key from environment variables."""

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("gemini_key")
    )

    if not api_key:
        raise ValueError(
            "Gemini API key not found. "
            "Check the existing .env configuration."
        )

    return api_key


def create_llm() -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model."""

    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=get_api_key(),
    )


# --------------------------------------------------
# Agent setup
# --------------------------------------------------


SYSTEM_PROMPT = """
You are an enterprise knowledge and workflow intelligence assistant.

Your job is to answer accurately using the available tools and to
explain what the evidence supports.

TOOL SELECTION RULES

1. Local knowledge-base questions:
   Use retrieve_documents for questions about the Azure documentation
   indexed in the local knowledge base.

2. Current events and external research:
   Use web_search_tool for recent developments, current information,
   or questions that require information from the public web.
   Do not call retrieve_documents unless the user also asks about
   the local knowledge base or requests a comparison with it.

3. Evidence verification and comparison:
   When asked to verify local documentation against current official
   sources, use retrieve_documents and web_search_tool.
   Compare the evidence and explicitly identify discrepancies.
   If evidence is missing or insufficient, say so.

4. Sales and database questions:
   Use query_sales_database for questions that require sales records,
   regional revenue, aggregations, or other information in the sales
   database. Do not invent database results.

5. Numerical calculations:
   Use calculate_statistics when a supported statistical calculation
   is requested. Do not claim that you used a tool unless you actually
   called it.

EFFICIENCY RULES

- Select only the tools needed to answer the user's question.
- Avoid duplicate searches and repeated retrieval when existing
  evidence is sufficient.
- Do not call every tool for every question.
- For multi-part questions, use the relevant tools for each part.
- If a tool fails, do not pretend it succeeded.
- Do not invent search results, database records, calculations,
  quotations, or source URLs.

TOOL EXECUTION AND STOPPING RULES:

- For a question that requires current external information, call
  web_search_tool and use its returned results to answer.

- A single successful web search is normally sufficient for a
  straightforward information request. Do not repeat the same search
  unless the initial results are empty, clearly insufficient,
  contradictory, or missing essential information.

- Do not call the same tool repeatedly with identical or substantially
  similar arguments.

- For current-version questions about software libraries, frameworks,
  or platforms, use web_search_tool to obtain up-to-date information.
  Prefer official release pages, documentation, or repositories.

- Once sufficient evidence has been gathered to answer the question,
  stop calling tools and generate the final response.

- Do not call additional tools merely to make an answer more detailed.
  Prioritize relevant evidence over the number of tool calls.

ANSWER QUALITY

- Give a direct answer first, followed by the reasoning.
- Support factual claims with evidence from the tools you used.
- When using web research, prioritize official and authoritative
  sources for technical or enterprise recommendations.
- Distinguish retrieved evidence from your own inference.
- Include source titles and URLs when the tools provide them.
- Be transparent about uncertainty and missing evidence.


GENERAL KNOWLEDGE AND NO-TOOL ROUTING:

- Answer greetings, acknowledgements, and simple conversational
  messages directly without calling any tools.

- Answer basic definitions, common concepts, and general educational
  questions using your existing knowledge when you can answer reliably.

- Do not call web_search_tool merely because a question is factual,
  conceptual, or educational.

- Use web_search_tool when the user explicitly requests recent,
  latest, current, or external information, or when up-to-date
  external research is genuinely necessary.

- If a question can be answered accurately without external tools,
  do not call any tool.

- Avoid unnecessary tool calls. Use only the tools that materially
  contribute to answering the user's question.

"""


def create_enterprise_agent():
    """Create the agent and register the four tools."""

    llm = create_llm()

    tools = [
        retrieve_documents,
        web_search_tool,
        query_sales_database,
        calculate_statistics,
    ]

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent


# --------------------------------------------------
# Run a user query
# --------------------------------------------------


def ask_agent(query: str) -> str:
    """Run a query through the agent and return its final answer."""

    if not query or not query.strip():
        raise ValueError("Please enter a non-empty question.")

    query = query.strip()

    reset_search_budget()

    # Initialize the sales database only if it does not exist.
    if not DB_FILE.exists():
        logger.info("Sales database not found; initializing it.")
        initialize_database()

    agent = create_enterprise_agent()

    start_time = perf_counter()

    try:
        try:
            result = agent.invoke(
                {
                    "messages": [
                        {"role": "user", "content": query}
                    ]
                },
                config={
                    "recursion_limit": 12,
                    "callbacks": [ToolTimingCallback()],
                },
            )

        except Exception as exc:
            # Log diagnostic information when execution fails.
            logger.exception("Agent execution failed.")

            # Try to extract partial state from the exception.
            partial_state = getattr(exc, "state", None)

            if partial_state:
                messages = partial_state.get("messages", [])

                logger.error(
                    "Agent stopped after %d messages.",
                    len(messages),
                )

                for message in messages:
                    tool_calls = getattr(message, "tool_calls", [])

                    for tool_call in tool_calls:
                        logger.error(
                            "Trace - selected tool: %s | Arguments: %s",
                            tool_call.get("name", "unknown"),
                            tool_call.get("args", {}),
                        )

                    if getattr(message, "type", "") == "tool":
                        logger.error(
                            "Trace - executed tool: %s | Result: %s",
                            getattr(message, "name", "unknown"),
                            str(message.content)[:300],
                        )

            raise

    except Exception as exc:
        error_message = str(exc).lower()

        is_quota_error = (
            getattr(exc, "status_code", None) == 429
            or getattr(exc, "code", None) == 429
            or "resource_exhausted" in error_message
            or "quota" in error_message
            or "rate limit" in error_message
            or "429" in error_message
        )

        if is_quota_error:
            logger.exception(
                "Gemini API quota or rate limit exceeded."
            )

            return (
                "The AI service has temporarily reached its usage "
                "limit. Please try again later."
            )

        raise

    finally:
        elapsed_time = perf_counter() - start_time

        logger.info(
            "Agent request duration: %.2f seconds",
            elapsed_time,
        )

    # Log tool calls to understand the agent's decisions.
    for message in result["messages"]:
        tool_calls = getattr(message, "tool_calls", [])

        for tool_call in tool_calls:
            logger.info(
                "Agent selected tool: %s | Arguments: %s",
                tool_call.get("name", "unknown"),
                tool_call.get("args", {}),
            )

    # Log actual tool execution results.
    for message in result["messages"]:
        if getattr(message, "type", "") == "tool":
            logger.info(
                "Tool executed: %s | Result preview: %s",
                getattr(message, "name", "unknown"),
                str(message.content)[:200],
            )

    # Extract the final assistant response.
    final_message = result["messages"][-1]
    content = final_message.content

    if isinstance(content, str):
        return content.strip()

    # Handle responses containing multiple content blocks.
    if isinstance(content, list):
        text_parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and "text" in block
        ]
        return "\n".join(text_parts).strip()

    return str(content)


# --------------------------------------------------
# Test
# --------------------------------------------------

def main():
    query = (
        "Find guidance on securing Azure Blob Storage in the local knowledge "
        "base. Use web search to verify the most important recommendations "
        "against current official Microsoft sources. Identify discrepancies, "
        "cite the evidence, and state when the available evidence is insufficient."
    )

    print("\n" + "=" * 80)
    print("ENTERPRISE KNOWLEDGE & WORKFLOW INTELLIGENCE PLATFORM")
    print("=" * 80)
    print(f"\nQuestion: {query}\n")

    try:
        answer = ask_agent(query)
        print("Answer:")
        print(answer)

    except Exception:
        logger.exception("Agent execution failed.")
        print(
            "\nThe agent could not complete the request. "
            "Check the error details in the terminal."
        )


if __name__ == "__main__":
    main()
