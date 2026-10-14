# LangChain Tool-Calling Assistant

A command-line AI assistant built with LangChain and LangGraph. It uses a Groq-hosted language model to answer questions and call tools for web search, sample weather data, arithmetic, and time-zone lookups. Independent tool calls can run in parallel.

## Features

- **Web search:** Uses Tavily and returns up to one result per search.
- **Weather:** Returns mock weather for Kolkata, Delhi, and Mumbai; no weather API key is needed.
- **Calculator:** Evaluates basic arithmetic expressions.
- **Time lookup:** Returns the current date and time for an IANA time zone, such as `Asia/Kolkata`.
- **Conversation state:** Uses LangGraph's in-memory checkpointer for the current process.

## Requirements

- Node.js 20.6 or later
- A Groq API key
- A Tavily API key for web search

## Setup

Install the dependencies:

```bash
npm install
```

Create a `.env` file in the project root with your API keys:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

The `.gitignore` excludes `.env` files. Keep API keys private and do not commit them.

## Run

From the project directory, start the assistant with:

```bash
node --env-file=.env index.js
```

Enter a question at the `You:` prompt. Type `bye` to exit.

For example:

```text
What is the weather in Kolkata and Delhi, the time in Asia/Kolkata, and what is 25 * 4?
```

The assistant can request multiple tools for a question and combine their results into one response.

## Tools and Configuration

The agent is configured to use the `openai/gpt-oss-120b` model through Groq, with temperature set to `0`. Tavily search is configured for general search with one result. Weather responses are fixed sample data, not live forecasts.

LangGraph routes each model response either to the tool node or to the end of the conversation. The in-memory checkpointer uses a single thread ID, so conversation state is not retained after the process exits.
