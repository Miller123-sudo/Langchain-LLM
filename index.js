import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import readline from "node:readline/promises";
import { ChatGroq } from "@langchain/groq";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { TavilySearch } from "@langchain/tavily";
import { MemorySaver } from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const checkpointer = new MemorySaver();

/**
 * Tool 1: Tavily web search (your original tool, renamed to avoid clashing
 * with the `tool` helper imported above)
 */
const tavilyTool = new TavilySearch({
  maxResults: 1,
  topic: "general",
});

/**
 * Tool 2: Weather (mock data, so you can test without another API key)
 */
const getWeather = tool(
  async ({ city }) => {
    console.log(`  [tool] getWeather -> ${city}`);
    const fakeWeather = {
      kolkata: "32°C, humid, light rain",
      delhi: "28°C, hazy",
      mumbai: "30°C, cloudy",
    };
    return fakeWeather[city.toLowerCase()] || `No weather data for ${city}`;
  },
  {
    name: "get_weather",
    description: "Get the current weather for a city.",
    schema: z.object({
      city: z.string().describe("City name, for example Kolkata"),
    }),
  },
);

/**
 * Tool 3: Calculator
 */
const calculator = tool(
  async ({ expression }) => {
    console.log(`  [tool] calculator -> ${expression}`);
    // Allow only digits, operators, dots, brackets and spaces
    if (!/^[0-9+\-*/().\s]+$/.test(expression)) {
      return "Invalid expression";
    }
    try {
      return String(Function(`"use strict"; return (${expression})`)());
    } catch {
      return "Could not calculate";
    }
  },
  {
    name: "calculator",
    description: "Evaluate a math expression such as (25 * 4) + 10.",
    schema: z.object({
      expression: z.string().describe("Math expression to evaluate"),
    }),
  },
);

/**
 * Tool 4: Current time in a time zone
 */
const getCurrentTime = tool(
  async ({ timezone }) => {
    console.log(`  [tool] getCurrentTime -> ${timezone}`);
    try {
      return new Date().toLocaleString("en-US", { timeZone: timezone });
    } catch {
      return `Unknown time zone: ${timezone}`;
    }
  },
  {
    name: "get_current_time",
    description: "Get the current date and time for an IANA time zone.",
    schema: z.object({
      timezone: z.string().describe("IANA time zone, for example Asia/Kolkata"),
    }),
  },
);

/**
 * Initialized tool node (it runs every tool call from one AI message in parallel)
 */
const tools = [tavilyTool, getWeather, calculator, getCurrentTime];
const toolNode = new ToolNode(tools);

/**
 * Initializing llm model
 */
const model = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0, // This mean not creative, but more factual and deterministic response
  maxRetries: 2,
}).bindTools(tools);

const SYSTEM_PROMPT =
  "You are a helpful assistant with tools for web search, weather, math and time. " +
  "When a question has several independent parts, call all the needed tools " +
  "in the same step instead of one by one. After you get the tool results, " +
  "combine them into one clear answer.";

const callModel = async (state) => {
  // Call the llm using APIs
  console.log("calling llm");

  // The system prompt is added on every call but is not saved in the state
  const response = await model.invoke([
    { role: "system", content: SYSTEM_PROMPT },
    ...state.messages,
  ]);

  console.log("call model: ", response.tool_calls);

  return { messages: [response] };
};

/**
 * Conditional edge: decide where to go after the agent node
 */
function shouldContinue(state) {
  const lastMessage = state.messages[state.messages.length - 1];
  const toolCalls = lastMessage.tool_calls || [];
  console.log("toolCalls: ", toolCalls);

  // Condition 1: no tool requested, so the answer is final
  if (toolCalls.length === 0) {
    return "__end__";
  }

  // Condition 2: more than one tool requested, so this is a multi tool call
  if (toolCalls.length > 1) {
    console.log(
      `multi tool call: ${toolCalls.length} tools -> ${toolCalls
        .map((t) => t.name)
        .join(", ")}`,
    );
  } else {
    console.log(`single tool call: ${toolCalls[0].name}`);
  }

  // ToolNode runs all of these calls in parallel, then returns to the agent
  return "tools";
}

/**
 * Build the graph
 */
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge("__start__", "agent")
  .addConditionalEdges("agent", shouldContinue)
  .addEdge("tools", "agent");
// Note: no .addEdge("agent", "__end__") here, because shouldContinue
// already decides when the graph ends.

/**
 * Compile the graph
 */
const app = workflow.compile({ checkpointer });

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log(
    'Try: "What is the weather in Kolkata and Delhi, the time in Asia/Kolkata, and what is 25 * 4?"',
  );

  while (true) {
    const userInput = await rl.question("You:");
    if (userInput === "bye") break;

    const finalState = await app.invoke(
      {
        messages: [{ role: "user", content: userInput }],
      },
      { configurable: { thread_id: "1" }, recursionLimit: 10 },
    );

    console.log(
      "Final State:",
      finalState.messages[finalState.messages.length - 1].content,
    );
  }

  rl.close();
}

main();

// Old working code

// import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
// import readline from "node:readline/promises";
// import { ChatGroq } from "@langchain/groq";
// import { ToolNode } from "@langchain/langgraph/prebuilt";
// import { TavilySearch } from "@langchain/tavily";
// import { MemorySaver } from "@langchain/langgraph";

// const checkpointer = new MemorySaver();

// /**
//  * Initialized tool node
//  */
// const tool = new TavilySearch({
//   maxResults: 1,
//   topic: "general",
//   // includeAnswer: false,
//   // includeRawContent: false,
//   // includeImages: false,
//   // includeImageDescriptions: false,
//   // searchDepth: "basic",
//   // timeRange: "day",
//   // includeDomains: [],
//   // excludeDomains: []
// });
// const tools = [tool];
// const toolNode = new ToolNode(tools);

// /**
//  * 1. Define node function
//  * 2. build the graph
//  * 3. compile and invoke the graph
//  */

// /**
//  * Initializing llm model
//  */
// const model = new ChatGroq({
//   model: "openai/gpt-oss-120b",
//   temperature: 0, // This mean not creative, but more factual and deterministic response
//   maxRetries: 2,
// }).bindTools(tools);

// const callModel = async (state) => {
//   // Call the llm using APIs

//   console.log("calling llm");

//   const response = await model.invoke(state.messages);

//   return { messages: [response] };
// };

// function shouldContinue(state) {
//   // whether to call a tool or end
//   const lastMessage = state.messages[state.messages.length - 1];
//   if (lastMessage.tool_calls.length > 0) {
//     return "tools";
//   }
//   return "__end__";
// }

// /**
//  * Build the graph
//  */
// const workflow = new StateGraph(MessagesAnnotation)
//   .addNode("agent", callModel)
//   .addNode("tools", toolNode)
//   .addEdge("__start__", "agent")
//   .addConditionalEdges("agent", shouldContinue)
//   .addEdge("tools", "agent")
//   .addEdge("agent", "__end__");

// /**
//  * Compile the graph
//  */
// const app = workflow.compile({ checkpointer });

// async function main() {
//   const rl = readline.createInterface({
//     input: process.stdin,
//     output: process.stdout,
//   });

//   while (true) {
//     const userInput = await rl.question("You:");
//     if (userInput === "bye") break;

//     const finalState = await app.invoke(
//       {
//         //   messages: [{ role: "user", content: "What is the weather in kolkata?" }],
//         messages: [{ role: "user", content: userInput }],
//       },
//       { configurable: { thread_id: "1" } },
//     );

//     console.log(
//       "Final State:",
//       finalState.messages[finalState.messages.length - 1].content,
//     );
//   }

//   rl.close();
// }

// main();
