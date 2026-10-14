import { StateGraph, MessagesAnnotation } from "@langchain/langgraph";
import readline from "node:readline/promises";
import { ChatGroq } from "@langchain/groq";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { TavilySearch } from "@langchain/tavily";
import { MemorySaver } from "@langchain/langgraph";

const checkpointer = new MemorySaver();

/**
 * Initialized tool node
 */
const tool = new TavilySearch({
  maxResults: 1,
  topic: "general",
  // includeAnswer: false,
  // includeRawContent: false,
  // includeImages: false,
  // includeImageDescriptions: false,
  // searchDepth: "basic",
  // timeRange: "day",
  // includeDomains: [],
  // excludeDomains: []
});
const tools = [tool];
const toolNode = new ToolNode(tools);

/**
 * 1. Define node function
 * 2. build the graph
 * 3. compile and invoke the graph
 */

/**
 * Initializing llm model
 */
const model = new ChatGroq({
  model: "openai/gpt-oss-120b",
  temperature: 0, // This mean not creative, but more factual and deterministic response
  maxRetries: 2,
}).bindTools(tools);

const callModel = async (state) => {
  // Call the llm using APIs

  console.log("calling llm");

  const response = await model.invoke(state.messages);

  return { messages: [response] };
};

function shouldContinue(state) {
  // whether to call a tool or end
  const lastMessage = state.messages[state.messages.length - 1];
  if (lastMessage.tool_calls.length > 0) {
    return "tools";
  }
  return "__end__";
}

/**
 * Build the graph
 */
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge("__start__", "agent")
  .addConditionalEdges("agent", shouldContinue)
  .addEdge("tools", "agent")
  .addEdge("agent", "__end__");

/**
 * Compile the graph
 */
const app = workflow.compile({ checkpointer });

async function main() {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  while (true) {
    const userInput = await rl.question("You:");
    if (userInput === "bye") break;

    const finalState = await app.invoke(
      {
        //   messages: [{ role: "user", content: "What is the weather in kolkata?" }],
        messages: [{ role: "user", content: userInput }],
      },
      { configurable: { thread_id: "1" } },
    );

    console.log(
      "Final State:",
      finalState.messages[finalState.messages.length - 1].content,
    );
  }

  rl.close();
}

main();
