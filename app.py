from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
#from langchain.chat_models import ChatOpenAI

def get_weather(city: str) -> str:
    """ get the weather for a given city """
    return f"The current weather in {city} is sunny with a temperature of 25°C."

agent = create_agent(
    model="google_genai:gemini-3.5-flash-lite",
    tools=[get_weather],
    system_prompt= "you are helpful assistant",
)

result =agent.invoke(
    {"messages":[{"role":"user","content":"what is the weather in San Francisco and New York?"}]}
)
print(result["messages"][2].text)  # goes for San Francisco
# print(result["messages"][-1].text)  # goes for New York





# BUILD A REAL AGENT - RESEARCH AGENT THAT CAN ANSWER QUESTION ABOUT TEXT FILES

# 1. Detailed system prompt for better agent behaviour
# 2. create tools that integrates with external data
# 3. model configuration for consistent responses
# 4. conversational memory for chat-like interactions
# 5. Deep agents for built-in features
# 6. Testing your agents

#1. Define the system prompt
# The system prompt defines the agent's role and beahaviour. keep it specific and actionable.
''' SYSTEM_PROMPT = """ You are a litterally data assistant.

## capabilities
- - `fetch_text_from_url`: loads document text from a URL into the conversation.
Do not guess line counts or positions—ground them in tool results from the saved file."""

#2. Create tools
# Tools let a model interact with external system(data/API) by calling functions you define.
# Tools can depend on runtime context and also interacts with agent memory. 
# Tools can be used to fetch data, perform calculations, or interact with APIs.

# this example uses a tool to load a document from a given URL
import urllib.error
import urllib.request

from langchain.tools import tool


@tool
def fetch_text_from_url(url: str) -> str:
    """ Fetch the document from a URL
    """
    req = urllib.request.Request(
        url,
        headers={"user-Agent":"Mozialla/5.0 (Compatable; quickstart-research/1.0)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read()
    except urllib.error.URLError as e:
        return f"Fetching Failed: {e}"
    text = raw.decode("utf-8", errors="replace")
    return text

#3. Model configuration
# Model configuration allows you to customize the model's behavior and output.
#  set up your language model with the right parameters for your case

# note: dependin on the model provider chosen, initialization parameters may vary; refer to their
# reference pages and details
from langchain.chat_models import init_chat_model
model = init_chat_model(
    "gemini-3.1-pro-preview",
    model_provider = "google_genai",
    temperature = 0.5,
    timeout = 600,
    max_tokens =25000,
    streaming = True
    )

#4. Add Memory: to your agent
# add memory to your agent to maintain state across interactions. This allows your agent to remember information from previous conversations and context.
# Memory allows your agent to remember information across interactions, enabling more context-aware responses.

from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

#5. create and run agents
# Now assemble your agent with all the components and run it.
# there are two different frameworks for creating agents: Langchain agent and deep agents.
# BOth Langchain and deep agents provide you with a fine grained control over tools, memory, and more.
# The main difference between both is that deep agents come with commonly useful capabilities built-in, such as planning, file-system tools and subagents, while Langchain agents require you to define them yourself.
# There are two different ways to create an agent: using the `create_agent` function or using the `Agent` class directly. The former is a simpler approach, while the latter provides more flexibility and control over the agent's behavior.

# note: use deep agents when you want maximum capabilities with minimal setup, chose Langchain agents when you want more control over the agent's behavior and capabilities.

# note: Let's try both
from langchain.agents import create_agent
from deepagents import create_deep_agent

agent = create_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer
)

deep_agent = create_deep_agent(
    model=model,
    tools=[fetch_text_from_url],
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer
    )


 content = f""" project Gutenberg host a full plain-text copy of F.Scott Fitzgerald's The Great Gatsby at the following URL: https://www.gutenberg.org/files/64317/64317-0.txt. 
 Answer as much as you can:
 1. How many lines in the complete Gutenberg file contain the substring "Gatsby" (count lines, not occurrences within a line, each line ends with a line break).
2) The 1-based line number of the first line in the file that contains `Daisy`.
3) A two-sentence neutral synopsis.

Do your best on (1) and (2). If at any point you realize you cannot **verify** an exact answer with
your available tools and reasoning, do not fabricate numbers: use `null` for that field and spell out
the limitation in `how_you_computed_counts`. If you encounter any errors please report what the error was and what the error message was."""
 
 
print("Running create_agent...", flush=True)
agent_result = agent.invoke(
        {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "great-gatsby-lc"}},
)
print("Running create_deep_agent...", flush=True)
deep_agent_result = deep_agent.invoke(
    {"messages": [{"role": "user", "content": content}]},
    config={"configurable": {"thread_id": "great-gatsby-da"}},
)


print("\ncreate_agent:")
print(agent_result["messages"][-1].content_blocks)
print("\ncreate_deep_agent:")
print(deep_agent_result["messages"][-1].content_blocks)

'''