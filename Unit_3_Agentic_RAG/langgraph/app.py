from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from tools import DuckDuckGoSearchRun, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool


# 初始化网络搜索工具
search_tool = DuckDuckGoSearchRun()

# 生成包含工具的聊天接口
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token="",
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool]
chat_with_tools = chat.bind_tools(tools)

# 生成 AgentState 和 Agent 图
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

##  构建流程图
builder = StateGraph(AgentState)

# 定义节点：执行具体工作
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# 定义边：控制流程走向
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # 如果最新消息需要工具调用，则路由到 tools 节点
    # 否则直接响应
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

# 示例 1：查找嘉宾信息
response = alfred.invoke({"messages": "Tell me about 'Lady Ada Lovelace'"})
print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

# 示例 2：烟花天气核查
response = alfred.invoke({"messages": "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?"})
print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

# 示例 3：给 AI 研究者留下深刻印象
response = alfred.invoke({"messages": "One of our guests is from Qwen. What can you tell me about their most popular model?"})
print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

# 示例 4：组合多工具应用
response = alfred.invoke({"messages":"I need to speak with 'Dr. Nikola Tesla' about recent advancements in wireless energy. Can you help me prepare for this conversation?"})
print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

# 高级功能：对话记忆
# 首次交互
response = alfred.invoke({"messages": [HumanMessage(content="Tell me about 'Lady Ada Lovelace'. What's her background and how is she related to me?")]})
print("🎩 Alfred's Response:")
print(response['messages'][-1].content)
print()
# 二次交互（引用首次内容）
response = alfred.invoke({"messages": response["messages"] + [HumanMessage(content="What projects is she currently working on?")]})
print("🎩 Alfred's Response:")
print(response['messages'][-1].content)