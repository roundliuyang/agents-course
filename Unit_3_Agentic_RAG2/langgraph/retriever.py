import datasets
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import Tool
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# 加载数据集
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# 转换为 Document 对象
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_dataset
]


# 创建 BM25 检索器，基于文档集合建立索引以便进行关键词匹配检索
bm25_retriever = BM25Retriever.from_documents(docs)

def extract_text(query: str) -> str:
    """
    根据查询关键词检索嘉宾详细信息
    Args:
        query: 查询字符串，通常是嘉宾姓名或关系描述
    Returns:
        匹配的嘉宾信息或未找到的提示信息
    """
    # 执行检索，查找与查询最相关的文档
    results = bm25_retriever.invoke(query)
    if results:
        # 返回前3个最相关结果的页面内容，用双换行分隔
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        # 如果没有找到匹配项，返回提示信息
        return "No matching guest information found."

# 创建工具对象，封装检索功能供LangGraph使用
guest_info_tool = Tool(
    name="guest_info_retriever",    # 工具名称
    func=extract_text,    # 关联实际执行的函数
    description="Retrieves detailed information about gala guests based on their name or relation."   # 工具功能说明
)

# 使用 Hugging Face 模型作为 LLM，这里选择 Qwen2.5-7B-Instruct 模型
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token="",
)

# 将 LLM 封装为 Chat 模型实例
chat = ChatHuggingFace(llm=llm, verbose=True)
# 定义可用工具列表，这里只有一个客人信息检索工具
tools = [guest_info_tool]
# 将工具绑定到聊天模型上，使模型能够知道何时使用工具
chat_with_tools = chat.bind_tools(tools)

# 定义AgentState的数据结构，继承自 TypedDict
class AgentState(TypedDict):
    # 消息列表，使用 add_messages 函数进行累加操作
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    """
       助手节点函数，负责处理输入状态并返回新的消息
       Args:
           state: 包含当前对话消息的状态对象
       Returns:
           包含新消息的状态更新字典
    """
    ai_message = chat_with_tools.invoke(state["messages"])
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(ai_message)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    return {
        "messages": [ai_message],
    }

## 构建流程图
builder = StateGraph(AgentState)  # 创建状态图实例，使用 AgentState 作为状态结构

# 定义节点：这些节点完成工作
builder.add_node("assistant", assistant)   # 添加助手节点，执行核心逻辑
builder.add_node("tools", ToolNode(tools))  # 添加工具节点，执行工具调用

# 定义边：这些决定了控制流如何移动
builder.add_edge(START, "assistant")  # 从开始节点连接到助手节点
builder.add_conditional_edges(
    "assistant",
    # 条件边：如果最新消息需要工具，则路由到工具节点
    # 否则，直接返回响应
    tools_condition,  # 决定下一步流向的条件函数
)
builder.add_edge("tools", "assistant")  # 从工具节点返回到助手节点
alfred = builder.compile()  # 编译状态图，生成可执行的代理

# 测试消息，询问关于特定客人的信息
messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]

# 执行agent，传入初始消息
response = alfred.invoke({"messages": messages})

# 输出最终响应结果
print("🎩 Alfred's Response:")
print(response['messages'][-1].content)  # 打印最后一条消息的内容