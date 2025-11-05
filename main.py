# main.py — 兼容 LangChain 1.0.3：自动适配 create_agent / create_tool_calling_agent 的签名
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

# 尝试优先使用新版函数；不可用则回退
USE_TOOL_CALLING = True
try:
    from langchain.agents import create_tool_calling_agent as _create_agent_impl
except Exception:
    from langchain.agents import create_agent as _create_agent_impl
    USE_TOOL_CALLING = False

# 可选的本地 Prompt（如果目标函数不支持 prompt，会自动忽略）
from langchain_core.prompts import ChatPromptTemplate
LOCAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful AI assistant. Use tools when needed and return a concise final answer."),
    ("human", "{input}")
])

load_dotenv()  # 需要 .env 里有 OPENAI_API_KEY、TAVILY_API_KEY

def create_agent_compat(llm, tools, prompt):
    """根据实际函数签名自动决定是否传 prompt。"""
    from inspect import signature
    sig = signature(_create_agent_impl)
    params = sig.parameters
    if "prompt" in params:
        return _create_agent_impl(llm, tools, prompt=prompt)
    else:
        # 你的 1.0.3 会走这里：create_agent/create_tool_calling_agent 不接受 prompt
        return _create_agent_impl(llm, tools)

def build_agent():
    tools = [TavilySearch(max_results=5)]
    # 没权限就换成 gpt-4o / gpt-4.1-mini / gpt-3.5-turbo
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_agent_compat(llm, tools, prompt=LOCAL_PROMPT)
    return agent

def main():
    agent = build_agent()
    # 1.x 的 agent（Runnable）用 messages 形式调用
    result = agent.invoke({
        "messages": [("user", "search for 3 AI engineer jobs and summarize the company & title")]
    })
    # 打印最后一条模型回复
    msgs = result.get("messages", [])
    print("\n=== FINAL OUTPUT ===")
    print(msgs[-1].content if msgs else result)

if __name__ == "__main__":
    main()
