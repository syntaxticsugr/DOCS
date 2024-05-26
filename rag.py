from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor



def langchain_rag_search(apikey_openai, docs_link, search_query):

    loader = WebBaseLoader(docs_link)

    docs = loader.load()
    docs = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)

    docs_vectordb = FAISS.from_documents(docs, OpenAIEmbeddings(api_key=apikey_openai))

    docs_retriever = docs_vectordb.as_retriever()

    docs_retriever_tool = create_retriever_tool(docs_retriever, name="doc_search", description="Search for the specified information about the given query.")

    tools = [docs_retriever_tool]

    llm = ChatOpenAI(api_key=apikey_openai, temperature=0)

    prompt = hub.pull("hwchase17/openai-functions-agent")

    agent = create_openai_tools_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    result = agent_executor.invoke({"input": search_query})

    return result['output']
