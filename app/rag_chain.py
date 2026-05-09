from typing import List
from langchain_core import Document
from langchain_openai import ChatOpenAI
from app.vector_store import vector_store
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StdOutputParser


class rag_chain:
    def __init__(self, vs: vector_store):
        self.chain = None
        self.vs = vector_store
        pass

    async def query(self, question: str) -> str:
        """Query the RAG chain with a question.
        Note:
            - Must return an answer.
        """
        documents = self.vs.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in documents])
        answer = await self.chain.invoke({
            "question": question,
            "context": context
        })
        return answer

    def get_relevant_documents(self, query: str, k: int = 5) -> List[Document]:
        """Get relevant documents with metadata.
        IMPORTANT: -
        Empty queries or queries shorter than 10 characters MUST be rejected with
        ValueError - The exact error message should be: "Query too short."
        Raises: ValueError: If query is empty or too short (less than 10 characters)
        """
        if not query or len(query) < 10:
            raise ValueError("Query too short.")

        relevant_docs = self.vs.similarity_search(query, k=k)
        return relevant_docs

    def create_chain(self):
        """Perform RAG chain with a query.
        Note:
            - Must return an answer.
        """
        llm = ChatOpenAI(temperature=0.9)
        prompt = self.get_prompt()

        # LCEL chain
        self.chain = prompt | llm | StdOutputParser()

    def get_prompt(self) -> str:
        """Get prompt for RAG chain.
        Note:
            - Must return a prompt.
        """
        prompt = ChatPromptTemplate.from_template("""You are a helpful assistant that answers questions based
        Answer the question based on the retrieved documents. If you don't know the answer, say you don't know.
        on the following retrieved documents.
        Context: {context}
        Question: {question}
        Answer:
        """)
        return prompt
