import streamlit as st
import os
import tempfile
import numpy as np
import time
import uuid
from datetime import datetime
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter

# === GUARDRAILS SETUP ===
from guardrails import Guard
from guardrails.hub import (
    PolitenessCheck, ResponsivenessCheck, LlmRagEvaluator,
    HallucinationPrompt, QARelevanceLLMEval, LLMCritic,
    ResponseEvaluator, UnusualPrompt, ProvenanceLLM,
    RestrictToTopic, SaliencyCheck
)

# === ADDITIONAL IMPORTS FOR NEW GUARDRAILS ===
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    st.warning("⚠️ sentence-transformers not installed. ProvenanceLLM guardrail will be disabled.")

# === CHROMADB IMPORTS ===
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    st.error("❌ ChromaDB not installed. Please install: pip install chromadb")

# Setup Ollama/OpenAI-compatible client
os.environ["OLLAMA_API_BASE"] = "http://localhost:11434"
client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# Available models
AVAILABLE_MODELS = {
    "Llama 2": "llama2:latest",
    "Mistral": "mistral:latest", 
    "Phi-3": "phi3:latest",
    "tinyllama": "tinyllama:latest" 
}

GUARDIAN_MODEL = "mistral:latest"

# === EMBEDDING FUNCTION FOR PROVENANCE CHECK ===
if SENTENCE_TRANSFORMER_AVAILABLE:
    embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    
    def embed_function(sources: list[str]) -> np.ndarray:
        return embedding_model.encode(sources)

# === TOPIC CHECKER FUNCTION ===
def ollama_topic_checker(text: str, topics: list, selected_model: str) -> str:
    topics_str = ", ".join(topics)
    prompt = f"Is the following text related to any of these topics: {topics_str}?\n\nText:\n{text}\n\nAnswer yes or no."
    
    response = client.chat.completions.create(
        model=selected_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=10
    )
    return response.choices[0].message.content.strip().lower()

# === RESPONSE TIME TRACKER ===
class ResponseTimer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        self.start_time = time.time()
    
    def stop(self):
        self.end_time = time.time()
        return self.get_duration()
    
    def get_duration(self):
        if self.start_time and self.end_time:
            return round(self.end_time - self.start_time, 2)
        return 0

# === GUARDRAIL EXECUTION FUNCTION ===
def run_guardrails(selected_guards, answer, question, rag_context, source_content, selected_model):
    """Execute selected guardrails and return results"""
    guardrail_results = {}
    
    for guard_type in selected_guards:
        try:
            if guard_type == "Politeness":
                guard = Guard().use(
                    PolitenessCheck,
                    llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                    on_fail="exception"
                )
                guard.validate(answer)

            elif guard_type == "Responsiveness":
                guard = Guard().use(
                    ResponsivenessCheck,
                    llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                    on_fail="exception"
                )
                guard.validate(answer, metadata={
                    "original_prompt": question,
                    "expected_answer": answer,
                    "temperature": 0,
                    "max_tokens": 1
                })

            elif guard_type == "Hallucination":
                guard = Guard().use(
                    LlmRagEvaluator(
                        eval_llm_prompt_generator=HallucinationPrompt(prompt_name="hallucination_judge_llm"),
                        llm_evaluator_fail_response="hallucinated",
                        llm_evaluator_pass_response="factual",
                        llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                        on_fail="exception",
                        on="prompt"
                    )
                )
                guard.validate(llm_output=answer, metadata={
                    "user_message": question,
                    "context": rag_context,
                    "llm_response": answer
                })

            elif guard_type == "QA Relevance":
                guard = Guard().use(
                    QARelevanceLLMEval,
                    llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                    on_fail="exception"
                )
                guard.validate(answer, metadata={"original_prompt": question})

            elif guard_type == "Summary Quality":
                guard = Guard().use(
                    LLMCritic,
                    metrics={
                        "informative": {"description": "An informative answer captures the main points of the input and is free of irrelevant details.", "threshold": 75},
                        "coherent": {"description": "A coherent answer is logically organized and easy to follow.", "threshold": 50},
                        "concise": {"description": "A concise answer is free of unnecessary repetition and wordiness.", "threshold": 50},
                        "engaging": {"description": "An engaging answer is interesting and holds the reader's attention.", "threshold": 50},
                    },
                    max_score=100,
                    llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                    on_fail="exception"
                )
                guard.validate(answer)

            elif guard_type == "Factuality":
                guard = Guard().use(
                    ResponseEvaluator,
                    llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                    on_fail="exception"
                )
                guard.validate(answer, metadata={
                    "validation_question": question,
                    "pass_on_invalid": False
                })

            elif guard_type == "Unusual Prompt":
                guard = Guard().use(
                    UnusualPrompt,
                    llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                    on="prompt",
                    on_fail="exception"
                )
                guard(
                    lambda prompt: answer,
                    prompt=question,
                    metadata={"temperature": 0.7, "max_tokens": 100}
                )

            # === ENHANCED GUARDRAILS ===
            elif guard_type == "Provenance Check":
                if SENTENCE_TRANSFORMER_AVAILABLE and source_content:
                    guard = Guard().use(
                        ProvenanceLLM,
                        validation_method="sentence",
                        llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                        top_k=3,
                        on_fail="exception",
                    )
                    guard.validate(
                        answer,
                        metadata={
                            "sources": source_content,
                            "embed_function": embed_function,
                            "pass_on_invalid": True
                        },
                    )
                else:
                    st.warning("⚠️ Provenance check skipped: No source content or missing dependencies")
                    continue

            elif guard_type == "Topic Restriction":
                # Get variables from Streamlit session state or global scope
                try:
                    # These variables should be defined in the sidebar configuration
                    if 'valid_topics' in globals() and 'invalid_topics' in globals():
                        guard = Guard().use(
                            RestrictToTopic(
                                valid_topics=valid_topics,
                                invalid_topics=invalid_topics,
                                disable_classifier=True,  
                                disable_llm=False,         
                                llm_callable=lambda text, topics: ollama_topic_checker(text, topics, selected_model),  
                                on_fail="exception"
                            )
                        )
                        guard.validate(answer)
                    else:
                        st.warning("⚠️ Topic restriction skipped: Valid/invalid topics not configured")
                        continue
                except NameError:
                    st.warning("⚠️ Topic restriction skipped: Topics variables not found")
                    continue

            elif guard_type == "Saliency Check":
                try:
                    # Create assets directory if it doesn't exist
                    assets_dir = "assets"
                    if not os.path.exists(assets_dir):
                        os.makedirs(assets_dir)
                    
                    # Check if saliency_threshold is defined
                    if 'saliency_threshold' in globals():
                        guard = Guard().use(
                            SaliencyCheck,
                            assets_dir,
                            llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                            threshold=saliency_threshold,
                            on_fail="exception"
                        )
                        guard.validate(
                            answer,
                            metadata={
                                "temperature": 0,
                                "max_tokens": 1,
                                "pass_on_invalid": False
                            }
                        )
                    else:
                        st.warning("⚠️ Saliency check skipped: Threshold not configured")
                        continue
                except NameError:
                    st.warning("⚠️ Saliency check skipped: Saliency threshold not found")
                    continue

            guardrail_results[guard_type] = "✅ Passed"
            st.success(f"✅ Passed {guard_type} check")
            
        except Exception as e:
            guardrail_results[guard_type] = f"❌ Failed: {str(e)}"
            st.error(f"❌ Failed {guard_type} check: {e}")
    
    return guardrail_results

# === GUARDRAIL EXECUTION FUNCTION ===

def run_guardrails(selected_guards, answer, question, rag_context, source_content, selected_model):
    """Execute selected guardrails and return results"""
    guardrail_results = {}
    
    for guard_type in selected_guards:
        try:
            if guard_type == "Politeness":
                guard = Guard().use(
                    PolitenessCheck,
                    llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                    on_fail="exception"
                )
                guard.validate(answer)

            elif guard_type == "Responsiveness":
                guard = Guard().use(
                    ResponsivenessCheck,
                    llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                    on_fail="exception"
                )
                guard.validate(answer, metadata={
                    "original_prompt": question,
                    "expected_answer": answer,
                    "temperature": 0,
                    "max_tokens": 1
                })

            elif guard_type == "Hallucination":
                guard = Guard().use(
                    LlmRagEvaluator(
                        eval_llm_prompt_generator=HallucinationPrompt(prompt_name="hallucination_judge_llm"),
                        llm_evaluator_fail_response="hallucinated",
                        llm_evaluator_pass_response="factual",
                        llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                        on_fail="exception",
                        on="prompt"
                    )
                )
                guard.validate(llm_output=answer, metadata={
                    "user_message": question,
                    "context": rag_context,
                    "llm_response": answer
                })

            elif guard_type == "QA Relevance":
                guard = Guard().use(
                    QARelevanceLLMEval,
                    llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                    on_fail="exception"
                )
                guard.validate(answer, metadata={"original_prompt": question})

            elif guard_type == "Summary Quality":
                guard = Guard().use(
                    LLMCritic,
                    metrics={
                        "informative": {"description": "An informative answer captures the main points of the input and is free of irrelevant details.", "threshold": 75},
                        "coherent": {"description": "A coherent answer is logically organized and easy to follow.", "threshold": 50},
                        "concise": {"description": "A concise answer is free of unnecessary repetition and wordiness.", "threshold": 50},
                        "engaging": {"description": "An engaging answer is interesting and holds the reader's attention.", "threshold": 50},
                    },
                    max_score=100,
                    llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                    on_fail="exception"
                )
                guard.validate(answer)

            elif guard_type == "Factuality":
                guard = Guard().use(
                    ResponseEvaluator,
                    llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                    on_fail="exception"
                )
                guard.validate(answer, metadata={
                    "validation_question": question,
                    "pass_on_invalid": False
                })

            elif guard_type == "Unusual Prompt":
                guard = Guard().use(
                    UnusualPrompt,
                    llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                    on="prompt",
                    on_fail="exception"
                )
                guard(
                    lambda prompt: answer,
                    prompt=question,
                    metadata={"temperature": 0.7, "max_tokens": 100}
                )

            # === ENHANCED GUARDRAILS ===
            elif guard_type == "Provenance Check":
                if SENTENCE_TRANSFORMER_AVAILABLE and source_content:
                    guard = Guard().use(
                        ProvenanceLLM,
                        validation_method="sentence",
                        llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                        top_k=3,
                        on_fail="exception",
                    )
                    guard.validate(
                        answer,
                        metadata={
                            "sources": source_content,
                            "embed_function": embed_function,
                            "pass_on_invalid": True
                        },
                    )
                else:
                    st.warning("⚠️ Provenance check skipped: No source content or missing dependencies")
                    continue

            elif guard_type == "Topic Restriction":
                # Access variables from session state or global scope
                if 'valid_topics' in globals() and 'invalid_topics' in globals():
                    guard = Guard().use(
                        RestrictToTopic(
                            valid_topics=valid_topics,
                            invalid_topics=invalid_topics,
                            disable_classifier=True,  
                            disable_llm=False,         
                            llm_callable=lambda text, topics: ollama_topic_checker(text, topics, selected_model),  
                            on_fail="exception"
                        )
                    )
                    guard.validate(answer)
                else:
                    st.warning("⚠️ Topic restriction skipped: Topics not configured")
                    continue

            elif guard_type == "Saliency Check":
                # Create assets directory if it doesn't exist
                assets_dir = "assets"
                if not os.path.exists(assets_dir):
                    os.makedirs(assets_dir)
                    
                if 'saliency_threshold' in globals():
                    guard = Guard().use(
                        SaliencyCheck,
                        assets_dir,
                        llm_callable=f"ollama_chat/{GUARDIAN_MODEL}",
                        threshold=saliency_threshold,
                        on_fail="exception"
                    )
                    guard.validate(
                        answer,
                        metadata={
                            "temperature": 0,
                            "max_tokens": 1,
                            "pass_on_invalid": False
                        }
                    )
                else:
                    st.warning("⚠️ Saliency check skipped: Threshold not configured")
                    continue

            guardrail_results[guard_type] = "✅ Passed"
            st.success(f"✅ Passed {guard_type} check")
            
        except Exception as e:
            guardrail_results[guard_type] = f"❌ Failed: {str(e)}"
            st.error(f"❌ Failed {guard_type} check: {e}")
    
    return guardrail_results

# === CHROMADB SETUP ===
@st.cache_resource
def setup_chromadb():
    """Initialize ChromaDB client and collection"""
    if not CHROMADB_AVAILABLE:
        return None, None
    
    try:
        # Create persistent ChromaDB client
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection
        collection_name = f"pdf_documents_{datetime.now().strftime('%Y%m%d')}"
        try:
            collection = chroma_client.get_collection(collection_name)
        except:
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "PDF document embeddings"}
            )
        
        return chroma_client, collection
    except Exception as e:
        st.error(f"ChromaDB setup failed: {e}")
        return None, None

# Streamlit UI setup
st.set_page_config(
    page_title="Responsible AI",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Responsible AI")

# === SIDEBAR CONFIGURATION ===
st.sidebar.header("⚙️ Configuration")

# Model Selection
selected_model_name = st.sidebar.selectbox(
    "🧠 Select Generation Model:",
    list(AVAILABLE_MODELS.keys()),
    index=1  # Default to Mistral
)
selected_model = AVAILABLE_MODELS[selected_model_name]

# Chat Mode Selection
chat_mode = st.sidebar.radio(
    "💬 Chat Mode:",
    ["📄 Document-based Chat", "🗨️ General Chat"],
    index=0
)

# Enhanced Guardrail selection UI
all_guardrails = [
    "Politeness", "Responsiveness", "Hallucination",
    "QA Relevance", "Summary Quality", "Factuality", 
    "Unusual Prompt", "Provenance Check", "Topic Restriction", 
    "Saliency Check"
]

if not SENTENCE_TRANSFORMER_AVAILABLE:
    all_guardrails.remove("Provenance Check")

selected_guards = st.sidebar.multiselect(
    "🛡️ Select Guardrails:",
    all_guardrails
)

# Additional configuration for guardrails
st.sidebar.header("🔧 Guardrail Settings")

# Topic Restriction Configuration
if "Topic Restriction" in selected_guards:
    st.sidebar.subheader("Topic Restriction")
    valid_topics = st.sidebar.text_input(
        "Valid Topics (comma-separated)", 
        value="education,science,technology,general"
    ).split(",")
    invalid_topics = st.sidebar.text_input(
        "Invalid Topics (comma-separated)", 
        value="politics,religion,adult"
    ).split(",")
    valid_topics = [topic.strip() for topic in valid_topics if topic.strip()]
    invalid_topics = [topic.strip() for topic in invalid_topics if topic.strip()]

# Saliency Check Configuration
if "Saliency Check" in selected_guards:
    st.sidebar.subheader("Saliency Check")
    saliency_threshold = st.sidebar.slider(
        "Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.01
    )

# Initialize ChromaDB
chroma_client, chroma_collection = setup_chromadb()

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None

# === DOCUMENT-BASED CHAT MODE ===
if chat_mode == "📄 Document-based Chat":
    st.header("📄 Document-based Chat")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_pdf_path = tmp_file.name

        with st.spinner("🔄 Processing PDF and storing in ChromaDB..."):
            try:
                # Load and split PDF
                loader = PyPDFLoader(temp_pdf_path)
                pages = loader.load_and_split()
                
                # Text splitting for better chunking
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                splits = text_splitter.split_documents(pages)
                
                # Setup embeddings
                embeddings = OllamaEmbeddings(model='nomic-embed-text:v1.5')
                
                # Create vector store with ChromaDB
                if CHROMADB_AVAILABLE and chroma_collection:
                    # Generate unique IDs for documents
                    doc_ids = [str(uuid.uuid4()) for _ in splits]
                    
                    # Store in ChromaDB
                    vector_store = Chroma.from_documents(
                        documents=splits,
                        embedding=embeddings,
                        client=chroma_client,
                        collection_name=chroma_collection.name,
                        ids=doc_ids
                    )
                    st.session_state.vector_store = vector_store
                    st.session_state.retriever = vector_store.as_retriever(search_kwargs={"k": 5})
                    
                    st.success(f"✅ PDF processed! {len(splits)} chunks stored in ChromaDB")
                else:
                    st.error("❌ ChromaDB not available. Using fallback in-memory storage.")
                    # Fallback to in-memory storage
                    from langchain_community.vectorstores import DocArrayInMemorySearch
                    vector_store = DocArrayInMemorySearch.from_documents(splits, embedding=embeddings)
                    st.session_state.vector_store = vector_store
                    st.session_state.retriever = vector_store.as_retriever()
                    st.success(f"✅ PDF processed! {len(splits)} chunks stored in memory")
                
            except Exception as e:
                st.error(f"❌ Error processing PDF: {e}")
            finally:
                os.remove(temp_pdf_path)
    
    # Chat interface for document-based mode
    if st.session_state.retriever:
        st.subheader("💬 Ask questions about your document")
        
        # Display chat history
        for i, (q, a, response_time, guardrail_results) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {q[:50]}..." if len(q) > 50 else f"Q{i+1}: {q}", expanded=False):
                st.markdown(f"**Question:** {q}")
                st.markdown(f"**Answer:** {a}")
                st.markdown(f"**Response Time:** {response_time}s")
                if guardrail_results:
                    st.markdown("**Guardrail Results:**")
                    for guard, result in guardrail_results.items():
                        if "Passed" in result:
                            st.markdown(f"- ✅ {guard}: {result}")
                        else:
                            st.markdown(f"- ❌ {guard}: {result}")
        
        # Question input
        question = st.text_input("Ask a question about the PDF:", key="doc_question")
        
        if st.button("🚀 Ask Question", key="ask_doc"):
            if question:
                timer = ResponseTimer()
                timer.start()
                
                with st.spinner("🤔 Thinking..."):
                    try:
                        # RAG pipeline setup
                        template = """
                        You are a helpful AI assistant that answers questions based on the provided document context.
                        Use only the context provided to answer the question. If you don't know the answer or
                        can't find it in the context, say so clearly.

                        Context: {context}

                        Question: {question}
                        
                        Answer:
                        """
                        prompt = PromptTemplate.from_template(template)

                        def format_docs(docs):
                            return "\n\n".join(doc.page_content for doc in docs)

                        llm = Ollama(model=selected_model)
                        chain = (
                            {
                                'context': st.session_state.retriever | format_docs,
                                'question': RunnablePassthrough(),
                            }
                            | prompt
                            | llm
                            | StrOutputParser()
                        )
                        
                        # Generate answer
                        context_docs = st.session_state.retriever.get_relevant_documents(question)
                        rag_context = format_docs(context_docs)
                        answer = chain.invoke(question)
                        
                        response_time = timer.stop()
                        
                        # Display answer
                        st.markdown("### ✅ Answer")
                        st.write(answer)
                        st.markdown(f"**⏱️ Response Time:** {response_time}s")
                        st.markdown(f"**🧠 Model Used:** {selected_model_name}")
                        
                        # Run guardrails
                        guardrail_results = {}
                        source_content = [doc.page_content for doc in context_docs]
                        
                        if selected_guards:
                            st.markdown("### 🛡️ Guardrail Results")
                            guardrail_results = run_guardrails(
                                selected_guards, answer, question, rag_context, 
                                source_content, selected_model
                            )
                        
                        # Store in chat history
                        st.session_state.chat_history.append((question, answer, response_time, guardrail_results))
                        
                    except Exception as e:
                        st.error(f"❌ Error generating answer: {e}")

# === GENERAL CHAT MODE ===
elif chat_mode == "🗨️ General Chat":
    st.header("🗨️ General Chat")
    st.write("Chat with the AI without any document context.")
    
    # Display chat history
    for i, (q, a, response_time, guardrail_results) in enumerate(st.session_state.chat_history):
        with st.expander(f"Chat {i+1}: {q[:50]}..." if len(q) > 50 else f"Chat {i+1}: {q}", expanded=False):
            st.markdown(f"**You:** {q}")
            st.markdown(f"**AI:** {a}")
            st.markdown(f"**Response Time:** {response_time}s")
            if guardrail_results:
                st.markdown("**Guardrail Results:**")
                for guard, result in guardrail_results.items():
                    if "Passed" in result:
                        st.markdown(f"- ✅ {guard}: {result}")
                    else:
                        st.markdown(f"- ❌ {guard}: {result}")
    
    # Chat input
    user_message = st.text_input("Type your message:", key="general_chat")
    
    if st.button("💬 Send Message", key="send_general"):
        if user_message:
            timer = ResponseTimer()
            timer.start()
            
            with st.spinner("🤔 Thinking..."):
                try:
                    # Direct LLM call without RAG
                    llm = Ollama(model=selected_model)
                    
                    general_prompt = """
                    You are a helpful AI assistant. Answer the user's question or engage in conversation.
                    Be helpful, informative, and conversational.
                    
                    User: {question}
                    
                    Assistant:
                    """
                    
                    prompt = PromptTemplate.from_template(general_prompt)
                    chain = prompt | llm | StrOutputParser()
                    
                    answer = chain.invoke({"question": user_message})
                    response_time = timer.stop()
                    
                    # Display answer
                    st.markdown("### 🤖 AI Response")
                    st.write(answer)
                    st.markdown(f"**⏱️ Response Time:** {response_time}s")
                    st.markdown(f"**🧠 Model Used:** {selected_model_name}")
                    
                    # Run guardrails (without document context)
                    guardrail_results = {}
                    if selected_guards:
                        st.markdown("### 🛡️ Guardrail Results")
                        guardrail_results = run_guardrails(
                            selected_guards, answer, user_message, "", 
                            [], selected_model
                        )
                    
                    # Store in chat history
                    st.session_state.chat_history.append((user_message, answer, response_time, guardrail_results))
                    
                except Exception as e:
                    st.error(f"❌ Error generating response: {e}")

# === FOOTER INFORMATION ===
if st.session_state.chat_history:
    total_questions = len(st.session_state.chat_history)
    avg_response_time = sum([entry[2] for entry in st.session_state.chat_history]) / total_questions

# Clear chat history button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()