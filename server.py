from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid, time
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import os
from langchain.tools import tool
from pinecone import Pinecone, ServerlessSpec
from langchain_core.prompts import PromptTemplate
import logging
import json
app = FastAPI()

# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# llm = ChatOpenAI(model_name= os.getenv("MODEL_NAME","gpt-4o-mini"), temperature=0.0, openai_api_key=os.getenv("OPENAI_API_KEY"))
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.environ.get("PINECONE_INDEX", "patient-record")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENVIRONMENT", "us-west1-gcp")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")

# initialize model safely
model = None
try:
    model = init_chat_model(MODEL_NAME)
    logger.info("Chat model initialized: %s", MODEL_NAME)
except Exception as e:
    logger.exception("Failed to initialize chat model: %s", e)
    model = None

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is not set. LLM calls will fail without it.")
if not PINECONE_API_KEY:
    logger.warning("PINECONE_API_KEY is not set. Pinecone operations will fail without it.")

# initialize Pinecone and vector store safely
pc = None
dense_index = None
vector_store = None
try:
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        if pc.has_index(INDEX_NAME):
            dense_index = pc.Index(INDEX_NAME)
            embeddings = OpenAIEmbeddings()
            vector_store = PineconeVectorStore(index=dense_index, embedding=embeddings)
            logger.info("Connected to Pinecone index: %s", INDEX_NAME)
        else:
            logger.warning("Pinecone index %s not found. Create the index first.", INDEX_NAME)
    else:
        logger.debug("Skipping Pinecone init because PINECONE_API_KEY is missing.")
except Exception as e:
    logger.exception("Failed to initialize Pinecone client/index: %s", e)
    pc = None
    dense_index = None
    vector_store = None


class AskReq(BaseModel):
    patient_id: str
    question: str
    user_id: str  # doctor id for audit

@app.post("/ask")
def ask(req: AskReq):
   
    trace_id = str(uuid.uuid4())
    # start = time.time()
    logger.info("Received /ask trace=%s user=%s patient=%s", trace_id, req.user_id, req.patient_id)
    try:
        answer = answer_patient_question(req.patient_id, req.question)
        ai_only = _extract_ai_content(answer)
        logger.info("ask completed trace=%s", trace_id)
        return {"trace_id": trace_id, "result": answer}
    except Exception as e:
        logger.exception("ask failed trace=%s: %s", trace_id, e)
        raise HTTPException(status_code=500, detail=f"Internal error (trace_id={trace_id}): {e}")

def _extract_ai_content(obj) -> str:
    """Return a best-effort plain string with only the AI/content text from many possible response shapes."""
    try:
        # plain string
        if isinstance(obj, str):
            return obj

        # common LangChain/LLM shape: {'answer': '...'} or {'content': '...'}
        if isinstance(obj, dict):
            for key in ("answer", "content", "text", "message", "output"):
                if key in obj and isinstance(obj[key], str):
                    return obj[key]

            # Chat generation-like shape
            if "generations" in obj:
                try:
                    return obj["generations"][0][0].get("text") or obj["generations"][0][0].get("message") or str(obj)
                except Exception:
                    pass

            # tools / artifact wrapper
            if "artifact" in obj and isinstance(obj["artifact"], dict):
                # try common nested locations
                if "docs" in obj["artifact"]:
                    return "\n\n".join(d.get("content", "") or str(d.get("metadata","")) for d in obj["artifact"]["docs"])

            # messages list
            if "messages" in obj and isinstance(obj["messages"], (list, tuple)):
                msgs = obj["messages"]
                # walk backwards to pick last assistant/tool message
                for m in reversed(msgs):
                    # m may be dict or tuple (role, message_obj) or LangChain message object
                    if isinstance(m, dict):
                        # dict with role/content
                        if m.get("role") in ("assistant", "ai", "tool") and (m.get("content") or m.get("text")):
                            return m.get("content") or m.get("text")
                        # sometimes nested
                        if "content" in m and isinstance(m["content"], str):
                            return m["content"]
                    elif isinstance(m, (list, tuple)) and len(m) >= 2:
                        candidate = m[1]
                        # candidate may be dict or object with .content
                        if isinstance(candidate, dict) and candidate.get("content"):
                            return candidate.get("content")
                        if hasattr(candidate, "content"):
                            return getattr(candidate, "content")
                        if hasattr(candidate, "text"):
                            return getattr(candidate, "text")
                # fallback: join all messages' text
                parts = []
                for m in msgs:
                    if isinstance(m, dict):
                        parts.append(m.get("content") or m.get("text") or str(m))
                    elif isinstance(m, (list, tuple)) and len(m) >= 2:
                        c = m[1]
                        parts.append(getattr(c, "content", None) or getattr(c, "text", None) or str(c))
                    else:
                        parts.append(str(m))
                return "\n\n".join(p for p in parts if p)

        # list of possible items
        if isinstance(obj, (list, tuple)):
            pieces = []
            for it in obj:
                pieces.append(_extract_ai_content(it))
            return "\n\n".join(p for p in pieces if p)

        # fallback: try to stringify
        return str(obj)
    except Exception:
        return str(obj)

try:
    prompt_text = open("clinical_prompt.txt", "r", encoding="utf-8").read()
    logger.info("Loaded clinical_prompt.txt")
except Exception as e:
    logger.exception("Failed to load clinical_prompt.txt: %s", e)
    # fallback to a simple string prompt
    prompt_text = "Use the following context: {context_chunks}\nQuestion: {question}\nAnswer:"


# Construct a tool for retrieving context (handle missing vector_store)
@tool(response_format="content_and_artifact")
def retrieve_context(patient_id: str, query: str):
    """Retrieve information to help answer a query.

    Returns a JSON-serializable dict with keys:
      - content: human-readable concatenated context string
      - artifact: structured JSON object with docs list
    This avoids returning non-serializable objects (Document instances),
    which causes errors like "src property must be a valid json object".
    """
    if vector_store is None:
        logger.error("Vector store not initialized; cannot retrieve context.")
        return {"content": "", "artifact": {"docs": []}}
    try:
        retrieved_docs = vector_store.similarity_search(query, k=2)

        docs_serialized = []
        for idx, doc in enumerate(retrieved_docs):
            # Ensure metadata is a plain dict and content is plain string
            meta = doc.metadata if isinstance(doc.metadata, dict) else {}
            docs_serialized.append({
                "id": meta.get("id") or f"doc_{idx}",
                "metadata": meta,
                "content": doc.page_content
            })

        serialized = "\n\n".join(
            (f"Source: {d['metadata']}\nContent: {d['content']}")
            for d in docs_serialized
        )

        # return a plain JSON-serializable object
        return {"content": serialized, "artifact": {"docs": docs_serialized}}
    except Exception as e:
        logger.exception("retrieve_context failed: %s", e)
        # raise a clear error so calling code/logs show details
        raise

tools = []
agent = None
# create agent only if model is available
if model is not None:
    try:
        tools = []
        if vector_store is not None:
            tools.append(retrieve_context)
        else:
            logger.info("Vector store missing; agent will be created without retrieval tool.")
        agent = create_agent(model, tools=tools, system_prompt=prompt_text)
        logger.info("Agent created with %d tools", len(tools))
    except Exception as e:
        logger.exception("Failed to create agent: %s", e)
        agent = None
else:
    logger.warning("Agent not created because chat model failed to initialize.")




# Build QA chain that uses our prompt


def answer_patient_question(patient_id: str, question: str):
    if agent is None and vector_store is None and dense_index is None:
        logger.error("No backend available to answer question (agent, vector_store, dense_index all None)")
        return {"error": "no_backend", "message": "No LLM agent or vector store configured. Set OPENAI_API_KEY or PINECONE_API_KEY."}

    # Prefer using the agent if available
    if agent is not None:
        try:
            logger.debug("Invoking agent for patient=%s question=%s", patient_id, question)
            last_step = None
            for step in agent.stream(
                {"messages": [{"role": "user", "content": f"{question} : of this patients {patient_id}"}]},
                stream_mode="values",
            ):
                # stream values may be printed or processed here
                try:
                    # try to pretty_print if present
                    if isinstance(step, dict) and "messages" in step:
                        step["messages"][-1].pretty_print()
                except Exception:
                    pass
                last_step = step
            logger.debug("Agent stream completed for patient=%s", patient_id)
            return last_step.get("messages") if isinstance(last_step, dict) else last_step
        except Exception as e:
            logger.exception("Agent failed; falling back to vector search: %s", e)
            # fall through to vector search fallback

    # Fallback: use Pinecone search if available
    # if dense_index is None:
    #     logger.error("No available method to answer question: agent and dense_index both unavailable")
    #     raise RuntimeError("No available method to answer question (agent and index unavailable)")

    # try:
    #     res = dense_index.search(
    #         namespace="namespace1",
    #         query={
    #             "inputs": {"text": question},
    #             "top_k": 2,
    #             "filter": {"patient_id": patient_id},
    #         },
    #         fields=["patient_id", "chunk_text"]
    #     )
    #     logger.debug("Pinecone search result: %s", res)
    # except Exception as e:
    #     logger.exception("Pinecone search failed: %s", e)
    #     raise RuntimeError("Search failed") from e

    # # convert results into Documents (LangChain style)
    # docs = []
    # for match in res.get("matches", []) or res.get("results", []):
    #     try:
    #         text = match.get("metadata", {}).get("chunk_text") or match.get("chunk_text") or match.get("text")
    #         meta = match.get("metadata", {}) or {}
    #         meta.update({
    #             "score": match.get("score") or match.get("distance"),
    #             "index_id": match.get("id")
    #         })
    #         if text:
    #             docs.append(Document(page_content=text, metadata=meta))
    #     except Exception as e:
    #         logger.exception("Failed to parse match entry: %s", e)

    # if not docs:
    #     logger.info("No documents found for patient=%s", patient_id)
    #     return {"error": "no_relevant_docs", "message": "No documents found for this patient."}

    # # build prompt and call LLM if available
    # context_concat = "\n\n---\n\n".join([f"doc_id:{d.metadata.get('doc_id','')} date:{d.metadata.get('date','')} text:{d.page_content}" for d in docs])
    # prompt_text = prompt.format(patient_id=patient_id, question=question, context_chunks=context_concat)

    # # call ChatOpenAI directly if available
    # try:
    #     client = None
    #     try:
    #         client = ChatOpenAI(model_name=MODEL_NAME, temperature=0.0, openai_api_key=OPENAI_API_KEY)
    #     except Exception as e:
    #         logger.exception("Failed to create ChatOpenAI client for request: %s", e)
    #         client = None

    #     if client is None:
    #         logger.error("LLM client not available to generate final answer.")
    #         return {"error": "llm_unavailable", "context": context_concat}

    #     resp = client(messages=[HumanMessage(content=prompt_text)])
    #     logger.debug("LLM response object: %s", getattr(resp, "__dict__", str(resp)))
    #     # Best-effort extraction of text
    #     try:
    #         final_text = resp.generations[0][0].text
    #     except Exception:
    #         final_text = getattr(resp, "content", None) or str(resp)
    #     return {"answer": final_text, "context": context_concat}
    # except Exception as e:
    #     logger.exception("Final LLM generation failed: %s", e)
    #     # return context so caller can still inspect results
    #     return {"error": "generation_failed", "message": str(e), "context": context_concat}
# ...existing code...