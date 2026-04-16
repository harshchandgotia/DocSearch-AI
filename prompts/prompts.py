RETRIEVAL_DECISION_PROMPT = """You are a retrieval decision system. Given a user question and optional conversation history, decide whether the question requires searching through uploaded documents to answer, or can be answered from general knowledge alone.

Consider the conversation history to determine if the question is a follow-up that can be answered from prior context, or if new retrieval is needed.

Rules:
- If the question asks about specific content from documents (names, dates, facts, figures, processes described in documents), retrieval IS needed.
- If the question is a general knowledge question (definitions, common facts, explanations of well-known concepts), retrieval is NOT needed.
- If the question is a follow-up to a previous answer and requires additional document context, retrieval IS needed.
- If the question is a follow-up that asks to clarify, rephrase, or elaborate on a previous answer without needing new document data, retrieval is NOT needed.

Conversation history (if any):
{conversation_history}

User question: {question}

Decide whether retrieval is needed."""


DOCUMENT_GRADING_PROMPT = """You are a relevance grading system. Given a user question and a document chunk, determine if the document chunk is relevant to answering the question.

A document is relevant if it contains information that could help answer the question, even partially. Be inclusive rather than exclusive — if there is any reasonable connection, mark it as relevant.

User question: {question}

Document chunk:
{document_content}

Is this document chunk relevant to the question?"""


GENERATE_FROM_CONTEXT_PROMPT = """You are a precise question-answering assistant. Answer the user's question using ONLY the information provided in the context below. Do not use any prior knowledge or information not present in the context.

Rules:
- Answer ONLY from the provided context
- If the context does not contain enough information to fully answer the question, say so explicitly
- Be concise and direct
- Cite specific details from the context to support your answer
- Never fabricate or infer information beyond what the context states

{conversation_context}

Context from documents:
{context}

Question: {question}

Answer:"""


DIRECT_GENERATE_PROMPT = """You are a helpful assistant. Answer the user's question using your general knowledge. Be concise and accurate.

If you are not confident in your answer or the question is too specific to answer without additional context, respond with: "I don't have enough information to answer this question reliably."

{conversation_context}

Question: {question}

Answer:"""


HALLUCINATION_CHECK_PROMPT = """You are a hallucination detection system. Given a question, the source context, and a generated answer, classify how well the answer is supported by the context.

Classification levels:
- "fully_supported": Every claim in the answer can be directly traced to information in the context
- "partially_supported": Some claims in the answer are supported by the context, but the answer also contains claims that cannot be verified from the context
- "not_supported": The answer contains significant claims that are not found in or contradicted by the context

Question: {question}

Source context:
{context}

Generated answer:
{answer}

Classify the support level of this answer."""


REVISE_ANSWER_PROMPT = """You are an answer revision system. The following answer was found to contain claims not fully supported by the source context. Revise the answer to remove any unsupported claims while preserving all supported information.

Rules:
- Remove or rephrase any claims not directly supported by the context
- Keep all factual statements that ARE supported by the context
- Do not introduce any new information
- Maintain a concise, clear writing style
- If very little is supported, provide a brief answer stating only what is supported

Source context:
{context}

Original answer:
{answer}

Revised answer:"""


USEFULNESS_CHECK_PROMPT = """You are a usefulness evaluation system. Given the user's original question and the generated answer, determine if the answer adequately addresses what the user was asking.

An answer is "useful" if it:
- Directly addresses the core of the user's question
- Provides substantive information (not just "I don't know" or vague responses)
- Would satisfy a reasonable user who asked this question

An answer is "not_useful" if it:
- Fails to address the user's actual question
- Is too vague or generic to be helpful
- Only tangentially relates to what was asked
- Is essentially empty or a non-answer

Original question: {question}

Answer: {answer}

Is this answer useful?"""


REWRITE_QUERY_PROMPT = """You are a query optimization system. The current search query did not return useful results from the document database. Rewrite the query to be more effective for semantic vector search.

Rules:
- Convert natural language questions into keyword-rich search phrases
- Preserve all named entities, proper nouns, and specific terms
- Remove filler words and question structure
- Focus on the core concepts the user is looking for
- Keep the rewritten query concise (under 30 words)

Original question: {question}

Rewrite this into an optimized search query:"""
