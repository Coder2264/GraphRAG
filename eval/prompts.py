"""
All prompt strings and builder functions for the eval pipeline.

Three categories:
  1. Document & QA generation  — prompts to paste into Claude/ChatGPT/Gemini web UIs
  2. QA validation             — prompts used by validate_qa.py (Gemini API)
  3. LLM judge                 — prompts used by eval_rag.py (Gemini API)

Keeping every prompt here makes iteration and auditing easy.
"""

from __future__ import annotations

# ===========================================================================
# 1. DOCUMENT & QA GENERATION  (paste into web UIs)
# ===========================================================================

# --- Claude Project Instructions (set once in Project Settings) ---

CLAUDE_PROJECT_INSTRUCTIONS = """\
You are a synthetic document generator for a GraphRAG multi-hop reasoning benchmark.
Your job is to create realistic fictional documents where key facts are DELIBERATELY
SCATTERED across different sections so that simple keyword or vector-similarity
retrieval cannot answer multi-hop questions.

OUTPUT FORMAT: Always respond with a single valid JSON object only.
No prose, no markdown fences, no explanation outside the JSON.

CRITICAL ENTITY SCATTERING RULE:
- If entity A connects to entity B, put that fact in one section.
- If entity B connects to entity C, put THAT fact in a DIFFERENT section (at least 2 away).
- Never place two consecutive links of the same answer chain in the same section.
- This is the entire point: defeating top-K chunk retrieval.\
"""


def doc_generation_prompt(domain: str, topic: str, doc_index: str) -> str:
    """
    Per-document generation prompt.  Paste this into the web UI once per document,
    replacing DOMAIN / TOPIC / DOCUMENT INDEX with the appropriate values.
    """
    return f"""\
Generate a synthetic document for GraphRAG benchmark testing.

DOMAIN: {domain}
TOPIC: {topic}
DOCUMENT INDEX: {doc_index}

Requirements:
1. Exactly 5 sections, each 450-550 words, labeled "## Section N: [Descriptive Title]"
2. 10-12 named entities total (mix: People, Organizations, Locations, Products, Awards)
3. CRITICAL — scatter connected facts across NON-ADJACENT sections:
   - Section 1: Introduce Company X and its founder Person A
                (do NOT mention where Person A studied)
   - Section 2: Describe Person A's early career at University Y
                (do NOT mention who else trained there)
   - Section 3: Introduce Person B (lead specialist), note their collaboration with Professor C
                (do NOT mention Professor C's institution)
   - Section 4: Describe Professor C's lab at Institute Z
                (do NOT mention who funds Institute Z)
   - Section 5: Name Institute Z's primary funder Organization W and its director Person D
   Adapt this pattern to your domain — the STRUCTURE (one hop per section) is mandatory.
4. Include 1-2 plausible distractor sentences near each entity name that are
   semantically similar to the answer but factually wrong — to confuse keyword search.
5. All names, organizations, and events must be entirely fictional.

After the document, generate exactly 6 multi-hop QA pairs.
Each pair MUST require information from at least 2 different sections.
At least 2 pairs should require 3+ hops.

RESPOND WITH THIS EXACT JSON STRUCTURE (no other text):
{{
  "doc_index": "{doc_index}",
  "domain": "{domain}",
  "topic": "exact topic string",
  "document_text": "## Section 1: [Title]\\n\\n[450-550 words]\\n\\n## Section 2: [Title]\\n\\n[450-550 words]\\n\\n## Section 3: [Title]\\n\\n[450-550 words]\\n\\n## Section 4: [Title]\\n\\n[450-550 words]\\n\\n## Section 5: [Title]\\n\\n[450-550 words]",
  "entities": [
    {{"name": "ExampleCorp", "type": "Organization", "introduced_in_section": 1, "connected_to": ["Alice Marsh"]}}
  ],
  "qa_pairs": [
    {{
      "id": "doc_{doc_index}_q1",
      "question": "Which organization funds the lab of the scientist who co-authored with ExampleCorp's lead researcher?",
      "answer": "Organization W",
      "hop_count": 4,
      "entity_chain": ["ExampleCorp", "Person B", "Professor C", "Institute Z", "Organization W"],
      "sections_involved": [1, 3, 4, 5],
      "answer_justification": "ExampleCorp's lead researcher is Person B (section 3); Person B co-authored with Professor C (section 3); Professor C works at Institute Z (section 4); Institute Z is funded by Organization W (section 5)"
    }}
  ]
}}\
"""


# --- ChatGPT / Gemini system context (paste once per session) ---

CHATGPT_SYSTEM_CONTEXT = """\
You are a synthetic document generator for a GraphRAG multi-hop reasoning benchmark.
CRITICAL RULE: Facts forming multi-hop answer chains MUST be in DIFFERENT sections —
never co-located. This defeats simple keyword/vector retrieval and proves multi-hop
reasoning is required.
Output: valid JSON only, no prose outside the JSON object.\
"""


GEMINI_CONTEXT = """\
You are generating synthetic multi-hop test documents for a GraphRAG research benchmark.
KEY REQUIREMENT: Each document must deliberately scatter facts across sections so
that no single 512-character text chunk contains all information needed to answer
the question. Answer chains must cross section boundaries.
Output format: JSON only.\
"""


# ===========================================================================
# 2. QA VALIDATION PROMPTS  (used by validate_qa.py)
# ===========================================================================

VALIDATION_SYSTEM = """\
You are a strict quality evaluator for a multi-hop reasoning benchmark.
You assess whether QA pairs genuinely require multi-hop reasoning and cannot
be trivially answered by simple retrieval.
Always respond with a JSON object containing "score" (float 0.0-1.0) and "reason" (string).\
"""


def consistency_prompt(doc_text: str, question: str, answer: str) -> str:
    """Is the stated answer actually correct per the document?"""
    return f"""\
DOCUMENT:
{doc_text}

QUESTION: {question}
STATED ANSWER: {answer}

Is the stated answer factually correct and fully supported by the document text?

Score 0.0-1.0:
- 1.0: Answer is exactly correct and directly supported
- 0.8: Answer is correct but slightly incomplete or differently phrased
- 0.5: Answer is partially correct (right direction, wrong detail)
- 0.2: Answer contradicts the document
- 0.0: Answer is completely wrong or not in the document

Respond with JSON: {{"score": 0.0, "reason": "..."}}\
"""


def multi_hop_necessity_prompt(doc_text: str, question: str, chunk_size: int = 512) -> str:
    """Would answering from a single chunk be impossible?"""
    return f"""\
DOCUMENT:
{doc_text}

QUESTION: {question}
CHUNK SIZE: {chunk_size} characters

Mentally split the document into chunks of {chunk_size} characters.
Can ANY single chunk contain ALL the information needed to fully answer the question?

Score 0.0-1.0:
- 1.0: Absolutely impossible from any single chunk — answer requires connecting facts from multiple sections
- 0.8: Very unlikely from a single chunk — key facts are well-separated
- 0.5: Possible but unlikely — facts are somewhat separated
- 0.2: A single chunk probably contains enough to answer
- 0.0: A single chunk definitely contains the full answer

Respond with JSON: {{"score": 0.0, "reason": "..."}}\
"""


def specificity_prompt(question: str, answer: str) -> str:
    """Is the question unambiguous with exactly one correct answer?"""
    return f"""\
QUESTION: {question}
STATED ANSWER: {answer}

Evaluate the question's specificity and answerability:

Score 0.0-1.0:
- 1.0: Question is perfectly specific, unambiguous, has exactly one correct answer
- 0.8: Question is clear but answer could have minor valid variations
- 0.5: Question is somewhat vague or could have multiple reasonable answers
- 0.2: Question is ambiguous or overly broad
- 0.0: Question is unanswerable as stated or completely open-ended

Respond with JSON: {{"score": 0.0, "reason": "..."}}\
"""


def difficulty_prompt(doc_text: str, question: str) -> str:
    """Would standard top-5 RAG retrieval likely fail?"""
    return f"""\
DOCUMENT:
{doc_text}

QUESTION: {question}

Imagine a standard RAG system: embed the question, retrieve the 5 most semantically
similar 512-character chunks from this document, then try to answer.

Score 0.0-1.0:
- 1.0: Top-5 retrieval would definitely fail — the needed facts are in semantically
       distant sections that wouldn't score high for the question
- 0.8: Top-5 retrieval would likely fail — at least one critical hop is in a
       section with low semantic similarity to the question
- 0.5: Top-5 retrieval might succeed but would be unreliable
- 0.2: Top-5 retrieval would probably succeed
- 0.0: Top-5 retrieval would easily succeed (answer is in the most similar chunk)

Respond with JSON: {{"score": 0.0, "reason": "..."}}\
"""


# ===========================================================================
# 3. LLM JUDGE PROMPTS  (used by eval_rag.py)
# ===========================================================================

RETRIEVAL_JUDGE_SYSTEM = """\
You are evaluating whether a retrieved context contains the information needed
to answer a multi-hop question. Be strict: partial information is not full credit.
Always respond with JSON: {"score": N, "reason": "..."}  where N is 0-10.\
"""


def retrieval_judge_prompt(
    question: str,
    gold_answer: str,
    context: str,
    answer_justification: str,
) -> str:
    """Score whether the retrieved context covers the required answer chain."""
    return f"""\
QUESTION: {question}
GOLD ANSWER: {gold_answer}
ANSWER CHAIN: {answer_justification}

RETRIEVED CONTEXT:
{context}

Score the retrieved context 0-10:
- 10: Context contains ALL facts in the answer chain — a reader could construct the full answer
- 7-9: Context contains most facts; only 1 minor gap
- 4-6: Context contains some relevant facts but is missing at least one critical hop
- 1-3: Context has barely relevant fragments but cannot support the answer
- 0:   Context is empty, off-topic, or completely irrelevant

Respond with JSON: {{"score": N, "reason": "short explanation"}}\
"""


ANSWER_JUDGE_SYSTEM = """\
You are evaluating the factual correctness and completeness of an answer
to a multi-hop reasoning question. Compare against the gold answer.
Always respond with JSON: {"score": N, "reason": "..."}  where N is 0-10.\
"""


def answer_judge_prompt(
    question: str,
    gold_answer: str,
    system_answer: str,
) -> str:
    """Score the system's answer against the gold answer."""
    return f"""\
QUESTION: {question}
GOLD ANSWER: {gold_answer}
SYSTEM ANSWER: {system_answer}

Score the system answer 0-10:
- 10: Fully correct and complete — matches gold answer (minor phrasing differences OK)
- 7-9: Mostly correct; minor omissions or slightly different but equivalent phrasing
- 4-6: Partially correct — right general direction but missing key entity or detail
- 1-3: Mostly wrong; perhaps mentions one relevant entity but core answer is incorrect
- 0:   Wrong, hallucinated, or "I don't have enough information"

Respond with JSON: {{"score": N, "reason": "short explanation"}}\
"""
