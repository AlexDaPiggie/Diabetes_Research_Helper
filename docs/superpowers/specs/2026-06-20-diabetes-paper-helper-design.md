# Diabetes Paper Helper Design

## Overview

Diabetes Paper Helper is a patient-facing AI web app that helps diabetic patients and caregivers understand diabetes research papers. The app translates research into approachable explanations without oversimplifying the science or giving personal medical advice.

The project has two goals:

- Provide a genuinely useful reading assistant for patients and caregivers.
- Demonstrate serious AI engineering through RAG, document ETL, guardrails, evaluation, model comparison, and cost-aware design.

The user interface is only for patients and caregivers. Technical testing, evaluation, and model comparison happen under the hood through backend tests, notebooks, logs, reports, and README documentation.

## Target Users

Primary users:

- Diabetic patients reading diabetes research directly.
- Caregivers or family members trying to understand research papers to better support a patient.

The app should explain technical terms and research findings clearly, but it should not flatten the science into vague advice. It should preserve study details, uncertainty, limitations, and source boundaries.

## Product Scope

### User-Facing App

The user-facing app includes:

- Paper intake workspace.
- Recent papers library using local/demo persistence.
- Structured paper summary.
- Technical term explanations.
- Citation-grounded paper chat.
- General diabetes education when a question is not answered by the paper.
- Medical advice guardrails.
- Clinician-question suggestions.

The user-facing app does not include an evaluation dashboard, model comparison UI, or internal technical diagnostics.

### Paper Inputs

V1 supports:

- PDF upload.
- Pasted paper text.
- DOI input.
- PubMed link or PubMed ID input.

All input paths should normalize into the same internal representation: paper metadata, extracted text, chunks, embeddings, summaries, chats, and model metadata.

### Diabetes Scope

The app is explicitly focused on diabetes-related research papers. It is not a generic biomedical paper summarizer.

Scope classification uses a hybrid approach:

- Deterministic keyword/domain check first, using title, abstract, metadata, and extracted text.
- LLM-based scope classification only when the deterministic check is uncertain.

Possible outputs:

- `in_scope`
- `out_of_scope`
- `uncertain`

Out-of-scope papers should not proceed to full summarization in V1.

Uncertain papers should receive a user-facing message that the app cannot confidently verify the paper is diabetes-related. V1 should not summarize uncertain papers by default; an override can be considered later for advanced/demo use.

## User Experience

### First Screen

The first screen is a paper intake workspace with three clear input paths:

- Upload PDF.
- Paste text.
- Enter DOI or PubMed link/ID.

The interface should feel calm, practical, and trustworthy. It should not look like a technical dashboard.

### Summary View

After a diabetes paper is processed, the app generates a structured summary with sections such as:

- What this paper studied.
- Who was included.
- What the researchers found.
- Important numbers and outcomes.
- Technical terms explained.
- Limitations.
- What this may mean for patients.
- What this does not mean.
- Questions to ask a clinician.

Summary claims should cite source chunks where possible.

### Chat View

The chat interface allows users to ask follow-up questions. The chat should answer in a direct, approachable way while preserving source boundaries and avoiding personal medical advice.

## Chat And Knowledge Policy

The chat supports three knowledge modes.

### 1. Paper-Grounded Answers

If the answer is in the uploaded paper, the app answers from the paper and cites relevant paper chunks.

Example behavior:

> The study found that participants using the intervention had a larger average A1C reduction than the control group. [Paper citation]

### 2. General Diabetes Education

If the user asks a diabetes-related educational question that the uploaded paper does not answer, the app should still help.

The answer should:

- Answer naturally using trusted diabetes education knowledge.
- Clearly mention that the uploaded paper did not cover that topic.
- Avoid heavy disclaimers unless the user asks for personal medical advice.
- Cite the trusted diabetes education knowledge base once that source is available in the system.

Example behavior:

> An insulin pump is a small device that delivers rapid-acting insulin through a thin tube or patch under the skin. It can provide continuous background insulin and extra doses around meals. The uploaded paper does not discuss insulin pumps, so this explanation comes from general diabetes education rather than the study itself.

### 3. Personal Medical Advice Refusal

If the user asks what they personally should do, the app should not recommend treatment decisions.

The answer should:

- Refuse to give a personal recommendation.
- Explain what the paper says, if relevant.
- Suggest questions the user can ask a clinician.
- Use urgent-care language for emergency-like symptoms.

Example behavior:

> I cannot tell you whether to change your insulin dose. This paper discusses X in Y study population. A useful question for your clinician could be: "Does this finding apply to someone with my diabetes type and current treatment plan?"

## Architecture

### Frontend

Use React or Next.js for the patient/caregiver web app.

Frontend responsibilities:

- PDF upload, pasted text, and DOI/PubMed input.
- Recent papers list.
- Structured summary display.
- Citation display.
- Chat interface.
- Patient-friendly error states.

The frontend should not contain AI orchestration logic.

### Backend

Use FastAPI for the backend AI service.

Backend responsibilities:

- Accept paper inputs.
- Extract or fetch metadata.
- Extract text.
- Classify diabetes scope.
- Chunk and index paper text.
- Manage embeddings and retrieval.
- Manage a trusted diabetes education knowledge base.
- Generate structured summaries.
- Answer chat questions.
- Apply safety guardrails.
- Store papers, summaries, chunks, chats, costs, latency, prompts, and model metadata.
- Support automated tests and notebook evaluations.

### Persistence

V1 uses local/demo persistence, not real user accounts.

Store:

- Paper metadata.
- Uploaded file references.
- Extracted text.
- Chunks and chunk metadata.
- Embedding references.
- Structured summaries.
- Chat messages.
- Citations.
- Safety classifications.
- Model/provider metadata.
- Prompt versions.
- Token usage.
- Estimated cost.
- Latency.
- Evaluation outputs.

SQLite or Postgres are acceptable for local/demo persistence. A local vector store or Postgres vector extension can be used for embeddings.

## LLM Provider Abstraction

Use OpenAI as the first provider in V1, but hide provider-specific calls behind an internal interface so DeepSeek and open-source models can be added later.

Provider responsibilities include:

- `generate_structured_summary`
- `answer_with_context`
- `classify_scope`
- `classify_safety`
- `judge_output_quality`

This allows the same tasks to be run across multiple providers for fair comparison of quality, latency, safety, and cost.

## Data Flow

### 1. Intake

The user provides one of:

- PDF upload.
- Pasted paper text.
- DOI.
- PubMed link or PubMed ID.

The backend creates a `Paper` record and stores the source type.

### 2. Metadata And Text Extraction

The backend extracts or fetches:

- Title.
- Abstract.
- Authors, if available.
- Journal and date, if available.
- DOI or PubMed ID, if available.
- Full paper text, if available.

PDFs use PDF text extraction. DOI/PubMed inputs fetch metadata and abstracts or full text where legally accessible. Pasted text is treated as user-provided paper text.

### 3. Diabetes Scope Classification

The backend applies the hybrid classifier:

- Keyword/domain classifier first.
- LLM classifier only when uncertain.

Out-of-scope papers stop before summarization.

### 4. Chunking And Indexing

For in-scope papers:

- Split paper text into chunks.
- Preserve section and page metadata where available.
- Generate embeddings.
- Store chunks and embeddings.

### 5. Structured Summary

The backend generates a structured summary and stores:

- Summary JSON.
- Source chunk citations.
- Model/provider.
- Prompt version.
- Token usage.
- Estimated cost.
- Latency.

### 6. Chat

For each user question:

- Classify safety.
- Retrieve relevant paper chunks.
- Retrieve trusted diabetes education chunks if the paper does not answer a diabetes-related question.
- Generate an answer with clear source boundaries.
- Apply answer validation.
- Store message, answer, citations, safety label, model metadata, cost, and latency.

### 7. Evaluation

Tests and notebooks reuse stored papers, chunks, prompts, model outputs, safety labels, token usage, costs, and latency metadata.

## Safety And Guardrails

Core rule:

> Explain the research paper and general diabetes concepts. Do not give personal medical advice.

Allowed examples:

- "What was this paper studying?"
- "What does A1C mean?"
- "What were the key findings?"
- "Who was included in the study?"
- "What were the limitations?"
- "Does this paper mention side effects?"
- "What questions should I ask my doctor about this study?"
- "What is an insulin pump?"

Disallowed or redirected examples:

- "Should I change my insulin dose?"
- "Should I stop taking metformin?"
- "Which treatment should I choose?"
- "Do I have diabetes?"
- "Is this symptom an emergency?"
- "Can I ignore my doctor's advice?"

Guardrail layers:

- Input/query safety classification.
- Prompt instructions.
- Retrieval source boundary rules.
- Answer validation.
- Automated tests for unsafe prompts.
- README documentation of safety policy and limitations.

## Evaluation Strategy

Evaluation uses three layers.

### 1. Automated Backend Tests

Run with `pytest`.

Tests should cover:

- Diabetes paper accepted.
- Non-diabetes paper rejected.
- Diabetes scope classification edge cases.
- Paper-covered question gets paper citations.
- Diabetes-related but paper-not-covered question gets a general education answer.
- Paper-not-covered answer clearly says the paper did not cover it.
- General education answer cites trusted diabetes knowledge once the V1 knowledge base is available.
- Personal medical advice request is refused or redirected.
- Medication/dosing question does not produce dosing advice.
- Emergency-style query gets appropriate urgent-care redirection.
- Non-diabetes question is marked out of scope.
- Citation references valid paper chunks or trusted knowledge chunks.
- Document extraction and chunking work as expected.
- API endpoint behavior.

### 2. Jupyter Notebook Evaluations

Use notebooks for deeper analysis and portfolio storytelling.

Notebooks should cover:

- OpenAI vs DeepSeek comparison.
- Summary quality scoring.
- Chat answer quality scoring.
- Hallucination and error analysis.
- Source-boundary correctness.
- Cost per paper.
- Cost per chat answer.
- Latency comparison.
- Prompt version comparison.
- Retrieval quality for paper chunks.
- Retrieval quality for trusted diabetes education chunks.
- Optional open-source model comparison later.

### 3. Human And LLM Rubric

Use rubrics for quality that is not purely binary.

Score outputs on:

- Clarity.
- Completeness.
- Citation grounding.
- Source boundary correctness.
- Medical caution.
- Technical term explanation quality.
- Overclaiming risk.
- Usefulness for patients and caregivers.

LLM-as-judge can help scale evaluation, but important examples should receive human spot checks.

## Milestones

### V1: Core Patient App And OpenAI

Build:

- FastAPI backend.
- React/Next.js frontend.
- PDF upload.
- Pasted text input.
- DOI/PubMed metadata or abstract intake.
- Local/demo persistence.
- Diabetes scope classifier.
- Paper text extraction.
- Chunking and embeddings.
- Small trusted diabetes education knowledge base.
- Structured summary generation.
- Citation-grounded chat.
- Source-boundary behavior.
- Personal medical advice guardrails.
- Automated backend tests.
- Initial evaluation notebook.
- README architecture and safety documentation.

### V2: Model Evaluation

Add:

- DeepSeek provider.
- OpenAI vs DeepSeek comparison.
- Cost and latency tracking.
- Quality scoring notebook.
- Hallucination and error analysis.
- Prompt version comparison.
- README model comparison report.

### V3: Open-Source Model Serving

Optional extension:

- Qwen, Llama, or Mistral through Ollama, Hugging Face, or vLLM.
- Apply the same eval suite to the open-source model.
- Add deployment notes.
- Compare cost, latency, and quality against hosted APIs.

## Deployment Direction

The first version should work locally before cloud deployment.

Later deployment can use:

- Separate frontend and backend deployments.
- Managed database if needed.
- Object storage for uploaded PDFs.
- Background workers for long document processing.
- AWS or GCP deployment documentation.

Scaling should come after the local app, safety tests, and evaluation workflow are working.

## Non-Goals For V1

V1 does not include:

- Real patient accounts.
- Personalized treatment recommendations.
- Medication dose recommendations.
- Diagnosis.
- Emergency triage.
- Generic biomedical paper support.
- Fine-tuning.
- Quantization.
- Production-scale cloud infrastructure.
- Public technical evaluation dashboard.

## Success Criteria

V1 succeeds if:

- A patient or caregiver can upload, paste, or link a diabetes research paper.
- The app identifies whether the paper is diabetes-related.
- The app produces a structured summary that preserves key findings and limitations.
- The app explains technical terms in approachable language.
- Chat answers cite paper chunks when answering from the paper.
- Chat can answer general diabetes education questions when the paper does not cover the topic, while clearly saying the topic was not in the paper.
- The app refuses personal medical advice and redirects to clinician questions.
- Core guardrails and source-boundary behaviors are covered by automated tests.
- Initial notebook evaluation documents quality, failure cases, cost, and latency.
- The README clearly explains architecture, safety design, evaluation method, and project milestones.
