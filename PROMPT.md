# Web Ingestion Path

1. A daily trigger runs and iterates over a saved list of homepages, extracting the title, link, publication date, source metadata, and domain information from each article.
2. The system checks the vector DB to determine whether the article already exists, using the title, URL, publication date, and a separately stored content hash for faster lookup. If title or domain variations are present, a fuzzy layer (e.g. MinHash) is applied as a fallback. To detect the same content republished across different domains, a semantic fingerprint (e.g. SimHash of the summary) is compared globally before confirming the article is new. If it already exists, it is skipped.
3. The full article text is fetched as markdown.
4. The text is cleaned, normalized, and enriched.
5. The article is summarized and keywords are extracted.
6. A structured vector object is built containing:

   * Source URL
   * Source ID
   * Source domain
   * Title
   * Content
   * Keywords
   * Category tags
   * Summary
   * Publication date
   * Ingestion timestamp
   * Embedding model version
   * Processing pipeline version
   * Authority score
   * Freshness score
   * Tenant ID (for data isolation in multi-tenant environments)

7. An input guard validates content before indexing — filtering anything outside the configured domain, toxic or illegal material, PII that should not be indexed, and text resembling an embedded prompt injection. For fetched articles, a dedicated LLM-based detector for adversarial instructions in the body text is also run. Rejected items are routed to an inspection queue (dead letter queue) with metadata describing the rejection reason, enabling auditing and selective reprocessing.
8. Approved content is sent to the vector DB under the `web_articles` group, scoped by `tenant_id` for correct isolation across users and teams.
9. Retrieval metadata is updated to support freshness- and authority-sensitive ranking in future searches.

---

# File Ingestion Path

1. PDFs are uploaded through the web configuration page.
2. Each upload triggers the ingestion pipeline.
3. A file hash is generated and checked against the vector DB to prevent duplicates. If the file already exists, the process stops. Incremental update support: if a PDF changes, only the modified chunks are re-ingested.
4. The file is placed in the `rag-files` bucket awaiting processing.
5. The PDF is processed with Docling, including enrichment of formulas, images, tables, and code blocks.
6. Semantic chunking is run.
7. Keywords and tags are extracted per chunk.
8. Metadata is generated for each chunk:

   * Chunk ID
   * Parent document ID
   * Processing version
   * Embedding model version
   * Extraction timestamp
   * Tenant ID

9. An input guard checks each chunk for out-of-domain content, PII, toxic material, and embedded prompt injections before indexing. Rejected chunks are routed to the dead letter queue with a recorded rejection reason, enabling later auditing.
10. Approved chunks are sent to the vector DB scoped by `tenant_id`.
11. Processed files are deleted from the bucket.

---

# Daily News Generation

This flow requires the user to have previously uploaded a reference article and a sample post to the configuration page. Both are stored in the vector DB and processed by a text descriptor agent that builds a style and structure model for future writing agents.

1. Completion of the web ingestion path triggers this flow.
2. A hybrid search retrieves the 20 articles most similar to the base article from the vector DB.
3. Retrieved results are ranked using:

   * Semantic similarity
   * Freshness score
   * Authority score

4. Results are reranked and the top 10 are selected.
5. An input guard validates the retrieved articles before passing them to the writer — confirming they are within the expected domain, free of anomalous content, and not contaminated by prompt injections. An additional layer detects adversarial instructions hidden in the text.
6. A writer agent (with access to image planning and flowchart agents) picks the first article and drafts a Daily News piece referencing the original source, including a link to it. Image descriptions are inserted inline using `<img-desc>` tags for auditing.
7. An auditor agent evaluates the draft against the style and structure model. It acts as a model-based evaluator using a rubric that checks tone, structure, factual grounding, source attribution, and model adherence — requesting revisions as needed and sending the draft back to the writer until approved. The revision loop is capped at a configurable maximum number of iterations; if exhausted, the draft is forwarded to the human review queue with the full feedback history.
8. While articles remain:

   * A planner agent receives the next article, the model, and the current draft.
   * The planner determines how to integrate the new content while preserving coherence.
   * Instructions are returned to the writer.
   * A per-article token limit and a compaction step are applied to prevent the final text from becoming excessively long or repetitive.
   * If the writer fails to integrate after 3 revision cycles, the article is skipped and the failure is logged.

9. Once all articles are incorporated:

   * `<img-desc>` tags are removed programmatically.
   * A validation agent confirms complete removal.
   * A second validation checks for natural-language image descriptions outside the tags (e.g. sentences beginning with "The image shows…") and removes or flags them.
   * Another pass removes any residual agent language (e.g. "Here is the completed article...").

10. A cover image description is generated and the cover image is created.
11. The post text is generated and validated for agent traces.
12. **Agent harness + output guard — final evaluation gate:**

* A **code-based evaluator** checks structural conditions:

  * Image tags removed
  * No agent language patterns detected
  * Required links present
  * Output schema validation

* A **model-based evaluator** assesses:

  * Hallucination risk
  * Source grounding
  * Citation accuracy
  * Tone alignment
  * Editorial compliance
  * PII exposure

* The **output guard** enforces:

  * Toxicity policies
  * Domain compliance
  * Sensitive content policies

* Because model outputs vary across runs, the harness may execute multiple attempts and uses **pass^k** as a consistency metric — requiring the article to pass consecutive evaluations before being approved.
* If any check fails, the article is returned to the writer with structured feedback specifying what must change. This loop repeats until all evaluators approve, respecting the configured maximum iteration limit. If the limit is exhausted, the article is forwarded to the human review queue.

13. A confidence score is calculated.
14. Outputs below a configurable threshold can optionally be routed to a human review queue.
15. The final output is packaged into a folder containing:

* An images subfolder
* A markdown article file
* A social media post file

---

# Chat

The chat agent has access to:

* Files group
* Web articles group
* Conversation memory group

1. The user's input message is received.
2. An input guard runs before any other processing — checking for:

   * Prompt injection
   * Jailbreak attempts
   * Data exfiltration attempts
   * Topics outside system policy
   * Malicious intent

3. Messages are classified into risk levels:

   * Safe
   * Suspicious
   * Prompt injection
   * Data exfiltration
   * System override attempt

   **Mapped actions:**
   - *Suspicious* → cautious response with warning, logging.
   - *Prompt injection / Exfiltration / Override* → request rejected.
   - *Safe* → normal processing.

4. The system evaluates the message objective and generates a resolution plan.
5. A query router determines the appropriate execution path:

   * Direct response
   * RAG retrieval
   * Tool use
   * Multi-step research
   * Hybrid execution

6. If RAG is required:

   * The message is rewritten into a form better suited for semantic retrieval.
   * Hybrid retrieval is executed.
   * The top 20 results are retrieved.
   * Results are reranked using semantic relevance, freshness, and authority.
   * The top 10 are selected.
   * Tenant/user_id filters are applied to isolate data across different users and teams.

7. Retrieved context may be compressed to remove redundancy while preserving factual content.
8. A response is constructed using the retrieved context.
9. The response is audited against the original execution plan.
10. **Agent harness + output guard — final evaluation gate:**

* A **code-based evaluator** runs deterministic checks:

  * Response is not empty
  * Source references are present when RAG was used
  * Response length within configured limits
  * Schema compliance

* A **model-based evaluator** assesses:

  * RAG faithfulness
  * Grounding quality
  * Relevance
  * Tone compliance
  * Hallucination risk

* The **output guard** checks:

  * PII exposure
  * Toxicity
  * Out-of-policy content

* For consistency-critical deployments, **pass^k** is used — the response must pass the harness on consecutive attempts before being delivered.
* If any evaluator fails, the response is returned to the response builder with structured feedback detailing what must be corrected. This loop repeats until all checks pass, respecting the maximum iteration limit; if exhausted, the response is forwarded to the human review queue.

11. The validated response is delivered to the user.

---

# Evaluation Strategy

In addition to the real-time harness checks embedded in each flow, the system must be backed by a dedicated evaluation layer that operates independently of production — enabling continuous measurement of agent quality without affecting real users.

## Evaluation Types to Maintain

### Capability Evaluations

Capability evaluations measure what each agent can do well. They start with a low pass rate and represent a target to be reached.

Examples:

* Correctly referencing source articles
* Producing grounded summaries
* Following editorial structure
* Detecting prompt injections

They are run during development to guide improvements.

### Regression Evaluations

Once a capability reaches a sufficiently high pass rate, it is promoted to the regression suite.

Regression evaluations are run on every significant change and are intended to detect quality degradation.

---

## Evaluator Types Used Across the System

### Code-Based Evaluators

Used for deterministic checks:

* Tag removal
* Link validation
* Schema validation
* Citation presence
* Pattern detection
* Output formatting

### Model-Based Evaluators

Used for subjective assessment:

* Tone compliance
* Hallucination detection
* Grounding verification
* Editorial compliance
* RAG faithfulness
* Safety evaluation

These evaluators must be periodically calibrated against expert human judgment. Calibration should occur at minimum quarterly and whenever human-model agreement falls below a configurable threshold (suggested: 80%). Continuous agreement monitoring enables automatic drift detection ahead of the quarterly cycle, triggering an alert and pausing the affected evaluators until recalibration is complete.

### Human Evaluators

Used for:

* Calibrating model evaluators
* Validating subjective quality
* Auditing evaluation quality
* Reviewing samples of production outputs

---

## Handling Non-Determinism

Because all agents produce variable outputs across runs, evaluations must account for variance.

### pass@k

Used for capability evaluations where finding at least one successful solution in k attempts is sufficient.

### pass^k

Used for production-facing agents where successful behavior is required across all consecutive attempts.

---

## Transcript Review

Evaluation scores alone are not sufficient.

Transcripts — including:

* Tool calls
* Intermediate outputs
* Guard decisions
* Revision loops
* Evaluator assessments

must be reviewed regularly to verify that:

* Evaluators are measuring the intended behavior.
* False failures are not occurring.
* Valid solutions are not being penalized.
* Agent behavior remains aligned with business objectives.

A failing evaluation that appears correct after transcript review indicates a problem with the evaluator, not necessarily with the agent.

---

## Long-Term Maintenance

The evaluation suite is a living artifact — and should be treated as code: with explicit versioning, a changelog, and a lock on examples used in production to prevent cross-version contamination.

As new:

* Article types
* File formats
* Query patterns
* User behaviors

emerge, new evaluation tasks must be added.

Evaluation saturation reduces the improvement signal and requires the periodic introduction of harder scenarios.

Responsibility for the evaluation suite should be shared between:

* Engineering teams (infrastructure and automation)
* Domain experts (task definition and evaluator calibration)

---

# Observability Layer

Across all flows, **Langfuse** or **LangSmith** provides full observability over:

* LLM calls
* Agent execution paths
* Guard decisions
* Harness evaluations
* Revision loops
* Retrieval performance
* Reranking performance

The platform records:

* Blocked inputs
* Rejected outputs
* Revision counts
* Latency per step
* Token usage
* Cost per request
* Retrieval hit rate
* Citation coverage
* Hallucination rates
* Evaluator results
* Prompt versions
* Pipeline versions

Additionally, the following are monitored:

* Distribution drift (input, retrieval, embeddings, user behavior)
* Number of articles skipped in the news flow due to iteration limits
* Agreement between model evaluators and human review (for early descalibration detection)
* Accumulated cost per run and per day, with alerts when configurable thresholds are reached
* Volume and patterns of items routed to the dead letter queue, by flow and rejection type

These data feed continuous improvement of:

* Guard rules
* Prompt templates
* Evaluator rubrics
* Retrieval strategies
* Agent workflows

Production monitoring through this layer also detects distribution drift, retrieval drift, embedding drift, and emerging user behaviors that offline evaluations may not anticipate, making observability a critical complement to the evaluation framework.