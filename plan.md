# Natural Language to SQL Chatbot with Plotting

*Project design document*

## Overview

The goal is to build a chatbot that lets users ask plain-language questions about a database or spreadsheet, converts those questions into SQL, executes them, and returns a natural-language answer — optionally accompanied by a chart. The system should also support follow-up questions that build on prior turns in the conversation.

**Core user flow:**
1. User asks a question in plain language (e.g., *"What were sales on 2/2/2026?"* or *"Compare this month's sales to last month"*).
2. The system translates the question into SQL.
3. The SQL runs against the data source, and an LLM turns the result into a readable answer.
4. If a chart would help — trends, comparisons, distributions — the system generates one.
5. Raw SQL and results remain visible in a separate tab for transparency.
6. Conversation history is retained so follow-up questions work naturally.

---

## Core Components

### 1. Natural Language → SQL

- A schema-aware LLM (e.g., GPT-4, Codex, SQLCoder) translates the question into SQL.
- The model needs table/column names, relationships, data types, and descriptions to work reliably.
- Handling complex schemas:
  - **Star/snowflake schemas** — include join paths and common aggregation patterns in the prompt.
  - **Cryptic column names** — maintain a data dictionary mapping technical names to human-friendly terms (`cust_id` → "customer ID").
  - **Domain jargon** — fine-tune on domain vocabulary or supply a glossary.

### 2. Query Execution

- Connects to a database (SQLite, PostgreSQL, etc.) or loads Excel/CSV into a local engine (DuckDB, pandas).
- Validates and runs the SQL; errors are captured and fed back to the LLM for iterative correction.

### 3. Answer Generation

- Raw query results, the original question, and the SQL used are passed to an LLM to produce a natural-language answer.

### 4. Plotting Decision & Visualization

- A rule-based system or lightweight classifier decides whether a chart is warranted.
- Charts are generated with Matplotlib, Plotly, or Vega-Lite.
- Chart type is chosen based on data shape and intent (line for time series, bar for comparisons, etc.).
- The UI shows both the raw result table and the chart in separate tabs for transparency.

### 5. Conversational Context Management

- Session history supports follow-ups that reference earlier queries or results.
- The system decides whether to reuse/modify the previous SQL or generate a fresh query.
- LLM-based context tracking resolves references like "that," "it," or "last month."

---

## Enriched Metadata (Data Dictionary)

A bare schema isn't enough — a rich metadata layer substantially improves translation accuracy. It should include:

| Element | Purpose |
|---|---|
| Table descriptions | Business meaning of each table |
| Column descriptions | Plain-language explanation, units, format, business rules |
| Relationships | Foreign keys, join conditions, cardinality |
| Example values | Typical samples per column (e.g., `status: 'active', 'inactive'`) |
| Data types & constraints | SQL types, allowed ranges, nullability, uniqueness |
| Synonyms | User-friendly terms mapped to actual column names |
| Aggregation hints | Whether a numeric column is typically summed, averaged, or counted |
| Time dimensions | Granularity of date columns; presence of a calendar table |

This metadata should be stored as structured YAML/JSON, and fed to the LLM in full (small schemas) or via retrieval (large schemas). Example:

```yaml
tables:
  - name: sales
    description: "Daily sales transactions"
    columns:
      - name: order_date
        type: date
        description: "Date the order was placed"
        example: "2026-02-02"
      - name: revenue
        type: float
        description: "Total revenue in USD, before discounts"
        example: 1250.50
    relationships:
      - foreign_key: customer_id
        references: customers.id
        type: many_to_one
```

---

## SQL Generation & Error Handling

- On execution failure, the error is passed back to the LLM alongside the original question so it can generate a corrected query — looping until it succeeds or a retry limit is hit.
- Empty results are also flagged, since they may indicate a logical error rather than a genuine absence of data.
- Two validation layers apply: **syntactic** (does it run?) and **semantic** (does it actually answer the question?).

---

## Validating SQL Results

**Syntactic validation (execution check)**
- Run the query; on error, retry with the error fed back to the LLM.
- Flag empty results for review.

**Semantic validation (intent alignment)**
- *LLM self-critique* — ask the model to judge whether the SQL and its result actually answer the question, with a confidence score.
- *Rule-based checks* — confirm the result shape matches expectations (e.g., date + numeric columns for a time-series question).
- *Data type checks* — verify expected types (e.g., an integer for a count).
- *Column matching* — confirm the result includes columns implied by the question.

**Ambiguity detection**
- If the model is uncertain, it should ask a clarifying question before running anything.

**User feedback loop**
- Show the generated SQL and raw results; let the user correct mistakes, and feed that correction back into future queries.

---

## Logging for Quality & Improvement

Each interaction should log:

- Raw user question and timestamp
- Generated SQL, including intermediate retry versions
- Execution status (success / error / empty)
- Result metadata (row count, column names, sample rows)
- Final answer text and chart type (if any)
- User feedback (if given)
- Session ID for conversation tracking

**What the logs are for:**
- Building regression test sets
- Spotting common failure modes (date parsing, wrong joins, etc.)
- Extracting high-quality question–SQL pairs for few-shot examples
- Tracking performance over time
- Audit and compliance

**Privacy:** anonymize personal data, restrict log access, and be transparent with users about what's recorded.

---

## Caching

Caching full query results generally isn't recommended for frequently changing data. Better alternatives:

- Cache only immutable metadata (schema, descriptions).
- Keep large files in memory to avoid repeated I/O.
- Use TTL-based caching only where update cycles are well understood.
- Invalidate via database triggers where feasible (adds complexity).
- Pre-compute and periodically refresh summary/aggregate tables.

**Recommendation:** skip response caching in an initial build and prioritize correctness first.

---

## Feasibility & Limitations

**Is it feasible?** Yes, with caveats:
- Simple, well-documented schemas achieve high accuracy with modern LLMs.
- Complex schemas need retrieval (to select relevant tables/columns), multi-step reasoning, or query decomposition to hold up.

**Key limitations:**

1. **Text-to-SQL accuracy** — ambiguous phrasing, implicit joins, date expressions, and complex aggregations remain common failure points.
2. **Data privacy** — cloud LLMs may see sensitive schema/data; consider local models or obfuscation.
3. **Performance** — large datasets and multiple LLM calls add latency.
4. **Plotting judgment** — deciding when and how to chart is inherently subjective; allow user override.
5. **Excel handling** — multiple sheets, merged cells, and non-tabular layouts require preprocessing.
6. **Conversational state** — balancing context retention with a clean "start new chat" option is tricky.
7. **Explainability** — showing the SQL and raw data (via UI tabs) is important for building user trust.

---

## Additional Considerations

**User interface**
Chat window with the answer text, a collapsible raw-data table, a chart panel, and controls for "Show SQL," "Download data," "Change chart type," and "Start new chat."

**Data connectivity**
Support for databases, CSV, Excel, and Google Sheets, with periodic refresh.

**Security & permissions**
Role-based access control and audit logging.

**Error handling & recovery**
Clear, graceful error messages, with suggested corrected queries where possible.

**Clarification dialogue**
A dialogue manager that asks for missing information (e.g., "total or average sales?") rather than guessing.

**Evaluation & testing**
Maintain a test set of natural-language questions paired with expected SQL, and use it to build few-shot examples.

**Local vs. cloud LLM**
Cloud models offer higher accuracy but raise cost and privacy concerns; local models trade accuracy for control (and need GPU resources). A hybrid approach is worth considering.

**Multi-modal output**
Support formatted tables, pivot tables, or dashboards for recurring queries.

**Schema compression**
For large schemas, retrieve relevant tables/columns via embeddings first, then feed only that subset to the SQL generator.

**Query decomposition**
Break complex questions into sub-questions, answer each, and combine the results.

**Feedback loop**
Learn from user corrections over time to improve future translations.

**Template matching**
Predefine SQL templates for common patterns (e.g., "sales by [dimension] for [time period]").

**Chart suggestion**
Use a lightweight classifier on the result's structure to recommend an appropriate chart type.

---

## Conclusion

This is an ambitious but achievable prototype. The main challenges are handling complex schemas, maintaining conversational continuity, and making sound plotting decisions. The recommended path is to start small — a single table with a simple schema — then expand outward, incorporating rich metadata, validation loops, logging, and user feedback along the way. Skip response caching initially and prioritize correctness. Over time, the system should improve through iterative refinement based on logged interactions and user feedback.