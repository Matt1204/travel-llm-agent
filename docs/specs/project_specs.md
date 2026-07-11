# Travel LLM Agent — Architectural Specification

This document specifies the current system as **version 1 (v1)** and provides an implementation-grade baseline for teams evolving it into **version 2 (v2)**. Statements marked **Inferred** describe intended operational policy where the repository does not encode a formal contract.

## 1. System Boundary and Context Analysis

### Purpose and objectives

Travel LLM Agent is a conversational trip-planning system that decomposes flight planning into specialized, stateful workflows. A supervisor interprets user requests and delegates to agents that elicit structured flight preferences, conduct interactive flight search and booking, or autonomously discover better-priced alternatives. The system also contains a taxi-management capability and legacy booked-flight management tools that demonstrate how the same orchestration model can extend across a trip lifecycle.

The principal product objective is to translate natural-language requests into validated domain state and controlled side effects. The flight-requirement workflow turns preferences into prioritized airport, time-window, and budget constraints. The flight-search workflow queries live Amadeus inventory, ranks offers, retrieves priced details, and records confirmed bookings. The discovery workflow establishes a baseline offer and iteratively explores nearby airports, time windows, and budgets until it finds multiple better deals or exhausts its attempt policy.

### Operating ecosystem and actors

The direct actor is an authenticated passenger using the terminal chat runner. `main.py` supplies a `passenger_id` and a LangGraph `thread_id` through `RunnableConfig`; the graph treats these values as the caller identity and conversation identity. The indirect actors are OpenAI chat models, the Amadeus OAuth and travel APIs, and a local SQLite database. The included database preparation utility can obtain a benchmark travel database from Google Cloud Storage and shift its sample dates to the current period.

Inside the system boundary are the CLI interaction loop, LangGraph supervisor, specialized agent graphs, prompts, Pydantic domain models, tool functions, Amadeus client and normalization/ranking logic, SQLite repositories expressed as parameterized queries, message-history reducers, and checkpoint state. Outside the boundary are terminal input/output, model inference, Amadeus inventory and pricing, credential issuance, network transport, the benchmark database download, and any real airline ticketing or payment settlement.

### Boundary data flows

User text enters the supervisor graph as `HumanMessage` objects. The graph sends prompts, message history, tool schemas, structured requirements, and selected state to OpenAI; model responses return as assistant messages and tool calls. Flight tools send IATA codes, dates, passenger counts, currency and optional airline/non-stop filters to Amadeus and receive JSON flight offers, airport metadata and priced offers. Persistence tools exchange parameterized scalar fields and JSON-serialized aggregates with SQLite. The CLI renders only relevant human and assistant messages and resumes graph interrupts with approval or revision text.

### Trust, security and ownership boundaries

The caller-facing trust boundary is `RunnableConfig`: `passenger_id` scopes passenger-owned queries and mutations, while `thread_id` scopes checkpointed conversational state. The external-service boundary is HTTPS plus OAuth bearer authentication to Amadeus and API-key authentication handled by the OpenAI SDK. Environment variables own secrets; graph state and tool results must not contain credentials. SQLite is an application-owned trust boundary and is assumed to be accessible only to the local runtime. The application owns validation, delegation, authorization checks on passenger records, tool invocation, confirmation policy and normalization; Amadeus owns inventory accuracy and price confirmation; the model provider owns inference availability; the deployment operator owns secret distribution and database protection.

## 2. Overall Architecture Design

### Architectural style

V1 is a stateful modular monolith with event-driven graph orchestration. It runs as one Python process, but internally isolates supervisor, requirement elicitation, interactive flight search, baseline discovery, iterative discovery, Amadeus integration and SQLite persistence behind tool and state contracts. LangGraph nodes behave as event-processing components: messages and `Command` updates advance a deterministic routing graph while LLM nodes choose semantically appropriate tools.

This style is suitable because the workflows share passenger state and domain models while requiring explicit handoffs and resumable conversations. A distributed service topology would add serialization, deployment and consistency costs without improving this single-user demonstration runtime. The graph provides a clean future seam for extracting high-load or privileged modules.

### High-level structural diagram

```text
Terminal Passenger
      |
      v
main.py CLI / approval resume loop
      |
      v
Supervisor StateGraph (MemorySaver)
  |        |                    |
  |        |                    +--> Primary read tools --> SQLite
  |        +--> Flight Search worker graph --> Amadeus --> booking table
  +--> Intent elicitation graph -----------> flight_requirements table
  +--> Baseline + iterative discovery -----> Amadeus
                                             |
                                             +--> flight_discovery table

All LLM nodes --> OpenAI Chat Completions through LangChain
```

### Runtime topology, scaling and deployment

The default topology is one CLI process, one in-memory supervisor checkpointer, a separate in-memory flight-search checkpointer, one SQLite file (`llm-agent-db.db`), and outbound HTTPS sessions. The code is launched with `python main.py`; dependency versions are pinned in `requirement.txt`. There are no required containers or infrastructure manifests.

V1 scales vertically and by independent process, provided each process has an isolated or deliberately shared database and checkpoint policy. SQLite permits simple local concurrency but serializes writes. In-memory checkpoints bind continuity to a process. **Inferred v2 policy:** externalize checkpoints and relational persistence before horizontally scaling, use one logical conversation writer per `thread_id`, and apply bounded concurrency to outbound flight searches.

Fault tolerance is localized: the `Assistant` wrapper reprompts once on an empty model output; Amadeus calls use timeouts and bounded exponential retries for HTTP 429, 5xx, timeout and connection failures; graph loop budgets and discovery attempt thresholds bound autonomous work; SQLite writes use explicit commits; worker handoffs return control to the supervisor. The system has no multi-region behavior in v1. **Inferred:** a multi-region design should keep a passenger/thread affinity or use a globally accessible checkpoint store and designate a write region for each booking aggregate.

### Trade-offs and alternatives

The modular-monolith design prioritizes traceable workflows and rapid evolution over independent service scaling. LangGraph supplies explicit state transitions and checkpoints, whereas an unstructured chain would make multi-agent handoffs and approval resumes difficult to reason about. Tool calling preserves domain operations as code instead of asking the LLM to manipulate databases or external APIs directly. SQLite optimizes local reproducibility; a managed relational database is the natural alternative for concurrent production traffic. Direct synchronous Amadeus calls simplify reasoning but make long date windows latency-sensitive; a queue-based discovery worker is a future alternative.

## 3. Complete Module Breakdown

### CLI and presentation (`main.py`, `utils.py`)

This module owns terminal interaction, thread creation, graph streaming, incremental message rendering and human resume input. Its public interface is the executable entry point; its internal interface is `graph.stream`, `graph.invoke(Command(resume=...))` and `graph.get_state`. It does not own conversational decisions or persistence. State is limited to per-message-list display counters. Errors while rendering fall back to safe text. Execution is synchronous and single-threaded. Its separation keeps UI concerns out of domain tools and enables replacement by an HTTP or web frontend.

### Supervisor orchestration (`graph.py`, `graph_setup.py`)

The supervisor owns graph topology, dispatch, worker handoff and aggregation of worker responses. `graph_setup.py` centralizes stable node identifiers. `fetch_user_info_node` loads passenger requirements, `decision_after_fetch_user_info` restores the active dialog mode, and `decision_after_primary_thought` routes declared tool calls. The flight-search adapter maps supervisor state to `FlightSearchState`, invokes the worker with the same runtime config, preserves a dedicated worker transcript, and maps handoff results back to the supervisor.

Its input is the shared `State`; its output is a state delta or `Command` specifying a destination. It owns no durable business records. The compiled graph uses `MemorySaver`, so state persists strongly within a process and thread. Routing errors are converted into a corrective `ToolMessage`. Concurrency is serialized per synchronous graph invocation. Security relies on preserving the original `RunnableConfig` into tools and worker graphs.

### Shared state and primary assistant (`primary_assistant_chain.py`)

This module defines the canonical graph state, reducers and supervisor prompt. `State` partitions the public transcript, requirement aggregate, worker-specific transcripts, dialog stack, discovery snapshots and loop budget. `update_dialog_stack` implements push/pop semantics; message reducers append or explicitly reset worker histories; the requirement reducer replaces aggregates atomically. `Assistant` normalizes missing message lists, invokes the runnable with tracing config, and performs one corrective reprompt for an empty response.

The primary assistant binds an OpenAI model to read tools and handoff schemas. It is responsible for task classification and natural-language continuity, not for directly changing requirements or booking records. Its performance is dominated by model latency and prompt size; specialized transcripts limit cross-worker context growth.

### Primary tools and handoff contracts (`primary_assistant_tools.py`)

This module owns passenger-scoped retrieval of flight requirements, discovered deals and booked flights, plus typed handoff contracts. Retrieval tools read SQLite and turn JSON columns into structured results. Handoff tools emit `Command` updates, add correlated `ToolMessage` acknowledgements, set `requirement_id`, update the dialog stack and route to specialist entry nodes. Pydantic classes such as `HandoffToFlightSearchAgent`, `HandoffToFlightDiscoveryAgent` and `WorkerCompleteOrEscalate` make LLM routing arguments explicit.

The module does not perform live search. Parameterized SQL provides injection resistance, and passenger filters enforce record ownership on user-facing reads. Tool-call IDs preserve LangChain message protocol correlation.

### Intent elicitation (`intent.py`, `intent_tools.py`, `intent_model.py`, `intent_assistant_chain.py`)

The intent subsystem owns creation and refinement of a `FlightRequirements` aggregate. Its ReAct agent receives a runtime prompt that distinguishes create from update, loads existing requirements by ID, and uses granular tools to add, update or remove priority levels. Completion is explicit through `IntentElicitCompleteOrEscalate`; persistence occurs only through `sync_flight_requirements_to_db` after the conversation reaches a confirmed state.

`FlightRequirements` contains `AirportRequirements`, `TimeRequirement` and `BudgetRequirement`, all extending `PriorityRequirement`. Pydantic validators normalize airport strings, ISO datetimes and numeric budget ranges. JSON serialization is the boundary representation. SQLite stores each requirement dimension as JSON alongside a UUID, passenger ID and concise description. The graph mutates one aggregate within a conversation and replaces it in state; the final database write is one transaction.

### Interactive flight-search worker (`flight_search_agent.py`, `flight_search_tool.py`)

This worker owns user-guided offer search and booking. Its entry node loads a requirement aggregate by `requirement_id`. A prebuilt ReAct graph receives a dedicated message list and tools for search, details, booking and supervisor handoff. Search applies exact endpoints, a two-value departure window, optional budget and grouped ranking. Detailed lookup searches candidate days, can filter carrier and flight number, and asks Amadeus to reprice the selected offer. Booking creates an application booking UUID and persists the priced offer JSON after checking passenger identity and required state.

The worker does not edit the requirement aggregate. It communicates synchronously with Amadeus and SQLite. A `remaining_steps` field bounds tool loops. Its checkpointer is an `InMemorySaver`, and the supervisor separately retains the full worker message list.

### Baseline and autonomous discovery (`flight_discovery_agent.py`, `fd_baseline_chain.py`, `fd_discovery_chain.py`, `fd_discovery_graph.py`)

The baseline subsystem converts the highest-priority requirement values into a canonical search filter, searches offers, evaluates candidates and finalizes a baseline snapshot. `FDSnapshot` records ID, filter, altered values, normalized flights, better-deal status and evaluation comment; `FDSessionInfo` owns one baseline plus discovered attempts.

The iterative discovery subsystem receives the baseline and repeatedly creates a blank attempt snapshot, asks the model to alter a filter, searches, compares candidate prices against the baseline, and records a reflective comment. It can resolve nearby airports through Amadeus reference-data endpoints. `review_discovery_session` stops after at least three successful attempts or six unsuccessful attempts. `save_discovery_session` serializes every attempt and the baseline into `flight_discovery` rows under a shared session UUID. Worker completion routes through `fd_return_node`, clears specialist histories and returns to the supervisor.

The subsystem's loop is synchronous and stateful. Snapshots are the transaction boundary for reasoning; the session database write is a batch commit. Its separation makes autonomous exploration auditable independently of the primary transcript.

### Amadeus integration and offer model (`amadeus_api.py`, `flight_model.py`)

`AmadeusClient` owns OAuth token acquisition/caching, authenticated HTTP, retry policy, daily offer search, multi-day aggregation, repricing, airport lookup and nearest-airport lookup. It accepts a pre-obtained bearer token or client credentials from environment variables. Its shared `requests.Session` reuses connections. Calls use a configurable 20-second timeout by default and up to three retries with exponential delay for rate limits, server errors and transport failures.

`FlightOfferModel`, `SegmentModel` and `FareModel` provide the normalized application contract. Normalization extracts flight number, endpoints, schedule, duration, stop count, directness, fare amounts and raw provider payload. Ranking computes price, duration and stop metrics and emits grouped direct/transfer results. This layer owns provider-specific JSON interpretation; agents consume stable application models.

### Legacy flight and taxi capabilities (`tools_flight.py`, `tools_taxi.py`, `flight_assistant_chain.py`, `taxi_assistant_chain.py`, `graph_flight.py`, `graph_taxi.py`)

These modules define reusable trip-management operations: passenger itineraries, local flight lookup, ticket rescheduling/cancellation, and taxi request create/read/update/delete. Sensitive mutations verify the configured passenger owns the target record, and graph handlers can interrupt before privileged tools. Although their graph registration is not enabled in the default supervisor build, they form an extension boundary for a broader trip assistant and document the intended safe-versus-sensitive tool pattern.

### Demonstration and alternate orchestration (`demo-supervisor.py`, `graph_1.py`, `assistant_chain_1.py`, `main_1.py`, `custom_tool_node.py`)

These modules provide experiments for supervisor/worker delegation, alternate assistant composition, fake-model execution and custom tool-node semantics. `custom_tool_node` demonstrates explicit tool lookup, invocation, `Command` propagation and correlation of tool results. They are development entry points rather than the default runtime and should remain isolated from production graph imports.

## 4. Inter-Module Relationships and Communication

### Communication contracts

CLI-to-supervisor communication is synchronous in-process invocation using LangChain message objects and `RunnableConfig`. Supervisor-to-agent communication uses shared-state fields and LangGraph `Command` routing. Agent-to-tool communication uses JSON-schema tool calls with a unique tool-call ID; tools return plain objects, `ToolMessage`, or `Command` state updates. Agent-to-model communication is synchronous through LangChain's OpenAI adapter. Tool-to-Amadeus communication is synchronous HTTPS/JSON with bearer authorization. Tool-to-SQLite communication is synchronous parameterized SQL over a file connection.

No message broker or public HTTP endpoint exists. Therefore topic naming, circuit breakers and cross-service distributed transactions do not apply in v1. Message ordering is the LangGraph node order plus append reducers. Per invocation, state updates are deterministic except for model output and external data. SQLite commits provide atomicity for an individual tool transaction; Amadeus calls and database writes are not one distributed transaction.

### Normal flight-requirement flow

The supervisor fetches requirements for the configured passenger, then the model calls the intent handoff contract. The handoff writes a correlated tool acknowledgement, requirement ID when updating, and `in_intent_elicitation_assistant` on the dialog stack. The intent entry node initializes or loads the aggregate. The ReAct agent invokes narrowly scoped mutation tools until required priority-one fields are present. On user confirmation, the sync tool JSON-serializes all dimensions and inserts or updates one row. Completion pops the dialog mode and returns to the primary assistant.

### Normal interactive search and booking flow

The primary assistant hands off with an existing requirement ID. The supervisor loads the aggregate into a worker state and forwards the latest user message. The worker calls `search_flights`; this expands the date window into per-day Amadeus queries, normalizes offers, filters budget and ranks direct and connecting groups. If the user selects a candidate, `get_flight_details` re-queries a narrow window, filters the flight number and invokes provider repricing. A confirmed booking is recorded under the current passenger. The worker calls its handoff tool when done; the supervisor stores its transcript, pops dialog state and summarizes the handoff.

### Normal discovery flow

The baseline entry loads the requirement, chooses highest-priority values, searches and evaluates a baseline. Discovery initializes a session and attempt snapshot. Each iteration calls a filter-update/search tool, compares normalized minimum prices with the baseline, and then calls evaluation with a concise directional comment. The entry node reviews completed snapshots; after the success/failure threshold, the entire session is persisted and a completion call returns control to the supervisor.

### Failure, retry and partial-outage behavior

Model empty output receives one reprompt; otherwise the graph ends or follows a declared handoff. Unknown supervisor tool names produce an explicit corrective message and return to the assistant. Amadeus 429 and 5xx responses and transport failures receive bounded retries; non-retryable errors are returned as structured error payloads. A partial Amadeus outage prevents live search but leaves requirements, prior bookings and stored discovery results readable from SQLite. A SQLite outage prevents requirement sync, booking and discovery persistence but does not inherently prevent stateless Amadeus querying.

Idempotency varies by operation. Requirement updates are idempotent for the same values, reads are idempotent, and discovery uses `INSERT OR REPLACE` keyed by deal identity. New requirements, sessions and bookings use generated UUIDs and are not retry-idempotent unless the caller preserves the generated identity. **Inferred v2 policy:** accept an operation ID for every external mutation, add unique constraints, and retry only when commit outcome is known or deduplicated.

Timeouts are explicit only at Amadeus HTTP calls. **Inferred:** model and graph invocations should receive deployment-level deadlines, and SQLite should use a busy timeout. The current retry policy has no formal circuit breaker; v2 should open a short-lived provider circuit after repeated failures and return a degraded response. Ordering is strong within one graph run, while external inventory is eventually changing and must be repriced before booking.

## 5. Domain Model and Behavior Design

### Core aggregates and value objects

`FlightRequirements` is the preference aggregate. Its child value objects are typed priority maps: airport code strings, two-datetime departure windows, and two-number budget boundaries. Lower priority numbers mean higher preference. The aggregate exposes add, update, remove, lookup and JSON conversion operations. The database record owns a requirement UUID, passenger ownership and human-readable description.

`FlightOfferModel` is a normalized provider offer with schedule, route, fares, minimum amount, segments, directness, stops, duration and raw payload. `FDSnapshot` represents one discovery experiment and `FDSessionInfo` aggregates a baseline plus ordered experiments. The booking aggregate is a database row containing booking UUID, passenger ID, requirement ID, priced flight JSON and creation time. Taxi requests and benchmark tickets are secondary aggregates scoped to a passenger.

### Invariants and transitions

A persisted flight requirement must have priority-one departure airport, arrival airport and departure time. A time window must contain exactly two values and end after start. A budget boundary must contain two numeric values with minimum below maximum. Requirement mutation happens in memory first and becomes durable only through the explicit sync tool.

The dialog stack transitions from primary to one specialist mode and returns through `pop`. A flight-search worker transitions `active -> awaiting_user` when it returns an answer without handoff, or `active -> handed_off` when its handoff flag is set. A discovery snapshot transitions `created -> searched -> evaluated`; creating a next snapshot is allowed only after filter, flights and comment are populated. A discovery session transitions `baseline_pending -> discovering -> completed -> persisted`. Baseline absence routes back to baseline generation rather than allowing discovery.

Booking follows `candidate_searched -> details_priced -> user_confirmed -> persisted`. The graph prompts and tool boundary enforce the user-confirmation step; actual payment and airline order creation remain outside scope. Legacy ticket rescheduling forbids changing a ticket owned by another passenger and forbids a new departure less than three hours away.

### Rules, orchestration and side effects

Interactive search starts with all priority-one criteria and then changes one criterion at a time when exploring alternatives. Discovery evaluates candidates relative to the baseline minimum price and seeks three successes before stopping; six unsuccessful attempts bound the search. Airport discovery resolves a city keyword to a representative airport and up to five nearby alternatives within the adapter's search radius.

Side effects are limited to explicit tools: requirement sync, booking persistence, discovery-session persistence, ticket update/cancellation and taxi mutation. Domain validation resides in Pydantic models and tool preconditions, preventing prompts from being the sole source of correctness. The transaction scope is one SQLite tool call. Conversation state and durable state have a hybrid consistency model: graph state changes immediately, while database state changes only at explicit synchronization points.

## 6. Data Architecture

### Technologies and schema

SQLite is the v1 system of record because it is portable, requires no service, supports transactions and matches the local conversational runtime. The repository uses both an application database (`llm-agent-db.db`) and a downloadable benchmark travel database utility. JSON columns preserve flexible priority maps and provider payloads without forcing every offer field into relational columns.

The code directly addresses `flight_requirements`, `flight_discovery`, `bookings`, `taxi_requests`, and benchmark tables including `tickets`, `ticket_flights`, `flights`, `boarding_passes` and `bookings`. `flight_requirements` contains requirement ID, passenger ID, description, departure-airport JSON, arrival-airport JSON, departure-time JSON and budget JSON. `flight_discovery` contains session ID, deal ID, source requirement ID, baseline filter/flight, discovery filter/flights, altered values and a better-deal flag. Booking records contain generated identity, ownership, source requirement and the priced offer.

### Indexing, partitioning and lifecycle

The natural access patterns require primary or unique indexes on requirement ID, deal ID, booking ID, request ID and ticket number. **Inferred v2 policy:** add composite indexes on `(passenger_id, requirement_id)`, `(source_requirement_id, session_id)`, `(passenger_id, created_at)` and ownership keys used by ticket/taxi queries. SQLite has no application partitioning. At production scale, partition discovery and booking history by passenger or time in a managed relational database.

V1 does not encode migrations, archival or retention. **Inferred:** introduce numbered migrations before schema evolution; retain active requirements and bookings for the account lifetime; apply a configurable retention window to graph transcripts and unsuccessful discovery payloads; preserve booking audit metadata according to product and legal policy. Provider raw payloads should be pruned or encrypted if they contain personal data.

### Caching and consistency

OAuth tokens are cached in memory until shortly before expiry. Graph checkpoints and worker histories are in-memory caches scoped by `thread_id`. No offer cache is used, which is appropriate for volatile inventory; pricing is explicitly refreshed before booking. SQLite provides strong consistency inside one connection transaction. The overall system is hybrid: persisted aggregates are strongly committed locally, model/checkpoint state is process-local, and external flight availability is eventually changing.

There is no read replica or read/write split. **Inferred v2 policy:** keep authoritative bookings and requirements on a primary database, serve non-critical discovery history from replicas, and invalidate any offer cache on a short TTL rather than mutation events.

## 7. API and Contract Design

### Public interface

V1's public API is conversational rather than HTTP. An invocation accepts a state update containing `messages`, plus configurable `passenger_id` and `thread_id`. It returns/streams graph states containing assistant messages and specialist state. The CLI recognizes `exit` and approval responses to interrupted actions. There is no formal URL version, authentication endpoint, rate limiter or HTTP error envelope.

**Inferred v2 API:** expose `/v1/conversations/{thread_id}/messages`, `/v1/requirements`, `/v1/flight-searches`, `/v1/discovery-sessions` and `/v1/bookings`, while retaining the graph as application orchestration. Responses should use stable IDs, ISO-8601 timestamps and a common error body `{code, message, retryable, correlation_id, details}`. Authentication should establish passenger identity server-side rather than accepting it from an untrusted request body.

### Internal tool contracts

Internal contracts are LangChain tools generated from Python signatures and docstrings. Requirement tools accept `priority` and typed values. Flight search accepts endpoint codes, a two-datetime window, optional budget and `top_k`, returning applied filters and grouped ranked results. Details lookup adds carrier, non-stop and flight-number selection and returns a selected offer plus provider pricing. Handoff contracts carry task description, requirement ID or completion reason. Injected state, runtime config and tool-call ID are not model-supplied arguments.

State contracts are `TypedDict` definitions with reducers. Messages must preserve tool-call/result pairing. Worker history is isolated under `flight_search_messages` or `flight_discovery_messages`; supervisor messages remain under `messages`. Domain data crossing the persistence boundary is JSON with ISO-formatted datetimes.

There are no event topics. Conceptually, the internal events are `RequirementConfirmed`, `SearchCompleted`, `OfferPriced`, `BookingConfirmed`, `DiscoveryAttemptEvaluated` and `DiscoverySessionCompleted`. **Inferred v2 policy:** if events are externalized, name topics `travel.requirement.v1`, `travel.booking.v1` and `travel.discovery.v1`; include event ID, aggregate ID, passenger subject, schema version, occurred-at timestamp and correlation ID; consumers must deduplicate by event ID.

### Versioning, deprecation and isolation

V1 contracts are source-level and pinned dependency versions. V2 should version public HTTP paths and event schemas, add fields compatibly, and reserve breaking field changes for a major version. Deprecations should emit telemetry and remain supported for at least one documented release window. Tool consumers are isolated through dedicated message keys and typed worker states, preventing one specialist's tool transcript from becoming another specialist's implicit contract.

## 8. Security Architecture

### Identity, authentication and authorization

V1 uses the configured `passenger_id` as the authenticated subject and `thread_id` as the session key. There is no token parsing or role hierarchy in the CLI. Authorization is resource-owner based: passenger-facing requirement, booking, ticket and taxi operations filter by passenger ID, and mutation tools verify ownership before changing records. Specialists receive only the state needed for their task.

**Inferred v2 policy:** authenticate with an OIDC access token, derive the immutable passenger subject from validated claims, and define passenger, support and administrator roles. Passenger permissions should be aggregate-level; support access should be purpose-bound and audited; administrative database access must not be exposed through LLM tools.

### Secret and transport protection

OpenAI and Amadeus credentials are environment-injected. Amadeus supports either a supplied bearer token with optional expiry metadata or client credentials exchanged by OAuth. HTTPS protects provider traffic in transit. SQLite at-rest encryption and filesystem access controls are deployment responsibilities in v1. V2 should use a managed secret store, rotate credentials, encrypt database volumes or sensitive columns, and prevent secrets from entering prompts, logs or checkpoints.

### Threat model and mitigations

Primary threats are prompt injection causing unauthorized tool use, cross-passenger access through forged configuration, unsafe side effects, SQL injection, credential leakage, provider payload manipulation and resource exhaustion through long agent loops or date windows. Mitigations present in code include parameterized SQL, passenger ownership predicates, typed tool schemas, specialist scopes, explicit handoffs, approval-capable graph interrupts in sensitive legacy flows, HTTP timeouts/retries, finite graph steps and discovery attempt bounds.

V2 must establish identity outside model-controlled inputs, validate maximum date-window size and result counts, allowlist tool transitions, redact sensitive tool results, add per-passenger quotas, and record immutable audit events for requirement confirmation, booking and cancellation. Provider JSON should be treated as untrusted data and never interpolated into executable instructions. Multi-tenancy is logical by passenger ID in v1; production isolation requires mandatory query scoping and authorization tests.

## 9. Non-Functional Design

### Performance and scalability

Latency is dominated by LLM inference and synchronous Amadeus calls. A multi-day search issues one provider request per calendar day; detailed lookup may repeat this pattern before repricing. Normalization and ranking are linear in offer count plus sorting cost. SQLite operations are small parameterized transactions. The state design avoids giving every specialist every transcript, reducing token cost and model latency.

V1 provides no numerical SLO. **Inferred targets:** cached/read-only local operations should complete under 200 ms at p95; a single model turn under 15 seconds at p95; one-day provider search under 5 seconds at p95; multi-day searches should publish progress and remain under a bounded 30-second request deadline or move to asynchronous execution. Enforce maximum window length, offer count, prompt size and agent steps as backpressure.

### Resilience and availability

The provider adapter retries transient failures, model responses are guarded against emptiness, and discovery has deterministic termination thresholds. Partial degradation should preserve local requirement and history reads when Amadeus is unavailable and allow conversation continuity when a specialist fails. **Inferred availability target:** 99.5% for the conversational API excluding declared third-party outages, with clear dependency-specific status reporting.

SQLite and process memory are scaling limits and single points for continuity. A production topology should externalize checkpoints, use connection-pooled relational storage, isolate long discovery jobs, and cap concurrency per provider credential.

### Observability

V1 exposes console output, state inspection, printable graph messages and LangChain callback/config propagation; `langsmith` is pinned and can receive traces when configured. The Amadeus adapter prints retry diagnostics. Discovery records attempts, filters, outcomes and comments, forming a domain-level audit trail.

**Inferred v2 model:** structured JSON logs must include timestamp, level, service, thread ID, passenger pseudonym, graph node, tool, correlation ID, duration and outcome, never credentials or full sensitive payloads. Metrics should cover turns, tokens, handoffs, tool latency/error rate, Amadeus status codes, offers returned, discovery success ratio, DB transaction latency and graph step exhaustion. Distributed traces should span supervisor, worker, model, provider and database calls. Alerts should cover elevated provider failures, booking persistence failures, checkpoint errors, runaway loop rate and latency-budget breaches.

## 10. Configuration and Environment Design

The runtime consumes `OPENAI_API_KEY` through the OpenAI client and `AMADEUS_ACCESS_TOKEN`/`AMADEUS_BEARER_TOKEN`, optional `AMADEUS_TOKEN_EXPIRES_AT`, or `AMADEUS_CLIENT_ID` and `AMADEUS_CLIENT_SECRET`. `AMADEUS_ENV` is read, while the current client establishes provider endpoints in its constructor. Database filename, model selection, recursion limit and passenger/thread identifiers are code or runtime-config values. Python dependencies are fully pinned.

V1 has no explicit development/staging/production profiles, feature flags, CI/CD or infrastructure as code. **Inferred v2 design:** use environment-specific configuration files containing no secrets; validate configuration at startup; inject database URL, model name, provider environment, maximum search window, retry policy, tracing switch and feature flags. Feature flags should gate autonomous discovery, real booking side effects and alternate models by passenger cohort.

CI should install pinned dependencies, run formatting/static analysis, unit tests for models and tools, graph routing tests, contract fixtures for provider JSON and integration tests against temporary SQLite. CD should build an immutable artifact, run migrations before compatible application rollout, deploy to staging, execute smoke conversations, then canary a small traffic percentage. Blue/green rollout is appropriate for the stateless API layer once checkpoints are external; workers must remain schema-compatible during the rollout. Infrastructure as code should provision compute, managed database, checkpoint store, secret manager, logs, dashboards and egress policy.

## 11. Dependency Graph and Technology Stack

The implementation language is Python. Core orchestration uses LangGraph 0.5.3, `langgraph-prebuilt`, checkpoint packages and LangChain 0.3.26/Core 0.3.68. Model integrations include `langchain-openai` 0.3.28, OpenAI 1.95.1, Google Generative AI and Tongyi adapters; the active primary and specialist paths use `ChatOpenAI`. Pydantic 2.11.7 defines domain and tool schemas. Requests 2.32.4 provides synchronous Amadeus HTTP. SQLite is from the standard library; pandas supports benchmark database preparation; PyMongo/Motor and LangGraph MongoDB checkpoint packages provide an optional checkpoint evolution path. LangSmith 0.4.5 supports tracing.

The internal layering is:

```text
Presentation (main, utils)
  -> Orchestration (graph, graph_setup, specialist graphs)
     -> Agent policies (assistant chains/prompts)
        -> Domain tools and models
           -> Provider adapter (amadeus_api) / Persistence (sqlite queries)
```

Presentation must not query databases. Prompts must not issue SQL or HTTP directly. Graph modules may compose tools but should not interpret provider JSON. Provider code may depend on normalized offer models and optionally requirement types, but must not depend on graph routing. Domain models may depend on persistence helpers only where repository functions currently reside; v2 should separate repository concerns from pure models.

Pinned versions make v1 reproducible. Upgrades should be grouped by boundary, with LangGraph/LangChain upgraded together and validated against graph snapshots and tool schemas. Breaking framework changes require an adapter branch, replay tests and a major application version if externally visible behavior changes. Provider API fixtures in `docs/` serve as contract examples for airport, nearest-airport, offer-search and offer-price responses.

## 12. Failure Analysis

### Model and orchestration failures

OpenAI unavailability blocks semantic turns; empty output receives one retry, but prolonged failure ends useful progress. Invalid tool selection is corrected through a tool message. A runaway agent is bounded by remaining steps and graph recursion limit. Checkpoint loss removes conversational continuity but does not delete committed SQLite aggregates. **Inferred recovery:** retry safe model reads with jitter, preserve the last committed graph state externally, and offer the user a resumable dependency-error response. Target RTO is 15 minutes and RPO is one accepted message after external checkpoints are adopted.

### Amadeus and network failures

Amadeus is a single external dependency for live inventory. Rate limits, 5xx and network failures receive bounded retries; a persistent outage blocks search, details and airport discovery. Existing requirements, bookings and prior discoveries remain available. Pricing changes between search and selection are expected and mitigated by repricing. **Inferred recovery:** circuit break after a threshold, retain no long-lived inventory cache, and ask the user to retry. RTO follows provider recovery; RPO is zero for application records because searches are read-only until booking persistence.

### SQLite failures

The local file is a single persistence point. Lock contention, corruption, missing schema or disk loss can block all durable operations. Individual commits limit partial writes, and discovery attempts are committed as one batch. There is no encoded backup job. **Inferred production plan:** use a managed relational database with point-in-time recovery, daily verified backups and migration checks. For a local deployment, snapshot the SQLite file while quiesced and test restore. Suggested production RTO is four hours and RPO is 15 minutes; booking records may warrant stricter targets.

### Process and checkpoint failures

Supervisor and worker checkpoints reside in memory, so process exit loses active conversations. Because committed domain records are separate, a restarted user can fetch durable requirements and history but not resume the exact unfinished turn. External checkpoint storage removes this single point. Worker transcript and supervisor state must be committed atomically per graph step to avoid replay ambiguity.

### Data and side-effect failures

A database failure after provider repricing but before booking insert produces no durable booking, which is safe but requires a retry. A lost acknowledgement after insert risks a duplicate booking on retry because v1 generates a new UUID. V2 must use client operation IDs and uniqueness constraints. Discovery persistence failure should retain the in-memory session and allow explicit retry. Restore procedures must validate ownership indexes, JSON parseability and referential links from bookings/discovery sessions to requirements.

## 13. Versioning and Evolution Strategy

The application should adopt semantic versioning: patch releases preserve contracts and fix behavior, minor releases add compatible tools, state fields or endpoints, and major releases change message/state/API semantics. V1 currently relies on source-level contracts; `output_specs.md`, typed states, Pydantic schemas and provider fixtures form the baseline for compatibility tests.

Public HTTP APIs introduced in v2 should use a `/v1` path and additive response evolution. Internal tool schema changes must remain readable by the currently deployed prompt and checkpoint population. State migrations must supply defaults for new fields and retain reducers for old checkpoints. Event schemas, if introduced, must carry an independent integer version and allow consumers to ignore unknown fields.

Database migration governance should use ordered, reviewed migrations with forward and rollback/repair procedures. Expand/contract changes are required: add nullable columns or new tables, deploy dual-read/write code where necessary, backfill, validate, switch reads, then remove old structures in a later major release. JSON aggregate changes require a `schema_version` field and deserializers for supported prior versions.

Contract testing should cover Pydantic validation, tool JSON schemas, LangGraph routing from representative states, SQLite schema queries, and recorded Amadeus fixtures. Backward compatibility is enforced in CI by replaying saved conversations/checkpoints and deserializing previous requirements, offers and discovery snapshots. **Inferred deprecation policy:** announce deprecations in release notes and telemetry, support them for at least two minor releases or 90 days, and remove only in a major release.

## 14. Formal Consistency and Invariants

### System-wide invariants

Every graph invocation must have a stable `thread_id`, and every passenger-owned operation must have a trusted `passenger_id`. Only declared graph edges and `Command` destinations may transfer control. A specialist must either produce a user-facing response, execute a valid tool call, or explicitly hand control back. These invariants are enforced through graph topology, typed handoff tools, dialog-stack reducers and assistant retry behavior; routing tests should verify every supported tool name and dialog mode.

### Data invariants

Every persisted requirement has a unique requirement ID, owning passenger, priority-one departure airport, priority-one arrival airport and valid priority-one time window. Airport values are strings; time windows contain ordered datetimes; budget ranges have lower values below upper values. Every discovery attempt has a unique deal ID, belongs to one session and requirement, and compares against the session's immutable baseline. Every booking has an owning passenger, source requirement, priced offer payload and creation identity. Pydantic validation, tool preconditions, database constraints and JSON round-trip tests must jointly enforce these rules.

### Transaction invariants

A requirement sync, booking insert, taxi mutation or ticket mutation is all-or-nothing within one SQLite transaction. A discovery session's attempt rows are committed together. A model message is not evidence that a side effect succeeded; only a successful tool result may be surfaced as completion. V2 idempotency keys must ensure retrying a confirmed mutation creates at most one durable outcome.

### Security invariants

Credentials never enter prompts, graph messages, persisted domain JSON or logs. Passenger identity is never accepted from model-generated tool arguments. Reads and mutations of passenger resources must include the trusted subject in their authorization predicate. Sensitive mutations require validated ownership and, where applicable, explicit confirmation. Parameterized SQL is mandatory. These invariants require authorization unit tests with cross-passenger fixtures and log-redaction tests.

### Operational invariants

All external calls have bounded time, retry count and result volume. Agent workflows have bounded graph steps; discovery terminates according to its success/failure threshold. Provider offers are repriced before booking because search results are not durable price guarantees. Every durable mutation emits a correlation-aware audit record in v2. Backups and checkpoints are considered valid only after automated restore/replay verification. Monitoring must alert when any bound is exceeded or when an operation reports success without a corresponding durable record.
