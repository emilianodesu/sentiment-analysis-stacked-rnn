---
inclusion: always
---

# Spec Driven Development

## Overview

Guide feature development through structured specification phases — project setup, idea honing, research, design, tasks, implementation — producing working documents that serve as the AI's reference throughout. The idea honing and research phases are iterative: you can move between them as questions arise.

## Usage

Use when:
- Starting a new feature or project
- User asks to "spec out" or "plan" a feature before coding
- User wants requirements, design, or implementation plan documents
- Beginning any non-trivial code change that benefits from upfront planning
- User wants to continue, resume, or pick up an existing spec-driven development flow

### Resuming an Existing Spec

**CRITICAL**: When the user asks to continue or resume spec-driven development, you MUST follow the resume flow below. Do NOT start a new spec or recreate files that already exist.

When the user asks to continue spec-driven development (e.g., "continue spec driven development", "resume spec", "pick up where we left off", "next phase"):

1. Scan `.kiro/specs/` for existing spec directories
2. If multiple specs exist, list them and ask the user which one to continue
3. If only one exists, confirm it with the user
4. **Read** all existing files in the spec directory (`rough-idea.md`, `idea-honing.md`, `requirements.md`, `design.md`, `tasks.md`, `research/*`) — you MUST read them, not recreate them
5. Determine the current phase by evaluating these conditions **in order from top to bottom** — stop at the **first match**:
   1. Only `rough-idea.md` exists → resume at Phase 1 (Idea Honing)
   2. `idea-honing.md` exists but `requirements.md` does NOT exist → resume at Iteration Checkpoint or Phase 1/2
   3. `requirements.md` exists but `design.md` does NOT exist → resume at Phase 4 (Design)
   4. `design.md` exists but `tasks.md` does NOT exist → resume at Phase 5 (Tasks)
   5. `tasks.md` exists with unchecked items (`- [ ]`) → resume at Phase 6 (Implementation), starting at the first unchecked task
   6. `tasks.md` exists with ALL items checked (`- [x]`) → spec is complete

   **CRITICAL**: Check conditions in the numbered order above. Do NOT skip to a later condition. For example, if `requirements.md` exists but `design.md` does not, the phase is Design (condition 3) — do NOT jump to Implementation.
6. Present a summary of where things stand and **confirm the next step with the user before proceeding** — you MUST wait for the user's explicit confirmation before starting any work

## Core Concepts

- **Spec Directory**: `.kiro/specs/<feature-name>/` in the project root
- **Phases**: Setup → Idea Honing ↔ Research → Design → Tasks → Implementation
- **Iterative**: Move between idea honing and research as questions arise
- **One Question at a Time**: Never batch questions — ask one, wait, record, repeat
- **Living Documents**: Specs are updated as implementation reveals new information
- **Acceptance Criteria**: Testable conditions written in WHEN/SHALL/IF-THEN format
- **Task Checkboxes**: Implementation items tracked with `- [ ]` / `- [x]`
- **Always Buildable**: After every task, the code must compile/build successfully — no broken intermediate states
- **Always Tested**: Every task must include tests that validate the behavior introduced or changed in that task

## Workflow

### Phase 0: Project Setup

Create the spec directory structure before starting any phase.

**Steps:**
1. Create the directory: `.kiro/specs/<feature-name>/`
2. Create the directory: `.kiro/specs/<feature-name>/research/`
3. Create `.kiro/specs/<feature-name>/rough-idea.md` — save the user's initial idea verbatim
4. Create `.kiro/specs/<feature-name>/idea-honing.md` — with header only, content added during Phase 1

After setup, present exactly these three options:
1. **Start with idea honing** (default) — refine the idea through Q&A
2. **Start with preliminary research** — investigate specific topics first
3. **Provide additional context** — share more details before starting

You **MUST** wait for the user's choice before proceeding. Do NOT begin any phase until the user responds.

### Phase 1: Idea Honing

Refine the rough idea into clear requirements through one-on-one dialogue.

Use `.kiro/specs/<feature-name>/idea-honing.md` to record the conversation.

**CRITICAL: One question at a time with file recording**

For EACH question, follow this exact sequence:

1. **Write the question** to `idea-honing.md` using the format below
2. **Present the question** to the user
3. **STOP and wait** for the user's response
4. When the user responds, **append their answer** to `idea-honing.md` under the question
5. Only THEN formulate and write the next question

**Format in idea-honing.md** (you MUST use this exact format):
```markdown
## Q1: [Question text]
**A:** [User's answer — written AFTER they respond]

## Q2: [Question text]
**A:** [User's answer — written AFTER they respond]
```

**After each user response, you MUST:**
1. Read the current `idea-honing.md`
2. Append `**A:** [their answer]` under the current question
3. Write the updated file
4. Then write the next question as a new `## QN:` section

**Rules:**
- You **MUST** ask only ONE question at a time
- You **MUST** write each Q&A pair to `idea-honing.md` incrementally — question first, answer after user responds
- You **MUST NOT** list multiple questions for the user to answer at once
- You **MUST NOT** pre-populate answers without user input
- You **MUST NOT** write multiple Q&A pairs to the file at once
- You **MAY** suggest possible answers, but **MUST** wait for the user's actual response
- You **SHOULD** ask about: scope, edge cases, user experience, technical constraints, success criteria, non-functional requirements
- You **SHOULD** adapt follow-up questions based on previous answers
- You **MUST** offer to pivot to research if a question arises that needs investigation
- When the conversation reaches a natural conclusion, **ask the user** if they feel idea honing is complete before moving on

### Phase 2: Research

Investigate technologies, existing code, patterns, or constraints that inform the design. This phase is collaborative — the user helps shape what gets researched.

**Starting research:**
1. Identify areas where research is needed based on the idea honing
2. Propose a research plan to the user listing topics to investigate
3. Ask the user for additional topics, specific resources they recommend, and areas where they have existing knowledge
4. Incorporate their suggestions into the plan

**Conducting research:**
- Document findings in separate files under `.kiro/specs/<feature-name>/research/` (e.g., `existing-code.md`, `technologies.md`, `alternatives.md`)
- Include mermaid diagrams when documenting architectures, data flows, or component relationships
- Include links to references and sources
- Periodically check in with the user to share findings, get feedback, and confirm direction

**Completing research:**
- Summarize key findings that will inform the design
- Ask the user if research is sufficient before proceeding
- Offer to return to idea honing if research uncovers new questions
- You **MUST** wait for the user to decide the next step

### Iteration Checkpoint

**MANDATORY** — Before moving to requirements/design, you MUST present an iteration checkpoint.

Present this summary to the user:
1. What was covered in idea honing (key decisions made)
2. What research has been done (if any)
3. Any open questions or unknowns

Then ask the user to choose ONE of:
- **Proceed to requirements** — idea honing and research are sufficient
- **Return to idea honing** — more questions need answering
- **Conduct more research** — investigate specific topics first

You **MUST NOT** proceed to requirements without the user explicitly choosing to proceed. If the user chooses research or idea honing, go to that phase — do NOT proceed to requirements.

### Phase 3: Requirements

After idea honing and research are confirmed complete (via the iteration checkpoint), consolidate findings into structured requirements.

Create `.kiro/specs/<feature-name>/requirements.md` using this exact structure:

```markdown
# Requirements

## Introduction
[1-2 paragraphs: what is being built and why]

## Glossary
- **Term**: Definition
[Only include terms that need clarification for this specific feature]

## Requirements

### Requirement 1
**User Story:** As a [role], I want [capability], so that [benefit].

#### Acceptance Criteria
1. WHEN [condition], THE system SHALL [behavior]
2. IF [condition], THEN THE system SHALL [behavior]
[Each criterion must be independently testable]
```

**CRITICAL format rules for acceptance criteria — MANDATORY, NO EXCEPTIONS:**
- Every single acceptance criterion MUST begin with the word WHEN or IF — no other starting word is allowed
- Every single acceptance criterion MUST contain the word SHALL
- Exact patterns (use these verbatim):
  - `WHEN [trigger], THE system SHALL [observable behavior]`
  - `IF [condition], THEN THE system SHALL [observable behavior]`
- Do NOT write acceptance criteria as plain sentences, bullet points, or any other format
- If an acceptance criterion does not contain both WHEN/IF and SHALL, it is WRONG — rewrite it

**Guidelines:**
- Consolidate all Q&A from `idea-honing.md` into structured requirements
- You **MUST** confirm requirements with the user before proceeding to design
- Keep requirements focused on WHAT, not HOW
- Each requirement gets a user story + numbered acceptance criteria
- Use SHALL for mandatory behavior, SHOULD for preferred, MAY for optional

### Phase 4: Design

After requirements are confirmed, produce the technical design.

Create `.kiro/specs/<feature-name>/design.md`:

```markdown
# Design

## Overview
[How the system will satisfy the requirements — architecture summary]

## Components
### Component Name
- **Purpose**: What it does
- **Interface**: Key methods/APIs
- **Dependencies**: What it relies on

## Data Models
[If applicable — key structures, schemas, types]

## Error Handling
[How failures are managed]

## Constraints
[Technical limitations, performance targets, compatibility requirements]
```

**Guidelines:**
- You **MUST** reference which requirements each design decision satisfies
- You **MUST** confirm design with the user before proceeding to tasks
- Adapt sections to what's relevant — skip sections that don't apply
- Include diagrams (mermaid) for complex flows
- Focus on decisions that affect implementation, not obvious details

### Phase 5: Tasks

Break the design into an ordered implementation plan.

Create `.kiro/specs/<feature-name>/tasks.md`:

```markdown
# Tasks

## Task 1: [Short description]
**Requirements:** REQ-1, REQ-2
- [ ] Subtask description
- [ ] Subtask description
- [ ] Add/update tests that validate this task's behavior
- [ ] Verify the build passes

## Task 2: [Short description]
**Requirements:** REQ-3
- [ ] Subtask description
- [ ] Add/update tests that validate this task's behavior
- [ ] Verify the build passes
```

**Guidelines:**
- Order tasks by dependency — earlier tasks unblock later ones
- Each task should be completable in one focused session
- Link tasks back to requirements they satisfy
- **MANDATORY SUBTASKS** — Every single task MUST end with these two subtasks (copy them verbatim):
  - `- [ ] Add/update tests that validate this task's behavior`
  - `- [ ] Verify the build passes`
- These two subtasks are required for EVERY task with NO exceptions — if a task is missing either one, add it
- Mark tasks complete (`- [x]`) as implementation progresses

### Phase 6: Implementation

After tasks are confirmed, begin coding.

**CRITICAL — Resuming Implementation**: When resuming at this phase, you MUST:
1. Present a summary showing which tasks are complete and which is next
2. **STOP and wait** for the user's explicit confirmation before writing ANY code or running ANY build — do NOT proceed until the user says yes

**WARNING**: Starting implementation without user confirmation is a critical failure. You MUST present the summary AND receive an explicit "yes" or confirmation from the user before doing any implementation work.

For each task, you MUST follow each step and DO NOT skip:
1. Read the relevant requirement + design sections
2. Implement the subtasks
3. Write or update tests that validate the task's behavior
4. Verify the build passes — do not proceed to the next step if it fails
5. Mark checkboxes as complete in tasks.md
6. Update specs if implementation reveals needed changes
7. **Git Commit** — Commit all changes (including new files) with a conventional commit message referencing the task number (e.g., `feat(spec): Implement task N — <short description>`)

## Quick Reference

| Phase | Output | Key Question |
|-------|--------|-------------|
| Setup | Directory structure | Where do artifacts go? |
| Idea Honing | idea-honing.md | What exactly do we want? |
| Research | research/*.md | What do we need to know? |
| Requirements | requirements.md | What are we building? |
| Design | design.md | How will we build it? |
| Tasks | tasks.md | In what order? |
| Implement | Code | Build it, check it off |

## Common Mistakes

**Batching questions**: Asking multiple questions at once during idea honing.
- Fix: One question at a time. Wait for the answer. Record it. Then ask the next.

**Skipping user confirmation**: Generating all files at once without checking.
- Fix: Pause at every phase boundary for user review.

**Skipping research**: Jumping straight to design without investigating unknowns.
- Fix: Propose a research plan. Even a brief one surfaces important constraints.

**Over-specifying**: Writing design details in requirements, or code in design.
- Fix: Requirements = what, Design = how, Tasks = when.

**Stale specs**: Implementing changes without updating the spec files.
- Fix: Update specs when implementation diverges from plan.

**Skipping tests**: Implementing a task without adding tests for the new behavior.
- Fix: Every task must include tests. Do not mark a task complete without them.

**Broken intermediate state**: Moving to the next task while the build is failing.
- Fix: Verify the build passes after every task before proceeding.

## CRITICAL Implementation Rules — NEVER VIOLATE

1. **NEVER use `python -c` to run any code** — it corrupts in the terminal
2. **ALWAYS write test code to a file** and run with `python -m pytest` or `python script.py`
3. **NEVER create temporary files then delete them** — write proper tests in `tests/`
4. If you need to verify something quickly, write it in `tests/test_<module>.py` and run pytest
5. NEVER create temporary files for verification — use pytest exclusively
6. NEVER delete files as a separate step — if a temp file was created by mistake, ignore it and move on