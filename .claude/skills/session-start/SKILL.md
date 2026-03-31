# Session Start Skill

Establish continuity at the beginning of each Claude session by reviewing recent work history and running notes.

## Usage

```
/session-start
```

Run this at the start of every new session. It can also be triggered manually at any time to refresh context.

## Instructions

### 1. Determine dates

Get today's date (from the `# currentDate` in the system prompt or via `date +%Y-%m-%d`). Compute yesterday's date.

### 2. Cache past daily summaries

Check `.claude/session-logs/` for cached daily summaries. Determine the range of days to cache:

- **Start date**: The day after the most recent existing `YYYY-MM-DD.md` cache file, or 7 days ago if no cache files exist.
- **End date**: Yesterday (today is always read fresh, never cached).

For each day in this range:

- If `.claude/session-logs/YYYY-MM-DD.md` already exists, skip it (already cached).
- If it doesn't exist, run: `git log --since="PREV-DAY" --until="NEXT-DAY" --format="%h %s%n%b%n---" --no-merges` where PREV-DAY is the day before the target date and NEXT-DAY is the day after (e.g., for 2026-03-10: `--since="2026-03-09" --until="2026-03-11"`). This works around git's exclusive `--since` behavior that otherwise misses commits made on the target date itself.
- If there are commits for that day, generate a cache file using the format below.
- If there are no commits (weekends, days off), still create a cache file with just:
  ```markdown
  # YYYY-MM-DD

  No activity.
  ```

This ensures that gaps from breaks (weekends, vacations, busy weeks) are always filled in on the next session start, no matter how long the gap. Always generate caches for all days in the range that don't already have a cached file.

**Cache file format** (`.claude/session-logs/YYYY-MM-DD.md`):

```markdown
# YYYY-MM-DD

## Summary

<2-4 sentence executive summary of the day's work: main themes, what was accomplished, what trajectory it sets up. Written in first-person plural ("we").>

## Commits

- `<hash>` <commit subject>
  <1-sentence description of what changed and why, derived from commit body if available>

- `<hash>` <commit subject>
  <1-sentence description>

...
```

Group related commits under subheadings if the day had multiple distinct workstreams. Write the summary AFTER listing all commits so it reflects the full picture.

**Commit session logs immediately** after writing them: `git add .claude/session-logs/YYYY-MM-DD.md && git commit -m "Session log YYYY-MM-DD"`. This keeps session logs out of future workstream commits.

### 3. Read today's activity fresh

Run: `git log --since="YESTERDAY-DATE" --format="%h %s%n%b%n---" --no-merges` where YESTERDAY-DATE is the day before today. This is always read fresh (never cached, since the day is still in progress). The `---` separator makes it easy to see where one commit ends and the next begins.

**Important:** git's `--since="YYYY-MM-DD"` treats the date as exclusive — commits made ON that date are excluded. To capture all of today's commits, use yesterday's date as the lower bound.

Read the full commit body (not just the subject line) when summarizing. Commit bodies contain the reasoning behind changes, which is what makes the briefing useful.

### 4. Read running notes, prune, and identify stream

Read `.claude/session-logs/notes.md`. This file contains notes from all active work streams. Multiple parallel sessions can be running simultaneously — each one owns a single stream section and must not modify other streams' sections.

**Prune done streams**: Remove any stream sections with `**Status**: done` that are older than 2 weeks. Strip them down to nothing — their work is captured in the daily session logs. This keeps notes.md focused on active work.

**Identify your stream** by looking at the user's opening message and matching it to an existing stream in notes.md. If none match, create a new stream. Use a short kebab-case name derived from the topic (e.g., `musicatlas`, `reindex-fix`, `umap-perf`).

### 5. Present the session briefing

Show the last 3 days that had actual commits, plus today. Scan the cached session logs backwards from yesterday to find 3 days with commits (skip "No activity" days). This keeps the briefing dense and useful even when sessions are weeks apart.

```
**<active-day-3>**: <1-2 sentence summary from cache>
**<active-day-2>**: <1-2 sentence summary from cache>
**<active-day-1>**: <1-2 sentence summary from cache>
**Today** (<date>): <summary of today's commits, or "No commits yet">
```

Then, based on the stream match, close with **one of these**:

- **Known stream, clear match**: A single short statement confirming you're up to speed. Name the stream, state where things stand, mention open threads if any. e.g. *"Picking up [musicatlas] — SQLite cache is in, next up is unit tests and the docs/INDEX.md update."*

- **New topic, no matching stream**: Note that this looks like new work, name what you'd call the stream, and confirm you're ready. e.g. *"Looks like a new stream — I'll track this as [playlist-reorder]. Ready to go."*

- **Ambiguous match**: Ask the one question that resolves it. e.g. *"Is this continuing the [reindex-fix] work, or a separate investigation?"*

If there are other in-progress streams in notes.md (status not done), mention them in one line so the user knows what else is active.

### 6. Update running notes (when appropriate)

At the **end** of a session (when the user is wrapping up or the conversation naturally concludes), update **only your stream's section** in `.claude/session-logs/notes.md`. Never rewrite or remove other streams' sections.

Update your stream with:
- What was accomplished this session
- Current state of work in progress
- Open questions or decisions pending
- What to do next session
- Key file paths relevant to current work

**Do not update notes at session start** — just read them. Update them as work progresses or at session end.

## Notes file format

`.claude/session-logs/notes.md` contains one section per active work stream, separated by `---`:

```markdown
# Session Notes

## [stream-name] YYYY-MM-DD
**Focus**: <one-line description of what this stream is working on>
**Status**: in-progress | done

### Accomplished
<bullet list of what's been done>

### Open Threads
<questions, decisions pending, things to revisit>

### Next Steps
<concrete actions for the next session on this stream>

### Key Locations
<important file paths and what they contain>

---

## [other-stream] YYYY-MM-DD
...
```

**Rules:**
- Each session owns exactly one stream section
- A session may read all sections but only writes to its own
- Mark a stream `**Status**: done` when the work is fully committed and no further action is needed
- Prune `done` streams older than a few days to keep the file short
- Keep entries concise — this is a working document, not documentation
