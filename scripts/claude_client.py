"""Thin wrapper around the Anthropic Claude API for the two lecture-processing
steps: cleaning up a diarization-isolated raw transcript, and summarizing it
with course context.

I'm deliberately keeping this as two separate API calls rather than one
combined "clean and summarize" call. Splitting them means the cleanup
step's job is narrowly "reproduce what was actually said, just tidied
up" — with an explicit instruction not to invent anything — while the
summarization step's job is "condense what's already been verified as
faithful". Mixing those two jobs into one prompt makes it easier for the
model to blend cleanup and summarization together and start paraphrasing
too early, before I have a clean, trustworthy transcript to fall back on
if I ever need to check exactly what I said.
"""

import anthropic

# claude-opus-5 is the model this project defaults to for both calls —
# see planning_summary.md for the reasoning (this pipeline runs
# unattended in GitHub Actions, so there's no live Claude Code session to
# lean on; it has to be a direct API call).
MODEL = "claude-opus-5"

# Generous ceiling so a long lecture's cleaned transcript or summary
# never gets cut off mid-sentence. A 50-75 minute lecture, once cleaned
# into prose, comfortably fits well under this.
MAX_TOKENS = 32000

# System prompt for the first Claude call: turn the raw, diarization-
# isolated transcript into clean, readable prose. The strongest
# constraint here is "do not add, invent, or infer content" — the whole
# point of this step is trustworthy cleanup, not creative rewriting, so
# I'd rather the model leave an honest [inaudible] marker on a garbled
# passage than confidently guess at what I probably said.
_CLEAN_SYSTEM_PROMPT = """\
You clean up raw speech-to-text transcripts of college lectures. The input is
the isolated speech of a single speaker (the instructor), extracted from a
multi-speaker classroom recording using automated speaker diarization. The
isolation is imperfect: it may contain stray words from other speakers,
dropped words, missing punctuation, filler words, and false starts.

Rewrite it into clean, readable prose that preserves the instructor's actual
content, meaning, and structure as closely as possible:
- Fix punctuation, capitalization, and obvious transcription artifacts.
- Remove filler words (um, uh, you know) and false starts/self-corrections,
  keeping only the corrected version of a restated thought.
- Do NOT add, invent, or infer content that isn't supported by the raw text.
- If a passage is too garbled or fragmentary to confidently reconstruct,
  leave a bracketed marker like [inaudible] rather than guessing.
- Preserve the original order and all substantive content — this is a
  cleanup pass, not a summary.

Output only the cleaned transcript text, with no preamble or commentary.
"""

# System prompt for the second Claude call: summarize the already-cleaned
# transcript into something a student who attended (or missed) the
# lecture can actually use. This one gets the course name/description as
# extra context, fed in per-call rather than baked into this system
# prompt, so the same prompt works for all three courses.
_SUMMARY_SYSTEM_PROMPT = """\
You write concise, well-organized Markdown summaries of college lecture
transcripts for the instructor's own records and for distribution to
students who attended. You will be given the course name and description
for context, followed by a cleaned lecture transcript.

Write a Markdown summary with:
- A one-paragraph overview of what the class meeting covered.
- A bulleted list of the main topics/concepts discussed, in the order
  presented.
- Any announcements, deadlines, or assignments mentioned, in their own
  section (omit this section entirely if none were mentioned).

Tone: write like a person, not a report. This goes out to the instructor's
own students, so keep it warm and collaborative, not clinical or
third-person-distant:
- Say "In this class meeting" / "In this meeting", not "This lecture" /
  "The lecture" / "This session".
- Refer to the instructor by first name — "Leo" — not "the instructor" or
  "the professor".
- Prefer plain, direct phrasing over stiff academic phrasing (e.g. "Leo
  covered..." / "Leo walked through..." rather than "The instructor
  presented an overview of...").

Do NOT start with a top-level title/heading (e.g. "# COMP 170 — Lecture
Summary") — whatever this summary is delivered in (an email, a file)
already supplies its own header. Start directly with the overview
paragraph, using "##"-level headings (not "#") for section titles like
"Main Topics Discussed" or "Announcements".

Keep it factual and grounded in the transcript — do not invent topics that
weren't discussed. Output only the Markdown summary, no preamble.
"""


def _run(client, system_prompt, user_content):
    """Shared plumbing for both Claude calls: stream the request and
    concatenate the returned text blocks into a single string.

    Both clean_transcript() and summarize() funnel through here so the
    streaming/response-handling logic only has to be written once.

    Why streaming (`client.messages.stream(...)` + `.get_final_message()`)
    instead of a plain non-streaming `client.messages.create(...)`: a
    full lecture transcript is a lot of text in both directions, and a
    single big non-streaming request risks hitting an HTTP timeout before
    Claude finishes generating. Streaming avoids that even though we
    don't actually care about the individual chunks as they arrive here —
    `get_final_message()` just waits for the stream to finish and hands
    back the assembled response, same shape as a non-streaming call would
    have given us.

    `output_config={"effort": "medium"}` — effort is a dial on how much
    internal reasoning the model does before answering. I picked
    "medium" because transcript cleanup and summarization are
    straightforward text-transformation tasks, not hard multi-step
    reasoning problems, so I don't need to pay for the highest effort
    tier here. This is a tunable knob, not a hard requirement — if I
    ever notice cleanup quality suffering, bumping this to "high" is the
    first thing to try.
    """
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        response = stream.get_final_message()

    # Claude's response can technically contain more than one content
    # block; for plain text output like ours it's effectively always one
    # block, but joining all "text"-typed blocks together is the
    # defensively-correct way to reassemble the full answer rather than
    # assuming response.content[0] is always the whole thing.
    return "".join(block.text for block in response.content if block.type == "text")


def clean_transcript(client, raw_transcript):
    """First Claude call: hand it the raw, speaker-isolated transcript and
    get back cleaned-up prose. See _CLEAN_SYSTEM_PROMPT above for exactly
    what "cleaned up" means here (fix punctuation/filler words, but never
    invent content).
    """
    return _run(client, _CLEAN_SYSTEM_PROMPT, raw_transcript)


def summarize(client, cleaned_transcript, course_name, course_description):
    """Second Claude call: summarize the already-cleaned transcript,
    giving the model the course's name and description as context so the
    summary reads like it knows what class this actually is.

    I build the user_content as a small plain-text preamble (course name
    + description) followed by the full transcript, rather than trying to
    pass the course metadata as some separate structured field — Claude's
    Messages API takes a single content string here, and this format is
    simple enough that the model has no trouble telling "this is context
    about the course" apart from "this is the actual lecture transcript
    to summarize".
    """
    user_content = (
        f"Course: {course_name}\n"
        f"Course description: {course_description}\n\n"
        f"Transcript:\n{cleaned_transcript}"
    )
    return _run(client, _SUMMARY_SYSTEM_PROMPT, user_content)


def make_client():
    """Build an Anthropic client.

    Deliberately a zero-argument constructor call — the Anthropic SDK
    picks up the API key from the ANTHROPIC_API_KEY environment variable
    automatically, which GitHub Actions injects from the repo's
    ANTHROPIC_API_KEY secret at runtime. That means the key itself never
    has to appear anywhere in this codebase.
    """
    return anthropic.Anthropic()
