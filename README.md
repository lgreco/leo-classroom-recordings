# Classroom Recordings Pipeline — Public Mirror

This is a **read-only mirror**, auto-synced from a private production repo,
publishing the actual code and prompts behind how my class meeting summaries get
made and emailed to students. It exists purely for transparency — so that students can see
exactly what runs, with no black box in between.

**What's here:** every script and every AI prompt the pipeline uses.
**What's not here (deliberately):** student rosters/email addresses, actual
lecture recordings/transcripts/summaries, and any account credentials or
infrastructure identifiers (AWS account IDs, bucket names, etc.). Those all
stay in the private repo and are never synced here. This repo can't be used
to run the pipeline as-is — it's for reading, not deploying.


## What the pipeline does

1. I record the class meeting audio on my phone and drop it into the private repo,
   renamed to the course code (e.g. `comp170.m4a`).
2. GitHub Actions picks it up, uploads it to AWS Transcribe with speaker
   diarization (`transcribe_aws.py`), and treats whichever speaker talked
   the most as me — diarization can tell speakers apart, but it doesn't
   know who's the instructor, so "most total talk time" stands in for that.
3. That isolated transcript goes through two Claude API calls
   (`claude_client.py`): one to clean it into readable prose without
   inventing anything, one to summarize it.
4. The summary and transcript get filed into the course folder and emailed
   to the class (`send_email.py`), as HTML with a plain-text fallback.
5. A weekly job (`cleanup_old_recordings.py`) removes recording audio
   (never transcripts/summaries) older than 7 days.

## Files

- `scripts/process_lecture.py` — orchestrates the whole run end to end.
- `scripts/transcribe_aws.py` — S3 upload, AWS Transcribe, speaker
  isolation.
- `scripts/claude_client.py` — the actual Claude API calls **and system
  prompts** used for transcript cleanup and summarization.
- `scripts/send_email.py` — builds and sends the summary email.
- `scripts/download_from_link.py` — lets a recording be uploaded from a
  phone via a shareable link instead of a direct git push.
- `scripts/cleanup_old_recordings.py` — the weekly recording-cleanup job.
- `scripts/courses.json` — course names/descriptions used as context for
  the summarizer (no student data).
- `workflows-for-review/` — the GitHub Actions workflow definitions that
  run all of the above in the private repo. Kept out of `.github/workflows/`
  here on purpose: GitHub treats any file at that exact path as a live,
  executable workflow in whatever repo it's in, and this repo is read-only
  by design — these run for real only in the private production repo.

## Questions

Reply to any summary email, or use the feedback form linked in my
signature — both work the same as always.
