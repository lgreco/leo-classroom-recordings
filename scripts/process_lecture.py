"""Orchestrates one lecture recording end to end: transcribe, isolate the
instructor's voice, clean up and summarize with Claude, file the results
into the course folder, commit/push, and email the summary.

Invoked once per matched root-level recording by the GitHub Actions
workflow (.github/workflows/process-lecture.yml), which passes the
course code and the path to the audio file as CLI arguments.

I'm writing the comments in this file (and the other scripts in this
folder) more verbosely than I normally would, on purpose — this pipeline
touches AWS, Anthropic, Gmail, and git all in one run, and if something
breaks at 8am before class I want to be able to reread this file cold and
immediately remember why each step exists, not just what it does.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import boto3

import claude_client
import send_email
import transcribe_aws

# SCRIPT_DIR is wherever this file physically lives on disk (the scripts/
# folder), and REPO_ROOT is one level up from that — the root of the
# classroom-recordings repo. I compute both from __file__ rather than
# hardcoding paths so this still works no matter what directory the
# GitHub Actions runner happens to check the repo out into.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
COURSES_PATH = os.path.join(SCRIPT_DIR, "courses.json")

# Summaries are sent from GMAIL_ADDRESS (lgreco@gmail.com), but student
# replies should land in Leo's work inbox. Reply-To doesn't touch sender
# authentication (unlike spoofing From — see send_email.py's docstring),
# so this is just a plain, safe header — not a secret, hence a constant
# here rather than a GitHub Actions secret.
REPLY_TO_ADDRESS = "leo@cs.luc.edu"

# All lecture dates are computed in Central time (America/Chicago), since
# that's where I actually teach. This matters because GitHub Actions
# runners default to UTC — without pinning the timezone here, a lecture
# recorded at, say, 9pm Central could get dated "tomorrow" once converted
# to UTC, which would misfile it relative to when it was actually taught.
LECTURE_TIMEZONE = ZoneInfo("America/Chicago")


def load_courses():
    """Read courses.json and hand back the course-code -> course-info dict.

    courses.json is the single source of truth for how a course code
    (like "comp170") maps to its folder name, the human-readable course
    name and description (which get fed to Claude for summarization
    context), and the path to that course's student email list. Keeping
    this in one JSON file instead of scattering it across scripts means I
    only have to update one place when I add/rename a course next term.
    """
    with open(COURSES_PATH, encoding="utf-8") as f:
        return json.load(f)


def run_git(*args):
    """Run a git command in the repo root and raise if it fails.

    Every git operation in this script goes through this one helper so
    that (a) I never accidentally run git from the wrong working
    directory, and (b) `check=True` means any git failure (a bad merge,
    a push rejected because someone else pushed in between, etc.) raises
    immediately instead of silently limping on with a half-finished
    commit.
    """
    subprocess.run(["git", *args], cwd=REPO_ROOT, check=True)


def commit_and_push(paths, message):
    """Stage the given paths, commit them, and push — unless there's
    nothing to commit.

    The `git diff --cached --quiet` check exists because `git commit`
    would otherwise exit non-zero (and blow up the `check=True` in
    run_git) if, for some reason, staging these paths produced no actual
    changes. That shouldn't normally happen for a freshly processed
    recording, but it's cheap insurance against a confusing crash.
    """
    run_git("add", *paths)
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=REPO_ROOT
    )
    if result.returncode == 0:
        print("Nothing to commit.")
        return
    run_git("commit", "-m", message)
    run_git("push")


def process_recording(course_code, audio_path):
    """Run the full pipeline for one recording: this is the heart of the
    whole project, so I'm walking through it step by step below.

    Given a course code (e.g. "comp170") and the path to the audio file
    that was just pushed to the repo root, this function:

    1. Looks up the course's metadata in courses.json.
    2. Figures out today's date (in Central time) — this becomes the
       filename prefix for the recording, transcript, and summary, since
       I decided the push date *is* the lecture date rather than trying
       to parse a timestamp out of the audio file itself (iPhone export
       paths like AirDrop or the Files app often strip or scramble that
       metadata anyway, so this is more robust in practice).
    3. Uploads the audio to S3 (AWS Transcribe needs the file to live in
       S3 — it can't transcribe a local file directly).
    4. Kicks off an AWS Transcribe batch job with speaker diarization
       turned on, and blocks until it finishes.
    5. Downloads the diarized transcript JSON and isolates just the
       instructor's speech (the "most total talk time" speaker — see
       transcribe_aws.isolate_primary_speaker_transcript for the
       reasoning behind that heuristic).
    6. Sends that raw, isolated transcript to Claude to get cleaned up
       into readable prose (fixing punctuation, dropping filler words,
       etc., without inventing content).
    7. Sends the cleaned transcript to Claude a second time to get a
       course-aware Markdown summary.
    8. Moves the audio into its course folder (via `git mv`, so git sees
       it as a rename rather than a delete+add) and writes the
       transcript and summary files alongside it.
    9. Commits and pushes those three files back to the repo.
    10. Emails the summary to the course's student roster.
    11. Best-effort deletes the temporary S3 objects it created (if that
        cleanup fails for some reason, the S3 bucket's lifecycle rule is
        the backstop that will eventually expire them anyway).
    """
    courses = load_courses()
    if course_code not in courses:
        # Defensive check — in practice the GitHub Actions workflow only
        # calls this script for filenames matching a known course code
        # pattern, so this should never actually trigger. But if I ever
        # add a fourth course and forget to update courses.json, I want
        # a clear error instead of a confusing KeyError three lines down.
        raise ValueError(f"Unknown course code: {course_code}")
    course = courses[course_code]

    date_str = datetime.now(LECTURE_TIMEZONE).strftime("%Y-%m-%d")
    ext = os.path.splitext(audio_path)[1].lstrip(".").lower()
    # If I ever rename a recording with a ".mpa" extension by mistake
    # (easy to do on some phones/exports), treat it as m4a for AWS
    # Transcribe's MediaFormat parameter, which doesn't recognize "mpa".
    media_format = "m4a" if ext == "mpa" else ext  # normalize a stray .mpa rename to m4a

    # These come from GitHub Actions secrets at runtime (see
    # process-lecture.yml) — never hardcoded here, so nothing sensitive
    # ends up committed to the repo.
    aws_region = os.environ["AWS_REGION"]
    s3_bucket = os.environ["AWS_S3_BUCKET"]
    s3 = boto3.client("s3", region_name=aws_region)
    transcribe = boto3.client("transcribe", region_name=aws_region)

    # A short random suffix keeps S3 keys and Transcribe job names unique
    # even if I somehow push two recordings for the same course on the
    # same day (a job name has to be unique per AWS account/region, and
    # AWS Transcribe will reject a duplicate).
    job_id = uuid.uuid4().hex[:8]
    s3_key = f"{course_code}/{date_str}-{job_id}.{ext}"
    print(f"Uploading {audio_path} to s3://{s3_bucket}/{s3_key}")
    media_uri = transcribe_aws.upload_audio(s3, s3_bucket, s3_key, audio_path)

    job_name = f"{course_code}-{date_str}-{job_id}"
    output_key = f"transcribe-output/{job_name}.json"
    print(f"Starting Transcribe job {job_name}")
    # This call blocks (polling every 30s, up to a 45-minute timeout)
    # until the job finishes or fails — see transcribe_aws.py for the
    # polling loop itself. A 50-75 minute lecture typically transcribes
    # well within that window, but if AWS is having a slow day, I'd
    # rather this fail loudly with a clear timeout error than hang the
    # GitHub Actions job forever.
    job = transcribe_aws.run_transcription_job(
        transcribe, job_name, media_uri, media_format, s3_bucket, output_key
    )

    print("Downloading transcript result")
    # Fetched via boto3 (not the job's TranscriptFileUri) because that URI
    # is only a pre-signed public link when AWS picks the output bucket for
    # us. Since we explicitly set OutputBucketName above, TranscriptFileUri
    # is just a plain HTTPS link to our own bucket/key, which correctly has
    # Block Public Access on — an unauthenticated GET against it 403s. We
    # already know the bucket/key we wrote it to, so read it directly.
    transcript_json = transcribe_aws.download_transcript_json(s3, s3_bucket, output_key)
    # This is where the "remove the students' voices" part of the ask
    # actually happens: AWS Transcribe's diarization only labels *which*
    # speaker said each word, it has no idea which speaker is "the
    # instructor" — so we pick the speaker with the most total talking
    # time as a stand-in for "me", on the assumption that in a lecture
    # format I talk far more than any single student does.
    raw_transcript = transcribe_aws.isolate_primary_speaker_transcript(transcript_json)

    claude = claude_client.make_client()
    print("Cleaning transcript with Claude")
    # First Claude call: turn the raw, diarization-isolated transcript
    # (which still has filler words, false starts, and the occasional
    # stray word that leaked in from a nearby student) into clean prose,
    # without letting the model invent anything that wasn't actually said.
    cleaned_transcript = claude_client.clean_transcript(claude, raw_transcript)
    print("Summarizing transcript with Claude")
    # Second Claude call: summarize the now-clean transcript, using the
    # course's name and description (from courses.json) as context so
    # the summary reads like it was written by someone who knows what
    # class this is, not a generic transcript summary.
    summary_markdown = claude_client.summarize(
        claude, cleaned_transcript, course["name"], course["description"]
    )

    course_folder = os.path.join(REPO_ROOT, course["folder"])
    os.makedirs(course_folder, exist_ok=True)

    # Final on-disk layout for this lecture, all inside the course's own
    # subfolder (e.g. comp170f26/):
    #   2026-08-28-recording.m4a
    #   2026-08-28-transcript.txt
    #   2026-08-28-summary.md
    recording_dest = os.path.join(course_folder, f"{date_str}-recording.{ext}")
    transcript_path = os.path.join(course_folder, f"{date_str}-transcript.txt")
    summary_path = os.path.join(course_folder, f"{date_str}-summary.md")

    # `git mv` (not a plain filesystem move) so git records this as a
    # rename of the LFS-tracked audio file rather than deleting the old
    # path and adding a brand new blob — keeps history cleaner and plays
    # nicer with Git LFS bookkeeping.
    run_git("mv", "-f", os.path.relpath(audio_path, REPO_ROOT), os.path.relpath(recording_dest, REPO_ROOT))
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(cleaned_transcript)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_markdown)

    commit_and_push(
        [recording_dest, transcript_path, summary_path],
        f"Process {course_code} recording for {date_str}",
    )

    email_list_path = os.path.join(REPO_ROOT, course["email_list"])
    recipients = send_email.read_recipient_list(email_list_path)
    # Short "COMP 170" style label for the subject/heading, derived from
    # the course code rather than duplicated in courses.json — every
    # course code so far is letters-then-digits ("comp170"), so this just
    # inserts the space and upper-cases it. Falls back to the raw code
    # uppercased if a future course code ever doesn't fit that shape.
    course_label = re.sub(r"^([a-z]+)(\d+)$", r"\1 \2", course_code).upper()
    send_email.send_summary_email(
        sender_address=os.environ["GMAIL_ADDRESS"],
        app_password=os.environ["GMAIL_APP_PASSWORD"],
        recipients=recipients,
        course_label=course_label,
        date_str=date_str,
        summary_markdown=summary_markdown,
        reply_to=REPLY_TO_ADDRESS,
    )

    # Housekeeping: the raw audio and the Transcribe job's JSON output
    # don't need to stick around in S3 once we've pulled everything we
    # need out of them (the audio itself now lives in the repo via Git
    # LFS). This is deliberately best-effort — if it fails for some
    # transient reason, I don't want that to fail the whole pipeline run
    # after the email has already gone out. The S3 bucket also has a
    # lifecycle rule that expires objects after a couple of days as a
    # backstop, in case this cleanup never runs at all.
    try:
        s3.delete_object(Bucket=s3_bucket, Key=s3_key)
        s3.delete_object(Bucket=s3_bucket, Key=output_key)
    except Exception as exc:  # cleanup is best-effort; bucket lifecycle rule is the backstop
        print(f"Warning: failed to clean up S3 objects: {exc}")


def main():
    """CLI entry point: `python process_lecture.py <course_code> <audio_path>`.

    This is exactly how the GitHub Actions workflow invokes this script
    for each matched recording it finds in a push — see
    .github/workflows/process-lecture.yml.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("course_code")
    parser.add_argument("audio_path")
    args = parser.parse_args()
    process_recording(args.course_code, args.audio_path)


if __name__ == "__main__":
    sys.exit(main())
