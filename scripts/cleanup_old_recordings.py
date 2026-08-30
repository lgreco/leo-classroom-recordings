"""Deletes committed lecture recordings older than 7 days to bound the
repo's Git LFS storage growth. Transcripts and summaries are left in place;
only the `*-recording.<ext>` audio files are removed. Run by the scheduled
cleanup-recordings.yml workflow.

Why this exists at all: a single lecture recording (50-75 minutes of
audio) is a real chunk of storage, and across three courses over a full
term that adds up fast in a git repo, even with Git LFS. I don't need to
keep the raw audio around indefinitely once it's been transcribed and
summarized — the transcript and summary are the parts with lasting value
— so this script runs on a weekly cron and prunes anything past a week
old. It never touches the transcript or summary files, only the audio.
"""

import glob
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
COURSES_PATH = os.path.join(SCRIPT_DIR, "courses.json")

# Same timezone choice as process_lecture.py, and for the same reason:
# I want "how old is this recording" measured in the timezone I actually
# teach in, not whatever timezone the GitHub Actions runner happens to
# default to (UTC).
LECTURE_TIMEZONE = ZoneInfo("America/Chicago")
MAX_AGE_DAYS = 7

# Matches filenames like "2026-08-28-recording.m4a" and captures just the
# YYYY-MM-DD date prefix. Deliberately anchored to the exact
# "-recording." naming convention process_lecture.py uses, so this never
# accidentally matches a transcript or summary file.
DATE_PREFIX_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-recording\.")


def run_git(*args):
    """Run a git command in the repo root and raise if it fails — same
    helper as in process_lecture.py, kept as a local copy here so this
    script has no import dependency on that one (they're invoked by two
    completely separate workflows and I'd rather each stand alone).
    """
    subprocess.run(["git", *args], cwd=REPO_ROOT, check=True)


def find_stale_recordings():
    """Scan every course's folder for `*-recording.*` files and return the
    full paths of any whose embedded date is older than MAX_AGE_DAYS.

    The key design choice here: "how old is this recording" is
    determined entirely by parsing the YYYY-MM-DD date out of the
    filename itself — never from git commit history or file modification
    times. That matters because git history/mtimes aren't reliable for
    this: a shallow clone, a rebase, or just how the repo happened to get
    checked out on a given runner could all make "when was this file
    committed" disagree with "what date was this lecture actually
    recorded". The filename is the one thing that's stable no matter how
    the repo got here.
    """
    with open(COURSES_PATH, encoding="utf-8") as f:
        courses = json.load(f)

    cutoff = datetime.now(LECTURE_TIMEZONE) - timedelta(days=MAX_AGE_DAYS)
    stale = []
    for course in courses.values():
        folder = os.path.join(REPO_ROOT, course["folder"])
        for path in glob.glob(os.path.join(folder, "*-recording.*")):
            match = DATE_PREFIX_RE.match(os.path.basename(path))
            if not match:
                # Doesn't match the expected naming pattern at all —
                # skip it rather than guess. I'd rather leave a
                # weirdly-named file alone than accidentally delete
                # something that isn't actually a dated lecture recording.
                continue
            recording_date = datetime.strptime(match.group(1), "%Y-%m-%d").replace(
                tzinfo=LECTURE_TIMEZONE
            )
            if recording_date < cutoff:
                stale.append(path)
    return stale


def main():
    """Entry point for the weekly cleanup workflow: find anything stale,
    `git rm` it, and push a single commit removing all of it at once.

    One commit for the whole batch (rather than one commit per file)
    keeps the repo history readable — a week's worth of expired
    recordings across three courses shows up as one clear "removed N
    recordings" commit instead of a pile of near-identical ones.
    """
    stale = find_stale_recordings()
    if not stale:
        print("No recordings older than 7 days found.")
        return

    for path in stale:
        print(f"Removing {os.path.relpath(path, REPO_ROOT)}")
        run_git("rm", os.path.relpath(path, REPO_ROOT))

    run_git("commit", "-m", f"Remove {len(stale)} recording(s) older than {MAX_AGE_DAYS} days")
    run_git("push")


if __name__ == "__main__":
    sys.exit(main())
