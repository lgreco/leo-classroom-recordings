"""Re-send an already-generated lecture summary.

For the case where process_lecture.py got all the way through
transcription/summarization/commit but then failed on the email step
(e.g. a transient Gmail SMTP rejection from a GitHub Actions runner IP --
see send_email.py's SMTP_RETRY_ATTEMPTS comment) -- the summary is
already sitting in the course folder, so this just re-sends it without
redoing the transcribe/summarize/commit work.

Invoked manually via the "Resend Lecture Email" workflow_dispatch
workflow, which is the only place GMAIL_ADDRESS / GMAIL_APP_PASSWORD are
available outside of the main pipeline run.
"""

import argparse
import os
import re

import process_lecture
import send_email


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("course_code")
    parser.add_argument("date_str", help="YYYY-MM-DD, matching the existing <date>-summary.md filename")
    args = parser.parse_args()

    courses = process_lecture.load_courses()
    course = courses[args.course_code]
    summary_path = os.path.join(
        process_lecture.REPO_ROOT, course["folder"], f"{args.date_str}-summary.md"
    )
    with open(summary_path, encoding="utf-8") as f:
        summary_markdown = f.read()

    email_list_path = os.path.join(process_lecture.REPO_ROOT, course["email_list"])
    recipients = send_email.read_recipient_list(email_list_path)
    course_label = re.sub(r"^([a-z]+)(\d+)$", r"\1 \2", args.course_code).upper()

    send_email.send_summary_email(
        sender_address=os.environ["GMAIL_ADDRESS"],
        app_password=os.environ["GMAIL_APP_PASSWORD"],
        recipients=recipients,
        course_label=course_label,
        date_str=args.date_str,
        summary_markdown=summary_markdown,
        reply_to=process_lecture.REPLY_TO_ADDRESS,
    )


if __name__ == "__main__":
    main()
