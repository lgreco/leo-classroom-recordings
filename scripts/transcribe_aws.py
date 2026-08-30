"""AWS Transcribe: upload audio, run a diarized batch job, and isolate the
instructor's speech from the raw result.

This is the "speech to text" and "isolate my voice" half of the pipeline.
AWS Transcribe doesn't have a way to say "only transcribe this specific
person's voice" up front — it can only tell speakers apart from each
other (diarization), not identify which one is "the instructor". So the
approach here is: transcribe everyone, then figure out after the fact
which speaker talked the most, and treat that speaker as me.
"""

import json
import time

import boto3

# How often to check whether the Transcribe job is done, and how long to
# wait before giving up entirely. I picked 30 seconds because polling
# more often than that just burns API calls for no real benefit — a
# 50-75 minute lecture is going to take AWS a while to process no matter
# how often I ask "are you done yet?". 45 minutes as a hard timeout is
# generous: I'd rather the pipeline fail loudly and let me know something
# is wrong than hang a GitHub Actions job indefinitely.
POLL_INTERVAL_SECONDS = 30
JOB_TIMEOUT_SECONDS = 45 * 60

# AWS Transcribe needs an upper bound on how many distinct speakers it
# should try to distinguish. 6 is comfortably more than "me + a handful
# of students who happen to speak up during a given lecture" — if a
# class discussion somehow has more than 6 distinct speakers, the extra
# voices just get folded into the nearest existing speaker label, which
# doesn't hurt the "who talked the most" heuristic below.
MAX_SPEAKER_LABELS = 6


def upload_audio(s3_client, bucket, key, local_path):
    """Upload the local recording to S3 and return its s3:// URI.

    AWS Transcribe can't read a file straight off the GitHub Actions
    runner's disk — it only accepts media that already lives in S3. So
    this is always the first real step: get the audio somewhere AWS's
    transcription service can actually see it.
    """
    s3_client.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def run_transcription_job(transcribe_client, job_name, media_uri, media_format,
                           output_bucket, output_key, language_code="en-US"):
    """Start a batch Transcribe job with speaker diarization on, then
    block (polling) until it completes, fails, or times out.

    A few things worth calling out about the job settings:

    - `ShowSpeakerLabels: True` + `MaxSpeakerLabels` is what turns on
      diarization — without it, AWS Transcribe just returns one big
      undifferentiated transcript with no notion of "who said what",
      which would make it impossible to separate my voice from the
      students'.
    - I'm writing the output to a specific `output_key` in our own S3
      bucket (rather than letting AWS manage the output location)
      because it makes cleanup straightforward: I know exactly which
      object to delete afterwards.

    This function returns the full `TranscriptionJob` dict from AWS on
    success, so the caller can pull `Transcript.TranscriptFileUri` out of
    it. On failure or timeout, it raises instead of returning something
    the caller would have to remember to check — I'd rather see the
    pipeline crash with a clear error in the Actions log than have it
    silently continue with no transcript.
    """
    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": media_uri},
        MediaFormat=media_format,
        LanguageCode=language_code,
        OutputBucketName=output_bucket,
        OutputKey=output_key,
        Settings={
            "ShowSpeakerLabels": True,
            "MaxSpeakerLabels": MAX_SPEAKER_LABELS,
        },
    )

    deadline = time.time() + JOB_TIMEOUT_SECONDS
    while True:
        response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        status = response["TranscriptionJob"]["TranscriptionJobStatus"]
        if status == "COMPLETED":
            return response["TranscriptionJob"]
        if status == "FAILED":
            # Surface AWS's own failure reason (bad audio format, corrupt
            # file, etc.) rather than a generic "it failed" — this is
            # usually enough to tell me what went wrong without having
            # to go dig through the AWS console.
            reason = response["TranscriptionJob"].get("FailureReason", "unknown reason")
            raise RuntimeError(f"Transcribe job {job_name} failed: {reason}")
        if time.time() > deadline:
            # Distinct exception type from the FAILED case above, on
            # purpose — a timeout doesn't necessarily mean anything is
            # actually wrong with the job, just that it's taking longer
            # than expected (or AWS is having a slow day). Worth being
            # able to tell the two apart at a glance in the logs.
            raise TimeoutError(
                f"Transcribe job {job_name} did not complete within "
                f"{JOB_TIMEOUT_SECONDS // 60} minutes (last status: {status})"
            )
        time.sleep(POLL_INTERVAL_SECONDS)


def download_transcript_json(s3_client, output_bucket, output_key):
    """Fetch the Transcribe job's output JSON straight from S3, using
    boto3's S3 client.

    This has to go through boto3 (authenticated) rather than a plain HTTPS
    GET against the job's `Transcript.TranscriptFileUri`: that URI is only
    a pre-signed, temporarily-public link when AWS picks the output
    bucket for you. Since we explicitly pass `OutputBucketName` when
    starting the job (so cleanup knows exactly which object to delete),
    `TranscriptFileUri` comes back as a plain HTTPS link to our own
    bucket/key — and that bucket correctly has Block Public Access on, so
    an unauthenticated GET against it 403s. We already know the
    bucket/key we told Transcribe to write to, so just read it directly.
    """
    obj = s3_client.get_object(Bucket=output_bucket, Key=output_key)
    return json.loads(obj["Body"].read())


def isolate_primary_speaker_transcript(transcript_json):
    """Return the raw transcript text for whichever speaker talked the most.

    AWS Transcribe's diarized output splits speaking-time attribution
    (`results.speaker_labels.segments`) from word-level text
    (`results.items[]`, each carrying its own `speaker_label`). We sum
    segment durations per speaker to find the dominant talker (assumed to
    be the instructor), then walk items in order and keep only that
    speaker's tokens, respecting punctuation attachment.

    Why "most total talk time" and not something fancier like voice
    enrollment or a fixed "speaker 0 is always me" rule: AWS Transcribe's
    batch API has no voice-ID/enrollment feature, so there's no way to
    tell it in advance what my voice sounds like. Diarization only
    clusters speakers relative to each other within one recording — it
    doesn't attach a name or role to any of them, and which numeric label
    ends up being "speaker_0" isn't guaranteed to be consistent run to
    run. But in a lecture format, I'm talking for the overwhelming
    majority of the class period compared to any single student who asks
    a question or answers one, so "most total seconds of talk time" turns
    out to be a simple and reliable stand-in for "the instructor" without
    needing anything fancier.
    """
    results = transcript_json["results"]
    segments = results.get("speaker_labels", {}).get("segments", [])

    # Sum up how many total seconds each speaker label was talking,
    # across every segment attributed to them.
    talk_time = {}
    for segment in segments:
        speaker = segment["speaker_label"]
        duration = float(segment["end_time"]) - float(segment["start_time"])
        talk_time[speaker] = talk_time.get(speaker, 0.0) + duration

    if not talk_time:
        # No diarization info (e.g. a single-speaker recording) — keep everything.
        return results["transcripts"][0]["transcript"]

    # Whichever speaker label accumulated the most total seconds is our
    # stand-in for "the instructor" — see the docstring above for why.
    primary_speaker = max(talk_time, key=talk_time.get)

    # Now walk every word/punctuation "item" in the transcript, in the
    # original spoken order, and keep only the ones attributed to the
    # primary speaker. AWS represents punctuation as its own item type
    # (not attached to the word before it), so `keep_next_punctuation`
    # tracks whether the word we just kept belongs to our speaker — if
    # so, the punctuation that immediately follows it should be glued
    # onto that same word rather than dropped or attached to whichever
    # word comes next.
    words = []
    keep_next_punctuation = False
    for item in results["items"]:
        speaker = item.get("speaker_label")
        content = item["alternatives"][0]["content"]
        if item["type"] == "pronunciation":
            if speaker == primary_speaker:
                words.append(content)
                keep_next_punctuation = True
            else:
                # A word from someone else (a student) — skip it, and
                # make sure we don't accidentally attach the *next*
                # punctuation mark to our own last-kept word.
                keep_next_punctuation = False
        elif item["type"] == "punctuation" and keep_next_punctuation:
            words[-1] = words[-1] + content

    return " ".join(words)
