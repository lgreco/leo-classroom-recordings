"""Download a lecture recording from a shareable cloud-storage link.

Used by the "Upload Lecture From Link" GitHub Actions workflow
(.github/workflows/upload-from-link.yml), which exists so a recording can
be pushed into the pipeline straight from a phone on the way out of
class: GitHub's own web "Upload files" flow doesn't reliably produce a
proper Git LFS pointer for a file this size (confirmed the hard way --
see next_checklist.md), so instead the phone just needs to get the
recording into a shareable link (e.g. Google Drive), and this script does
the actual download server-side, inside a GitHub Actions runner that
already has working git-lfs (see process-lecture.yml's checkout step).

Google Drive specifically needs special handling: for anything but small
files, Drive interposes an interstitial "Google Drive can't scan this
file for viruses" confirmation page instead of serving the raw bytes
directly, so a plain streamed GET would silently save that HTML page as
if it were the audio file. `gdown` exists specifically to handle that
confirmation flow correctly, so it's tried first for any drive.google.com
link; anything else (Dropbox, a plain direct-download URL, etc.) falls
back to a plain streamed HTTP download.
"""

import argparse
import os
import sys
import urllib.request


def looks_like_html(path, sniff_bytes=256):
    """A cheap sanity check for the most common failure mode here: the
    link wasn't actually shareable (not set to "Anyone with the link"),
    so what we downloaded is a Google sign-in/permission-denied page
    instead of audio. Those pages are tiny, valid UTF-8, and start with
    an HTML doctype/tag -- a real m4a/mp3/etc. file never does.
    """
    with open(path, "rb") as f:
        head = f.read(sniff_bytes).lstrip().lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html")


def download_plain(url, output_path):
    """Stream a direct-download URL to disk. Used for anything that isn't
    a Google Drive link, which (unlike Drive) generally serves file bytes
    straight from a GET with no confirmation-page detour.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(output_path, "wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="Shareable link to the recording")
    parser.add_argument("output_path", help="Where to save it, e.g. comp170.m4a")
    args = parser.parse_args()

    if "drive.google.com" in args.url:
        import gdown

        # gdown's own URL parser pulls the file ID out of whatever shape
        # of Drive link this is (a /view link, a /open?id= link, etc.),
        # then we hand that ID straight to download() -- more robust than
        # passing the URL through as-is, which only works for Drive's raw
        # /uc?id=... download-link format.
        file_id, _ = gdown.parse_url.parse_url(args.url)
        if file_id:
            gdown.download(id=file_id, output=args.output_path, quiet=False)
        else:
            gdown.download(url=args.url, output=args.output_path, quiet=False)
    else:
        download_plain(args.url, args.output_path)

    if not os.path.exists(args.output_path) or os.path.getsize(args.output_path) == 0:
        sys.exit(
            f"Download produced no file at {args.output_path} -- "
            "check the link is correct."
        )

    if looks_like_html(args.output_path):
        sys.exit(
            "Downloaded content looks like an HTML page, not audio -- the "
            'link probably isn\'t set to "Anyone with the link" sharing, '
            "or isn't a direct file link."
        )

    size = os.path.getsize(args.output_path)
    print(f"Downloaded {size:,} bytes to {args.output_path}")


if __name__ == "__main__":
    main()
