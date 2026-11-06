"""Submit a finished kernel version to the Enveda CASMI 2026 code competition.

Kaggle only accepts submissions produced by a notebook for this competition, so
the flow is: push the kernel (scripts/build_kernel.py + POST /kernels/push or
the SaveKernel gateway), wait for status "complete", then submit that version
through the code-submission gateway with the kernel's output file name.

The request is the same one `kaggle competitions submit -k ... -v ...` makes
(competitions.CompetitionApiService/CreateCodeSubmission), sent through the
omac kaggle sidecar's generic /proxy so no CLI runs inside the sandbox.

Usage (from anywhere; needs the sidecar running):
    uv run python scripts/submit_kernel.py --version 3 [--file submission.csv]
        [--message "stage-2 re-ranker"] [--kernel maxdietsch/casmi26-stage2-sub]
        [--competition enveda-CASMI26-molecule-id-mass-spectra]

If Kaggle answers 403 IdentityVerificationRequired, verify the account once at
https://www.kaggle.com/settings and run this again.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("OMAC_KAGGLE_BASE", "http://127.0.0.1:44841/kaggle")
TOKEN = os.environ.get("OMAC_FACADE_TOKEN", "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kernel", default="maxdietsch/casmi26-stage2-sub",
                        help="<owner>/<slug> of the finished kernel")
    parser.add_argument("--version", type=int, required=True, help="kernel version to submit")
    parser.add_argument("--file", default="submission.csv",
                        help="output file the kernel produced (code competitions: submission.csv)")
    parser.add_argument("--competition", default="enveda-CASMI26-molecule-id-mass-spectra")
    parser.add_argument("--message", default="stage-2 learned re-ranker over library-search shortlists")
    args = parser.parse_args()
    owner, slug = args.kernel.split("/")

    body = {"method": "POST",
            "path": "/api/v1/competitions.CompetitionApiService/CreateCodeSubmission",
            "json": {"competitionName": args.competition, "kernelOwner": owner,
                     "kernelSlug": slug, "kernelVersion": args.version,
                     "fileName": args.file,
                     "submissionDescription": args.message}}
    req = urllib.request.Request(BASE + "/proxy", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "X-Omac-Facade-Token": TOKEN})
    try:
        res = json.loads(urllib.request.urlopen(req, timeout=300).read())
    except urllib.error.HTTPError as e:
        res = json.loads(e.read())
    body = res.get("body")
    if res.get("ok"):
        print("submitted:", json.dumps(body))
        print(f"kernel: https://www.kaggle.com/code/{owner}/{slug}")
        print("scores appear in the submissions list once graded (a few minutes)")
        sys.exit(0)
    err = body if isinstance(body, dict) else {"body": body}
    msg = (err.get("error") or {}).get("message") or str(err)[:300]
    print(f"submission failed (status {res.get('status')}): {msg}")
    if "Identity Verify" in msg:
        print("\nKaggle requires one-time identity verification for submissions:")
        print("  1. open https://www.kaggle.com/settings and verify your account (phone number)")
        print("  2. run this script again")
    sys.exit(1)


if __name__ == "__main__":
    main()
