#!/bin/sh
# Import the demo mailbox credentials and both workflows, publish them, start n8n.
# The inbox workflow starts one "case" execution per email, so every email waits for its own
# reviewer decision. The credentials belong to the local GreenMail test server only.
set -e
n8n import:credentials --input=/workflows/credentials.json
n8n import:workflow --input=/workflows/case.json
n8n import:workflow --input=/workflows/triage.json
n8n publish:workflow --id=firstReplyCase || echo "case workflow not published (called by id)"
n8n publish:workflow --id=firstReplyTriage
exec n8n start
