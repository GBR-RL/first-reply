#!/bin/sh
# Import the demo mailbox credentials and the triage workflow, publish it, start n8n.
# The credentials belong to the local GreenMail test server only.
set -e
n8n import:credentials --input=/workflows/credentials.json
n8n import:workflow --input=/workflows/triage.json
n8n publish:workflow --id=firstReplyTriage
exec n8n start
