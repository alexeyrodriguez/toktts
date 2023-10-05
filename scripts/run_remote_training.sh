#!/bin/bash

# Arguments:
#  - gcs bucket where to copy model
#  - additional trainer args

echo "Starting remote train"

GCS_PATH=$1
POWER_OFF=$2
shift 2

# Get required env
. /etc/profile

set -x
set -e

cd "$(dirname "$0")"/..

# put the below stuff into requirements
pip install -r requirements.txt

python trainer.py "$@"
python generate_samples.py "$@"

if [ "$POWER_OFF" == "yes" ]; then
    sudo poweroff
fi