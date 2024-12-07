#!/bin/bash
# The pre-trained Google News word2vec embeddings are 1.5G compressed and 3.4G
# uncompressed, and as such will not be included in the repo. This script will
# download and extract the embeddings for use in training.

set -eE
trap 'rm -f tmp.gz' ERR INT

# from https://code.google.com/archive/p/word2vec/
DRIVE_FILE_HASH='0B7XkCwpI5KDYNlNUTTlSS21pQmM'
OUTPUT_FILE='GoogleNews-vectors-negative300.bin'
wget "https://drive.usercontent.google.com/download?id=$DRIVE_FILE_HASH&export=download&confirm=yes" -O tmp.gz

if command -v pv >/dev/null; then
    zcat tmp.gz | pv >"$OUTPUT_FILE"
else
    zcat tmp.gz >"$OUTPUT_FILE"
fi

rm tmp.gz
