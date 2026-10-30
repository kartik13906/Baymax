#!/bin/sh
# Starts the Baymax server. With BAYMAX_TLS=1 it makes a self-signed certificate first, because browsers only
# allow the microphone on https:// (or http://localhost). Other devices on your network need the https option.
set -eu

if [ "${BAYMAX_TLS:-0}" = "1" ]; then
  dir=/data/tls
  mkdir -p "$dir"
  if [ ! -f "$dir/cert.pem" ] || [ ! -f "$dir/key.pem" ]; then
    echo "Making a self-signed certificate in $dir (your browser will warn about it once) ..."
    openssl req -x509 -newkey rsa:2048 -nodes -days 825 -subj "/CN=baymax" \
      -keyout "$dir/key.pem" -out "$dir/cert.pem" 2>/dev/null
    chmod 600 "$dir/key.pem"
  fi
  export BAYMAX_SSL_CERT="$dir/cert.pem" BAYMAX_SSL_KEY="$dir/key.pem"
fi

exec python server.py "$@"
