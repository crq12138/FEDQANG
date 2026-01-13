#!/usr/bin/env bash
set -e

ports=(50051 50053 50055 50057 50059 50061 50063 50065 50067 50069)

for port in "${ports[@]}"; do
    lsof -i:"$port" -t | xargs kill || true
done
