#!/usr/bin/env bash
# Local Postgres 17 for development: start | stop
set -euo pipefail

PG_BIN="${PG_BIN:-/opt/homebrew/opt/postgresql@17/bin}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PGDATA="$ROOT/.local/pgdata"
PGPORT=54329
PGLOG="$ROOT/.local/pg.log"

is_running() {
  "$PG_BIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1
}

# psql as the superuser against a given database, quiet, tuples only.
admin_psql() {
  PGOPTIONS="-c client_min_messages=warning" "$PG_BIN/psql" -h localhost -p "$PGPORT" -U postgres -d "$1" -v ON_ERROR_STOP=1 -Atq -c "$2"
}

ensure_role() {
  local role="$1"
  if [ -z "$(admin_psql postgres "select 1 from pg_roles where rolname = '$role'")" ]; then
    admin_psql postgres "create role $role login nosuperuser nobypassrls"
  fi
}

start() {
  mkdir -p "$ROOT/.local"
  if [ ! -d "$PGDATA" ]; then
    "$PG_BIN/initdb" -D "$PGDATA" -U postgres --auth=trust -E UTF8 --locale=en_US.UTF-8
  fi
  if ! is_running; then
    "$PG_BIN/pg_ctl" -D "$PGDATA" -o "-p $PGPORT -k $ROOT/.local" -l "$PGLOG" start
  fi
  until "$PG_BIN/pg_isready" -h localhost -p "$PGPORT" -q; do
    sleep 0.5
  done

  ensure_role eg_owner
  ensure_role eg_app
  if [ -z "$(admin_psql postgres "select 1 from pg_database where datname = 'evidencegraph'")" ]; then
    admin_psql postgres "create database evidencegraph owner eg_owner"
  fi
  admin_psql evidencegraph "create extension if not exists vector"
  admin_psql evidencegraph "create extension if not exists citext"

  echo "app:       postgresql://eg_app@localhost:$PGPORT/evidencegraph"
  echo "migration: postgresql://eg_owner@localhost:$PGPORT/evidencegraph"
}

stop() {
  if is_running; then
    "$PG_BIN/pg_ctl" -D "$PGDATA" stop
  fi
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  *) echo "usage: $0 start|stop" >&2; exit 2 ;;
esac
