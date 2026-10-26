-- Global schema: tables without row-level security, and grants for the app role.
-- Roles (eg_owner, eg_app) and extensions are created by the superuser before
-- migrations run; the statements below are no-ops in that case.
create extension if not exists citext;
create extension if not exists vector;

create table tenants (
    id uuid primary key default gen_random_uuid(),
    name text not null,
    created_at timestamptz not null default now()
);

create table users (
    id uuid primary key default gen_random_uuid(),
    email citext not null unique,
    password_hash text not null,
    display_name text not null,
    created_at timestamptz not null default now(),
    disabled_at timestamptz null
);

create table memberships (
    tenant_id uuid not null references tenants (id),
    user_id uuid not null references users (id),
    role text not null check (role in ('owner', 'admin', 'member')),
    created_at timestamptz not null default now(),
    primary key (tenant_id, user_id)
);

create table sessions (
    id uuid primary key default gen_random_uuid(),
    token_hash bytea not null unique,
    csrf_hash bytea not null,
    user_id uuid not null references users (id),
    tenant_id uuid not null references tenants (id),
    created_at timestamptz not null default now(),
    expires_at timestamptz not null,
    revoked_at timestamptz null
);

create table api_keys (
    id uuid primary key default gen_random_uuid(),
    tenant_id uuid not null references tenants (id),
    user_id uuid not null references users (id),
    name text not null,
    prefix text not null,
    key_hash bytea not null unique,
    created_at timestamptz not null default now(),
    last_used_at timestamptz null,
    revoked_at timestamptz null
);

create table auth_throttle (
    key text not null,
    window_start timestamptz not null,
    failures int not null default 0,
    primary key (key, window_start)
);

create table jobs (
    id bigserial primary key,
    tenant_id uuid null references tenants (id),
    kind text not null,
    payload jsonb not null default '{}',
    priority int not null default 0,
    status text not null default 'queued'
        check (status in ('queued', 'running', 'succeeded', 'failed', 'dead')),
    attempts int not null default 0,
    max_attempts int not null default 3,
    run_after timestamptz not null default now(),
    locked_by text null,
    locked_at timestamptz null,
    heartbeat_at timestamptz null,
    last_error text null,
    idempotency_key text null unique,
    request_id text null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index jobs_claim_idx on jobs (status, priority desc, run_after, id);

create table workers (
    id text primary key,
    kinds text[] not null,
    started_at timestamptz not null default now(),
    heartbeat_at timestamptz not null default now()
);

create table audit_log (
    id bigserial primary key,
    tenant_id uuid null references tenants (id),
    user_id uuid null references users (id),
    action text not null,
    target_type text null,
    target_id text null,
    request_id text null,
    at timestamptz not null default now(),
    details jsonb null
);

create index audit_log_tenant_at_idx on audit_log (tenant_id, at desc, id desc);

grant usage on schema public to eg_app;
grant select, insert, update, delete on all tables in schema public to eg_app;
grant usage, select on all sequences in schema public to eg_app;
alter default privileges for role eg_owner in schema public
    grant select, insert, update, delete on tables to eg_app;
alter default privileges for role eg_owner in schema public
    grant usage, select on sequences to eg_app;
revoke insert, update, delete on schema_migrations from eg_app;
