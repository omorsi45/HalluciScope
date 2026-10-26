-- Collections and per-user grants. Every tenant table copies the RLS pattern below.
create table collections (
    id uuid primary key default gen_random_uuid(),
    tenant_id uuid not null references tenants (id),
    name text not null,
    description text not null default '',
    kind text not null default 'normal' check (kind in ('normal', 'quick', 'benchmark')),
    created_by uuid references users (id),
    created_at timestamptz not null default now(),
    unique (tenant_id, name)
);

alter table collections enable row level security;
alter table collections force row level security;
create policy tenant_isolation on collections
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

create table collection_grants (
    tenant_id uuid not null references tenants (id),
    collection_id uuid not null references collections (id) on delete cascade,
    user_id uuid not null references users (id),
    permission text not null check (permission in ('read', 'write', 'admin')),
    created_at timestamptz not null default now(),
    primary key (collection_id, user_id)
);

alter table collection_grants enable row level security;
alter table collection_grants force row level security;
create policy tenant_isolation on collection_grants
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);
