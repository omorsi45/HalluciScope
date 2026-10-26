-- Collections and per-user grants. Every tenant table copies the RLS pattern below.
create table collections (
    id uuid primary key default gen_random_uuid(),
    tenant_id uuid not null references tenants (id),
    name text not null,
    description text not null default '',
    kind text not null default 'normal' check (kind in ('normal', 'quick', 'benchmark')),
    created_by uuid references users (id),
    created_at timestamptz not null default now(),
    unique (tenant_id, name),
    unique (tenant_id, id)
);

alter table collections enable row level security;
alter table collections force row level security;
create policy tenant_isolation on collections
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

create table collection_grants (
    tenant_id uuid not null references tenants (id),
    collection_id uuid not null,
    user_id uuid not null,
    permission text not null check (permission in ('read', 'write', 'admin')),
    created_at timestamptz not null default now(),
    primary key (collection_id, user_id),
    foreign key (tenant_id, collection_id) references collections (tenant_id, id) on delete cascade,
    foreign key (tenant_id, user_id) references memberships (tenant_id, user_id) on delete cascade
);

alter table collection_grants enable row level security;
alter table collection_grants force row level security;
create policy tenant_isolation on collection_grants
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);
