-- Documents, immutable versions, passages, and search requests. Every tenant table
-- copies the RLS pattern from 0002 and references other tenant tables with tenant_id
-- in the foreign key, so a row can never point at another tenant's parent. Grants
-- reach eg_app through the default privileges set in 0001.

-- Token count of a tsvector: one per position, or one for a lexeme without positions.
create function eg_token_count(tsvector) returns int
    language sql immutable parallel safe
as $$
    select coalesce(sum(coalesce(array_length(positions, 1), 1)), 0)::int from unnest($1)
$$;

create table blobs (
    tenant_id uuid not null references tenants (id),
    sha256 bytea not null,
    bytes bytea not null,
    size int not null,
    created_at timestamptz not null default now(),
    primary key (tenant_id, sha256)
);

alter table blobs enable row level security;
alter table blobs force row level security;
create policy tenant_isolation on blobs
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

create table documents (
    id uuid primary key default gen_random_uuid(),
    tenant_id uuid not null references tenants (id),
    collection_id uuid not null,
    title text not null,
    filename text not null,
    mime_type text not null,
    current_version_id uuid null,
    created_by uuid references users (id),
    created_at timestamptz not null default now(),
    deleted_at timestamptz null,
    unique (tenant_id, id),
    unique (tenant_id, id, collection_id),
    foreign key (tenant_id, collection_id) references collections (tenant_id, id)
);

create index documents_collection_idx
    on documents (tenant_id, collection_id, created_at desc, id desc);

alter table documents enable row level security;
alter table documents force row level security;
create policy tenant_isolation on documents
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

-- Versions are append-only (R5: no content-hash uniqueness, re-uploads are decided in code).
create table document_versions (
    id uuid primary key default gen_random_uuid(),
    tenant_id uuid not null references tenants (id),
    document_id uuid not null,
    version_number int not null,
    effective_from timestamptz not null,
    content_sha256 bytea not null,
    blob_sha256 bytea not null,
    text text not null,
    page_map jsonb not null default '[]',
    sections jsonb not null default '[]',
    metadata jsonb not null default '{}',
    parser text not null,
    parser_version text not null,
    char_count int not null,
    created_by uuid references users (id),
    created_at timestamptz not null default now(),
    unique (document_id, version_number),
    unique (tenant_id, id),
    unique (tenant_id, id, document_id),
    foreign key (tenant_id, document_id) references documents (tenant_id, id)
);

alter table document_versions enable row level security;
alter table document_versions force row level security;
create policy tenant_isolation on document_versions
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

alter table documents
    add constraint documents_current_version_fk
    foreign key (tenant_id, current_version_id) references document_versions (tenant_id, id)
    deferrable initially deferred;

create function eg_document_versions_immutable() returns trigger
    language plpgsql
as $$
begin
    raise exception 'document_versions are immutable';
end
$$;

create trigger document_versions_immutable
    before update or delete on document_versions
    for each row execute function eg_document_versions_immutable();

create table passages (
    id uuid primary key default gen_random_uuid(),
    tenant_id uuid not null references tenants (id),
    document_id uuid not null,
    version_id uuid not null,
    collection_id uuid not null,
    ordinal int not null,
    start_char int not null,
    end_char int not null,
    page_start int,
    page_end int,
    section text,
    text text not null,
    text_sha256 bytea not null,
    tsv tsvector generated always as (to_tsvector('english', text)) stored,
    token_count int generated always as (eg_token_count(to_tsvector('english', text))) stored,
    embedding vector(384) not null,
    embedding_model text not null,
    is_current boolean not null default false,
    created_at timestamptz not null default now(),
    unique (version_id, ordinal),
    unique (tenant_id, id),
    unique (tenant_id, id, version_id),
    foreign key (tenant_id, collection_id) references collections (tenant_id, id),
    foreign key (tenant_id, version_id, document_id)
        references document_versions (tenant_id, id, document_id),
    check (0 <= start_char and start_char < end_char)
);

create index passages_embedding_hnsw
    on passages using hnsw (embedding vector_cosine_ops) with (m = 16, ef_construction = 64);
create index passages_tsv_idx on passages using gin (tsv);
create index passages_current_idx on passages (tenant_id, is_current, collection_id);
create index passages_document_version_idx on passages (document_id, version_id);
create index passages_text_hash_idx on passages (tenant_id, text_sha256, embedding_model);

alter table passages enable row level security;
alter table passages force row level security;
create policy tenant_isolation on passages
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

-- The passage must be exactly the version text slice. A missing version is left to the
-- foreign key, which reports it as a ForeignKeyViolation.
create function eg_passages_slice_check() returns trigger
    language plpgsql
as $$
declare
    version_text text;
    version_chars int;
begin
    select v.text, v.char_count into version_text, version_chars
    from document_versions v
    where v.tenant_id = new.tenant_id and v.id = new.version_id;
    if not found then
        return new;
    end if;
    if substr(version_text, new.start_char + 1, new.end_char - new.start_char) <> new.text
        or new.end_char > version_chars
        or new.text_sha256 <> sha256(convert_to(new.text, 'UTF8'))
    then
        raise exception 'passage text does not match version slice';
    end if;
    return new;
end
$$;

create trigger passages_slice_check
    before insert on passages
    for each row execute function eg_passages_slice_check();

-- Only the retrieval flags may change: is_current and collection_id.
create function eg_passages_immutable() returns trigger
    language plpgsql
as $$
begin
    if tg_op = 'DELETE' then
        raise exception 'passage content is immutable';
    end if;
    if (new.id, new.tenant_id, new.document_id, new.version_id, new.ordinal, new.start_char,
        new.end_char, new.page_start, new.page_end, new.section, new.text, new.text_sha256,
        new.embedding, new.embedding_model, new.created_at)
        is distinct from
        (old.id, old.tenant_id, old.document_id, old.version_id, old.ordinal, old.start_char,
        old.end_char, old.page_start, old.page_end, old.section, old.text, old.text_sha256,
        old.embedding, old.embedding_model, old.created_at)
    then
        raise exception 'passage content is immutable';
    end if;
    return new;
end
$$;

create trigger passages_immutable
    before update or delete on passages
    for each row execute function eg_passages_immutable();

create table passage_terms (
    tenant_id uuid not null references tenants (id),
    passage_id uuid not null,
    term text not null,
    tf int not null,
    primary key (passage_id, term),
    foreign key (tenant_id, passage_id) references passages (tenant_id, id)
);

create index passage_terms_term_idx on passage_terms (tenant_id, term);

alter table passage_terms enable row level security;
alter table passage_terms force row level security;
create policy tenant_isolation on passage_terms
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

-- Runs as the inserting role (not security definer), so forced RLS on passage_terms
-- applies and the rows carry the passage's own tenant.
create function eg_passages_fill_terms() returns trigger
    language plpgsql
as $$
begin
    insert into passage_terms (tenant_id, passage_id, term, tf)
    select new.tenant_id, new.id, u.lexeme, coalesce(array_length(u.positions, 1), 1)
    from unnest(new.tsv) u;
    return null;
end
$$;

create trigger passages_fill_terms
    after insert on passages
    for each row execute function eg_passages_fill_terms();

create table uploads (
    id uuid primary key default gen_random_uuid(),
    tenant_id uuid not null references tenants (id),
    document_id uuid not null,
    blob_sha256 bytea not null,
    filename text not null,
    declared_mime text not null,
    effective_from timestamptz not null,
    status text not null default 'queued'
        check (status in ('queued', 'processing', 'created_version', 'duplicate', 'failed')),
    version_id uuid null,
    error_code text null,
    stats jsonb not null default '{}',
    created_by uuid references users (id),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    foreign key (tenant_id, document_id) references documents (tenant_id, id),
    foreign key (tenant_id, blob_sha256) references blobs (tenant_id, sha256),
    foreign key (tenant_id, version_id) references document_versions (tenant_id, id)
);

alter table uploads enable row level security;
alter table uploads force row level security;
create policy tenant_isolation on uploads
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

-- One row holds the query and its result, so job payloads carry only the id (R4).
create table search_requests (
    id uuid primary key default gen_random_uuid(),
    tenant_id uuid not null references tenants (id),
    user_id uuid not null references users (id),
    query text not null,
    params jsonb not null,
    status text not null default 'queued' check (status in ('queued', 'done', 'failed')),
    hits jsonb null,
    error_code text null,
    created_at timestamptz not null default now(),
    completed_at timestamptz null
);

alter table search_requests enable row level security;
alter table search_requests force row level security;
create policy tenant_isolation on search_requests
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

-- Indexes deferred from the Plan A final review.
create index memberships_user_idx on memberships (user_id);
create index collection_grants_tenant_user_idx on collection_grants (tenant_id, user_id);
create index sessions_tenant_user_idx on sessions (tenant_id, user_id);
create index api_keys_tenant_user_idx on api_keys (tenant_id, user_id);
