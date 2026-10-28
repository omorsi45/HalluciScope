-- Analyses and the evidence graph: claims, evidence links, citations, claim re-checks, and
-- feedback. Every tenant table copies the RLS pattern from 0002 and references other tenant
-- tables with tenant_id in the foreign key, so a row can never point at another tenant's
-- parent. Only users and tenants are referenced by a single column. Grants reach eg_app
-- through the default privileges set in 0001.

create table analyses (
    id uuid primary key default gen_random_uuid(),
    tenant_id uuid not null references tenants (id),
    user_id uuid not null references users (id),
    kind text not null check (kind in ('ask', 'verify', 'quick')),
    question text null,
    answer text null,
    collection_ids uuid[] not null,
    as_of timestamptz null,
    known_at timestamptz not null,
    status text not null default 'queued'
        check (status in ('queued', 'running', 'succeeded', 'failed')),
    progress jsonb not null default '{}',
    error_code text null,
    options jsonb not null default '{}',
    config jsonb not null default '{}',
    timings jsonb not null default '{}',
    summary jsonb not null default '{}',
    created_at timestamptz not null default now(),
    completed_at timestamptz null,
    unique (tenant_id, id)
);

create index analyses_user_idx on analyses (tenant_id, user_id, created_at desc, id desc);
create index analyses_tenant_idx on analyses (tenant_id, created_at desc, id desc);
create index analyses_status_idx on analyses (tenant_id, status);

alter table analyses enable row level security;
alter table analyses force row level security;
create policy tenant_isolation on analyses
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

create table claims (
    id uuid primary key default gen_random_uuid(),
    tenant_id uuid not null references tenants (id),
    analysis_id uuid not null,
    ordinal int not null,
    text text not null,
    answer_start int not null,
    answer_end int not null,
    extraction_method text not null
        check (extraction_method in ('llm_atomic', 'sentence', 'sentence_fallback')),
    verdict text not null check (verdict in ('supported', 'contradicted', 'insufficient')),
    p_supported float8 not null,
    p_contradicted float8 not null,
    p_insufficient float8 not null,
    features jsonb not null,
    diagnostics jsonb not null default '{}',
    created_at timestamptz not null default now(),
    unique (analysis_id, ordinal),
    unique (tenant_id, id),
    foreign key (tenant_id, analysis_id) references analyses (tenant_id, id) on delete cascade
);

alter table claims enable row level security;
alter table claims force row level security;
create policy tenant_isolation on claims
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

-- Each link names an exact (passage, version) pair; the composite foreign key makes the
-- database reject a passage paired with a version it does not belong to.
create table evidence_links (
    id uuid primary key default gen_random_uuid(),
    tenant_id uuid not null references tenants (id),
    claim_id uuid not null,
    passage_id uuid not null,
    version_id uuid not null,
    relation text not null check (relation in ('supports', 'contradicts', 'related')),
    rank int not null,
    span_start int not null,
    span_end int not null,
    nli_entail float8,
    nli_contra float8,
    nli_neutral float8,
    rerank_score float8 null,
    dense_cos float8,
    created_at timestamptz not null default now(),
    foreign key (tenant_id, claim_id) references claims (tenant_id, id) on delete cascade,
    foreign key (tenant_id, passage_id, version_id)
        references passages (tenant_id, id, version_id),
    check (span_start < span_end)
);

create index evidence_links_version_idx on evidence_links (tenant_id, version_id);
create index evidence_links_claim_idx on evidence_links (claim_id, rank);

alter table evidence_links enable row level security;
alter table evidence_links force row level security;
create policy tenant_isolation on evidence_links
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

-- An unknown marker has no passage. With a null passage_id or version_id the composite
-- foreign key is not enforced (MATCH SIMPLE), so the status check carries that case.
create table citations (
    id uuid primary key default gen_random_uuid(),
    tenant_id uuid not null references tenants (id),
    analysis_id uuid not null,
    number int not null,
    marker_start int not null,
    marker_end int not null,
    sentence_start int not null,
    sentence_end int not null,
    passage_id uuid null,
    version_id uuid null,
    status text not null check (status in ('valid', 'unknown_marker')),
    foreign key (tenant_id, analysis_id) references analyses (tenant_id, id) on delete cascade,
    foreign key (tenant_id, passage_id, version_id)
        references passages (tenant_id, id, version_id),
    check ((status = 'valid') = (passage_id is not null and version_id is not null))
);

create index citations_analysis_idx on citations (analysis_id);

alter table citations enable row level security;
alter table citations force row level security;
create policy tenant_isolation on citations
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

-- Re-verification history. created_at uses clock_timestamp() so rows written in one
-- transaction still order by time when the latest check is picked.
create table claim_checks (
    id uuid primary key default gen_random_uuid(),
    tenant_id uuid not null references tenants (id),
    claim_id uuid not null,
    reason text not null check (reason in ('source_changed', 'manual')),
    verdict text not null check (verdict in ('supported', 'contradicted', 'insufficient')),
    p_supported float8 not null,
    p_contradicted float8 not null,
    p_insufficient float8 not null,
    evidence jsonb not null default '[]',
    checked_version_ids uuid[] not null,
    changed boolean not null,
    created_at timestamptz not null default clock_timestamp(),
    foreign key (tenant_id, claim_id) references claims (tenant_id, id) on delete cascade
);

create index claim_checks_claim_idx on claim_checks (claim_id, created_at desc, id desc);

alter table claim_checks enable row level security;
alter table claim_checks force row level security;
create policy tenant_isolation on claim_checks
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);

-- Append-only: UPDATE is rejected. DELETE is left alone because deleting an analysis
-- cascades to its claims and their checks.
create function eg_claim_checks_append_only() returns trigger
    language plpgsql
as $$
begin
    raise exception 'claim_checks are append-only';
end
$$;

create trigger claim_checks_append_only
    before update on claim_checks
    for each row execute function eg_claim_checks_append_only();

-- One feedback row per user per claim, upserted by the repository.
create table feedback (
    id uuid primary key default gen_random_uuid(),
    tenant_id uuid not null references tenants (id),
    claim_id uuid not null,
    user_id uuid not null references users (id),
    label text not null check (label in ('correct', 'incorrect')),
    note text null check (note is null or length(note) <= 1000),
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (claim_id, user_id),
    foreign key (tenant_id, claim_id) references claims (tenant_id, id) on delete cascade
);

alter table feedback enable row level security;
alter table feedback force row level security;
create policy tenant_isolation on feedback
    using (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid)
    with check (tenant_id = nullif(current_setting('app.tenant_id', true), '')::uuid);
