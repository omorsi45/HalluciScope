# Task 7 open findings (round 1)

1. (Important, verbatim) 0002_collections.sql:19-26: collection_grants.tenant_id is not tied to the referenced collection's tenant. Reproduced live: inside tenant_tx(A) a grant with tenant_id=A and collection_id=<tenant B's collection> was accepted; FK checks bypass RLS, giving tenant A an existence oracle for any collection UUID; B deleting its collection cascade-deleted tenant A's grant row.

Controller ruling (binding; supersedes the brief's single-column FK):
- collections gets `unique (tenant_id, id)`.
- collection_grants references `foreign key (tenant_id, collection_id) references collections (tenant_id, id) on delete cascade` (replacing the single-column collection FK).
- collection_grants also references `foreign key (tenant_id, user_id) references memberships (tenant_id, user_id) on delete cascade` (replacing the single-column users FK), so a grant can only name a tenant member and removing a member removes their grants.
- Add pg tests: (a) inside tenant_tx(A), inserting a grant whose collection_id belongs to tenant B raises psycopg.errors.ForeignKeyViolation; (b) a grant for a user who is not a member of the tenant raises ForeignKeyViolation; (c) deleting the membership deletes the grant.
- Edit 0002_collections.sql in place (it has never been applied outside test databases, which are recreated per session). Delete the local dev database's applied 0002 row is NOT needed: dev DB only has migrations applied if someone ran halluciscope migrate; check `psql -h localhost -p 54329 -U eg_owner -d evidencegraph -Atc "select version from schema_migrations"` and if 2 is present, drop and recreate the dev database with `bash scripts/dev-postgres.sh` logic (dropdb as postgres, then rerun start) and report it.
