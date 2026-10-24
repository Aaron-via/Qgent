-- Widen the claims.claim_type constraint to accept 'plan'.
--
-- init.sql created the CHECK inline and without a name, so PostgreSQL named it
-- claims_claim_type_check.  This drops that constraint if it is there and
-- re-adds it with the five types extract_pipeline.py can produce.
--
-- Safe to run more than once.  Run it once per database, before the first
-- import that contains a plan claim:
--
--     psql "$DATABASE_URL" -f migrate_claims_plan.sql
--
-- (or spell the connection string out: postgresql://<user>:<password>@<host>:5432/<db>)
--
-- extract_pipeline.py --write-db checks for this constraint up front and
-- refuses to import (before spending any API call) while it is still missing.

BEGIN;

ALTER TABLE claims DROP CONSTRAINT IF EXISTS claims_claim_type_check;

ALTER TABLE claims ADD CONSTRAINT claims_claim_type_check
    CHECK (claim_type IN (
        'contribution',
        'limitation',
        'assumption',
        'position',
        'plan'
    ));

COMMIT;
