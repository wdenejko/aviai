-- The login that model-written SQL runs as (sftgen/engines.PostgresEngine). It can log in and read
-- what an engine grants it on one scratch schema, nothing else: no superuser, so no COPY to or
-- from files or programs, and every statement stops after 30 s.
CREATE ROLE sftgen_reader LOGIN PASSWORD 'sftgen_reader' NOSUPERUSER NOCREATEDB NOCREATEROLE;
ALTER ROLE sftgen_reader SET statement_timeout = '30s';
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
