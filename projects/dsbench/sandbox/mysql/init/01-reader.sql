-- The login that model-written SQL runs as (sftgen/engines.MySQLEngine). An engine grants it
-- SELECT on one scratch database; it has no FILE privilege, so no INTO OUTFILE or LOAD_FILE().
CREATE USER 'sftgen_reader'@'%' IDENTIFIED BY 'sftgen_reader';
