-- BTS "Reporting Carrier On-Time Performance" — operational subset (ADR-003 Appendix A).
-- The ingest loader (ingest/bts.py) fills this; metar/taf/notam/airports tables are created
-- by their own loaders so schema and load stay together. One month, aligned airport set.
CREATE TABLE IF NOT EXISTS aviation.flights
(
    FlightDate                       Date,
    Reporting_Airline                LowCardinality(String),
    Flight_Number_Reporting_Airline  String,
    Origin                           LowCardinality(String),
    Dest                             LowCardinality(String),
    CRSDepTime                       String,        -- 'hhmm' local, kept as-is (BTS quirk: '2400' etc.)
    DepTime                          String,
    DepDelay                         Nullable(Float32),
    DepDelayMinutes                  Nullable(Float32),
    TaxiOut                          Nullable(Float32),
    WheelsOff                        String,
    WheelsOn                         String,
    TaxiIn                           Nullable(Float32),
    CRSArrTime                       String,
    ArrTime                          String,
    ArrDelay                         Nullable(Float32),
    ArrDelayMinutes                  Nullable(Float32),
    Cancelled                        UInt8,
    CancellationCode                 LowCardinality(String),
    Diverted                         UInt8,
    AirTime                          Nullable(Float32),
    Distance                         Nullable(Float32),
    CarrierDelay                     Nullable(Float32),
    WeatherDelay                     Nullable(Float32),
    NASDelay                         Nullable(Float32),
    SecurityDelay                    Nullable(Float32),
    LateAircraftDelay                Nullable(Float32)
)
ENGINE = MergeTree
ORDER BY (FlightDate, Origin, Dest);
