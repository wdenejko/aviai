"""TAF layer tests — schema sparsity, both oracle voices, and period-aligned consensus
(align by sequence, timing from the stated-timing voice, ties are hard cases)."""

from avtext.consensus.vote_taf import consensus_taf
from avtext.oracles.avwx_taf import AvwxTafOracle
from avtext.oracles.mivek_taf import MivekTafOracle
from avtext.schema import ChangeType, ForecastPeriod, TafForecast, Wind

# A fixed US TAF: initial + an FM with a gust + a TEMPO with no wind (sparsity), P6SM visibility.
KORD = (
    "TAF KORD 011130Z 0112/0218 18008KT P6SM SCT250 "
    "FM011600 21012G20KT P6SM BKN035 "
    "TEMPO 0116/0120 5SM -SHRA BKN025"
)


def _p(ct: ChangeType, fd: int, fh: int, **kw) -> ForecastPeriod:
    return ForecastPeriod(change_type=ct, from_day=fd, from_hour=fh, **kw)


def _fc(periods: list[ForecastPeriod], station: str = "KXXX") -> TafForecast:
    return TafForecast(station=station, issue_day=1, issue_hour=0, issue_minute=0, periods=periods)


def test_schema_sparsity_and_defaults():
    # A change group states only what changes: an unstated field is None, not a gap.
    t = _fc([_p(ChangeType.TEMPO, 1, 16, visibility_m=8047)])
    assert t.periods[0].wind is None  # not stated
    assert t.periods[0].visibility_plus is False  # default


def test_avwx_decodes_kord():
    r = AvwxTafOracle().decode(KORD)
    assert r.ok
    f = r.forecast
    assert f.station == "KORD" and f.issue_hour == 11 and f.issue_minute == 30
    assert f.valid_from_day == 1 and f.valid_from_hour == 12
    assert f.periods[0].change_type is ChangeType.INITIAL
    assert f.periods[0].wind.direction == 180 and f.periods[0].wind.speed_kt == 8
    assert any(p.wind and p.wind.gust_kt == 20 for p in f.periods)  # the FM group's gust


def test_mivek_decodes_kord():
    r = MivekTafOracle().decode(KORD)
    assert r.ok
    assert r.forecast.station == "KORD"
    assert r.forecast.periods[0].wind.direction == 180
    assert r.forecast.periods[0].wind.speed_kt == 8


def test_consensus_aligns_two_voices_and_takes_mivek_timing():
    a = AvwxTafOracle().decode(KORD).forecast
    m = MivekTafOracle().decode(KORD).forecast
    con = consensus_taf({"avwx": a, "mivek": m})
    assert not con.structural_dissent
    assert con.header["station"] == "KORD"
    assert con.periods[0]["change_type"] == "INITIAL"
    assert con.periods[0]["wind_dir"] == 180 and con.periods[0]["wind_speed"] == 8
    # timing is NOT voted — it comes from the stated-timing voice (mivek)
    assert con.periods[0]["from_day"] == m.periods[0].from_day
    assert con.periods[0]["from_hour"] == m.periods[0].from_hour


def test_consensus_tie_is_undecided():
    # two voices disagree on the initial wind speed -> 1-1 tie -> None + flagged
    f1 = _fc([_p(ChangeType.INITIAL, 1, 12, wind=Wind(direction=180, speed_kt=8))])
    f2 = _fc([_p(ChangeType.INITIAL, 1, 12, wind=Wind(direction=180, speed_kt=10))])
    con = consensus_taf({"avwx": f1, "mivek": f2})
    assert con.periods[0]["wind_speed"] is None
    assert "p0.wind_speed" in con.disagreements
    assert con.periods[0]["wind_dir"] == 180  # the agreed field is still decided


def test_visibility_plus_coupled_to_visibility():
    # one voice reports 9999+ vis, the other abstains on vis -> plus must NOT read as a disagreement
    f1 = _fc([_p(ChangeType.INITIAL, 1, 12, visibility_m=9999, visibility_plus=True)])
    f2 = _fc([_p(ChangeType.INITIAL, 1, 12)])  # vis None, plus default False
    con = consensus_taf({"avwx": f1, "mivek": f2})
    assert "p0.visibility_plus" not in con.disagreements  # abstains when vis is absent
    assert con.periods[0]["visibility_plus"] is True  # only the voice with a vis opinion votes


def test_structural_dissent_on_period_count_mismatch():
    f1 = _fc([_p(ChangeType.INITIAL, 1, 12), _p(ChangeType.FM, 1, 16)])
    f2 = _fc([_p(ChangeType.INITIAL, 1, 12)])
    con = consensus_taf({"avwx": f1, "mivek": f2})
    assert con.structural_dissent  # voices disagree on how many change groups
