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


def test_awc_voice_decodes_bundled_forecast():
    # The AWC voice reads NOAA's bundled <forecast> decode from the XML element, not the text.
    import xml.etree.ElementTree as ET

    from avtext.oracles.awc_taf_voice import decode_awc_taf

    xml = (
        "<TAF><raw_text>TAF KXXX 011130Z 0112/0218 18008KT 6SM BKN035</raw_text>"
        "<forecast><fcst_time_from>2026-08-01T12:00:00Z</fcst_time_from>"
        "<fcst_time_to>2026-08-02T18:00:00Z</fcst_time_to>"
        "<wind_dir_degrees>180</wind_dir_degrees><wind_speed_kt>8</wind_speed_kt>"
        "<visibility_statute_mi>6+</visibility_statute_mi>"
        '<sky_condition sky_cover="BKN" cloud_base_ft_agl="3500"/></forecast></TAF>'
    )
    f = decode_awc_taf(ET.fromstring(xml))
    assert f.station == "KXXX" and f.issue_hour == 11
    assert f.periods[0].change_type is ChangeType.INITIAL
    assert f.periods[0].wind.direction == 180 and f.periods[0].wind.speed_kt == 8
    assert f.periods[0].clouds[0].base_ft == 3500
    assert f.periods[0].visibility_plus is True  # '6+' = P6SM


def test_classify_taf_labels_and_reference():
    from avtext.harness.hardcases import CLEAN, PARSE_FAIL
    from avtext.harness.hardcases_taf import classify_taf

    c = classify_taf(KORD)  # both voices decode + agree
    assert c.label == CLEAN
    assert c.reference["header"]["station"] == "KORD"
    assert len(c.reference["periods"]) >= 2
    assert not c.failed

    junk = classify_taf("GARBAGE NOT A TAF @@@")  # neither voice decodes
    assert junk.label == PARSE_FAIL
    assert junk.reference["periods"] == []


def test_taf_scoring_roundtrip_is_perfect():
    # reference -> SFT target JSON -> parsed prediction -> score must be perfect (EM, no halluc).
    # This is the load-bearing consistency check across prompt_taf / build_sft_taf / score_taf.
    from avtext.finetune.build_sft_taf import _target_json
    from avtext.harness.hardcases_taf import classify_taf
    from avtext.harness.prompt_taf import parse_prediction_taf
    from avtext.harness.score import Outcome
    from avtext.harness.score_taf import period_count_match, score_taf_record

    ref = classify_taf(KORD).reference
    pred = parse_prediction_taf("JSON:\n" + _target_json(ref))  # model emits exactly the target
    s = score_taf_record(ref, pred)
    assert s.exact_match
    assert period_count_match(ref, pred)
    assert not any(o in (Outcome.WRONG, Outcome.HALLUCINATE) for o in s.outcomes.values())


def test_taf_scoring_penalises_extra_and_missing_periods():
    from avtext.harness.score import Outcome
    from avtext.harness.score_taf import score_taf_record

    ref = {"header": {"station": "KXXX"}, "periods": [{"change_type": "INITIAL", "wind_speed": 8}]}
    invents = {
        "header": {"station": "KXXX"},
        "periods": [
            {"change_type": "INITIAL", "wind_speed": 8},
            {"change_type": "FM", "wind_speed": 10},  # a group that isn't in the reference
        ],
    }
    s_extra = score_taf_record(ref, invents)
    assert any(o is Outcome.HALLUCINATE for o in s_extra.outcomes.values())  # invented group
    assert not s_extra.exact_match

    drops = {"header": {"station": "KXXX"}, "periods": []}
    s_missing = score_taf_record(ref, drops)
    assert any(o is Outcome.ABSTAIN for o in s_missing.outcomes.values())  # dropped group = safe
