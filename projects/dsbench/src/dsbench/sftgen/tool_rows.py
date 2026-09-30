"""Tool rows for Revision 2 (ADR-004 Revision 2, action item 4): prompts with a tool list, and the
check the base's reply must pass.

WHY: every Gate-2 row that offered tools opened with a call, and the adapter learned to call tools
that didn't fit (BFCL irrelevance 89.2 -> 67.1). Revision 2 trains both halves, from the base's own
replies, thinking on:
- **fit** rows: a request that one tool in the list answers. Kept if the base's call matches the
  gold call on function, arguments and values: the rules BFCL's AST checker applies;
- **decline** rows: the same tool lists with a request no tool serves, either an action the list
  can't do (another domain's tool) or a question that needs no tool. Kept if the base calls
  nothing.

Everything here is written for this generator: the tools, their schemas, the requests and the
values. No BFCL item or schema is used, and every item passes dsbench's gate and the battery's
(no shared 13-gram at all, not only rule 4's fifth) before it is written. Teacher-free, seeded,
so Apache-2.0 like Target A.

Requests state every value the call needs, in words a person would use ("7:30 pm", "euros",
"March 4, 2027"); the schema asks for a format ("HH:MM", ISO 4217, YYYY-MM-DD), so a fit row also
checks that the base maps one to the other. Values that could be written two ways are accepted
both ways (a city with or without its country, a time with or without seconds).

    uv run python -m dsbench.sftgen.tool_rows generate --battery-items data/battery/items \\
        --out data/sft/rev2_tool_prompts.jsonl --report data/sft/rev2_tool_prompts_manifest.json
    uv run python -m dsbench.sftgen.tool_rows verify --items <items> --gen <gen> --out <verified>

The items use the reasoning pilot's format plus `tools` and `verify`, and `reasoning_pilot.py
generate` sends the tools and streams the reply, as the battery does for BFCL.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import string
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from dsbench.sftgen.decontaminate import strict_gate

SEED = 20260930
# ADR-004's table: 250 fit rows and 200 decline rows kept, divided by the share expected to survive
# the check and the length limits (the base declined 89.2% of BFCL's irrelevance items).
FIT_ROWS, DECLINE_ROWS = 250, 200
FIT_KEEP, DECLINE_KEEP = 0.85, 0.90
# How many tools a list offers: mostly a few, sometimes one, rarely five.
LIST_SIZES = {1: 0.15, 2: 0.25, 3: 0.30, 4: 0.20, 5: 0.10}
OTHER_TOOL_SHARE = 0.6  # of decline rows: an action the list can't do; the rest need no tool


@dataclass(frozen=True)
class Val:
    value: Any  # what the call must carry
    say: str  # how the request words it
    also: tuple = ()  # other values the check accepts


@dataclass(frozen=True)
class Template:
    text: str  # {param} slots, filled with each value's `say`
    fixed: dict = field(default_factory=dict)  # values the wording implies ("open right now")


@dataclass(frozen=True)
class Tool:
    name: str
    domain: str
    description: str
    properties: dict  # JSON-schema properties
    required: tuple[str, ...]
    draw: Callable[[random.Random], dict[str, Val]]
    templates: tuple[Template, ...]
    broad: bool = False  # can answer almost anything (web search): never in a decline list

    def schema(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.properties,
                           "required": list(self.required)}}}


# --- values ----------------------------------------------------------------------------------

CITIES = (("Lisbon", "Portugal"), ("Porto", "Portugal"), ("Valencia", "Spain"),
          ("Seville", "Spain"), ("Lyon", "France"), ("Marseille", "France"), ("Munich", "Germany"),
          ("Hamburg", "Germany"), ("Vienna", "Austria"), ("Zurich", "Switzerland"),
          ("Oslo", "Norway"), ("Helsinki", "Finland"), ("Gdansk", "Poland"),
          ("Wroclaw", "Poland"), ("Prague", "Czechia"), ("Budapest", "Hungary"),
          ("Dublin", "Ireland"), ("Edinburgh", "United Kingdom"), ("Vancouver", "Canada"),
          ("Denver", "United States"), ("Austin", "United States"), ("Osaka", "Japan"),
          ("Busan", "South Korea"), ("Melbourne", "Australia"), ("Auckland", "New Zealand"),
          ("Nairobi", "Kenya"), ("Montevideo", "Uruguay"), ("Bogota", "Colombia"))
AIRPORTS = (("LIS", "Lisbon"), ("OPO", "Porto"), ("VLC", "Valencia"), ("SVQ", "Seville"),
            ("VIE", "Vienna"), ("ZRH", "Zurich"), ("OSL", "Oslo"), ("HEL", "Helsinki"),
            ("GDN", "Gdansk"), ("WRO", "Wroclaw"), ("PRG", "Prague"), ("BUD", "Budapest"),
            ("DUB", "Dublin"), ("EDI", "Edinburgh"), ("YVR", "Vancouver"), ("DEN", "Denver"),
            ("AUS", "Austin"), ("AKL", "Auckland"), ("NBO", "Nairobi"), ("MVD", "Montevideo"))
CURRENCIES = (("EUR", "euros"), ("USD", "US dollars"), ("GBP", "British pounds"),
              ("JPY", "Japanese yen"), ("CHF", "Swiss francs"), ("PLN", "Polish zloty"),
              ("SEK", "Swedish kronor"), ("CAD", "Canadian dollars"),
              ("AUD", "Australian dollars"), ("NOK", "Norwegian kroner"),
              ("CZK", "Czech koruna"), ("MXN", "Mexican pesos"))
LANGUAGES = (("es", "Spanish"), ("fr", "French"), ("de", "German"), ("pl", "Polish"),
             ("pt", "Portuguese"), ("it", "Italian"), ("nl", "Dutch"), ("sv", "Swedish"),
             ("ja", "Japanese"), ("tr", "Turkish"), ("cs", "Czech"), ("ko", "Korean"))
UNITS = {"length": ("meter", "kilometer", "centimeter", "mile", "foot", "inch"),
         "mass": ("kilogram", "gram", "pound", "ounce"),
         "volume": ("liter", "milliliter", "gallon")}
PLURAL = {"foot": "feet", "inch": "inches"}
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "ten")


def _choice(rng: random.Random, *pairs: tuple[Any, tuple[str, ...]]) -> Val:
    """One of (value, ways to say it)."""
    value, says = rng.choice(pairs)
    return Val(value, rng.choice(says))


def _city(rng: random.Random) -> Val:
    name, country = rng.choice(CITIES)
    return Val(name, name, (f"{name}, {country}",))


def _date(rng: random.Random, after: date | None = None) -> tuple[date, Val]:
    """A day in 2027, or 1-9 days after `after` (a hotel's check-out)."""
    if after is None:
        d = date(2027, 1, 4) + timedelta(days=rng.randrange(330))
    else:
        d = after + timedelta(days=rng.randrange(1, 10))
    forms = (f"{d:%B} {d.day}, {d.year}", f"{d.day} {d:%B} {d.year}", d.isoformat())
    return d, Val(d.isoformat(), rng.choice(forms))


def _time(rng: random.Random) -> Val:
    h, m = rng.randrange(7, 22), rng.choice((0, 15, 30, 45))
    hhmm = f"{h:02d}:{m:02d}"
    half = "am" if h < 12 else "pm"
    twelve = f"{(h - 1) % 12 + 1}:{m:02d} {half}" if m else f"{(h - 1) % 12 + 1} {half}"
    return Val(hhmm, rng.choice((hhmm, twelve)), (f"{hhmm}:00",))


def _count(rng: random.Random, lo: int, hi: int) -> Val:
    n = rng.randint(lo, hi)
    return Val(n, rng.choice((str(n), NUMBER_WORDS[n])) if n <= 10 else str(n))


def _amount(rng: random.Random, options: tuple) -> Val:
    x = rng.choice(options)
    return Val(x, f"{x:,}" if isinstance(x, int) else str(x))


def _currency(rng: random.Random, exclude: str = "") -> Val:
    code, name = rng.choice([c for c in CURRENCIES if c[0] != exclude])
    return Val(code, rng.choice((code, name)))


def _quoted(rng: random.Random, options: tuple[str, ...]) -> Val:
    text = rng.choice(options)
    return Val(text, text)


def _airports(rng: random.Random) -> tuple[Val, Val]:
    (a, city_a), (b, city_b) = rng.sample(AIRPORTS, 2)
    return (Val(a, rng.choice((a, f"{city_a} ({a})"))),
            Val(b, rng.choice((b, f"{city_b} ({b})"))))


def _units(rng: random.Random) -> dict[str, Val]:
    kind = rng.choice(sorted(UNITS))
    a, b = rng.sample(UNITS[kind], 2)
    x = rng.choice((2, 5, 12, 26.2, 100, 3.5, 0.75))
    plural_a = PLURAL.get(a, a + "s") if x != 1 else a
    return {"quantity": Val(x, str(x)), "source_unit": Val(a, plural_a),
            "target_unit": Val(b, PLURAL.get(b, b + "s"))}


def _phone(rng: random.Random) -> Val:
    prefix, groups = rng.choice((("+351", (3, 3, 3)), ("+48", (3, 3, 3)), ("+49", (3, 4, 4)),
                                 ("+44", (4, 3, 3)), ("+1", (3, 3, 4))))
    parts = ["".join(rng.choice(string.digits[1:]) for _ in range(n)) for n in groups]
    return Val(prefix + "".join(parts), " ".join((prefix, *parts)))


def _tracking(rng: random.Random) -> Val:
    code = "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(12))
    return Val(code, code)


# --- the catalog -----------------------------------------------------------------------------

def _p(kind: str, description: str, **extra) -> dict:
    return {"type": kind, "description": description, **extra}


_TEMP_UNITS = (("celsius", ("Celsius", "degrees Celsius")),
               ("fahrenheit", ("Fahrenheit", "degrees Fahrenheit")))

TOOLS: tuple[Tool, ...] = (
    Tool("get_current_weather", "weather",
         "Return the current conditions (temperature, sky, wind) for a city.",
         {"city": _p("string", "Name of the city, for example Lisbon."),
          "units": _p("string", "Temperature scale for the result.",
                      enum=["celsius", "fahrenheit"], default="celsius")},
         ("city",),
         lambda rng: {"city": _city(rng), "units": _choice(rng, *_TEMP_UNITS)},
         (Template("What's the weather like in {city} right now?"),
          Template("Tell me the current temperature in {city} in {units}."),
          Template("Is it raining in {city} at the moment? Give me the temperature in {units}."))),
    Tool("get_weather_forecast", "weather",
         "Forecast daily weather for a city over the coming days.",
         {"city": _p("string", "Name of the city, for example Lisbon."),
          "days": _p("integer", "How many days ahead to cover, from 1 to 14."),
          "units": _p("string", "Temperature scale for the result.",
                      enum=["celsius", "fahrenheit"], default="celsius")},
         ("city", "days"),
         lambda rng: {"city": _city(rng), "days": _count(rng, 2, 10),
                      "units": _choice(rng, *_TEMP_UNITS)},
         (Template("What will the weather be in {city} over the next {days} days?"),
          Template("Give me a {days}-day forecast for {city}, in {units}."),
          Template("I'm travelling to {city}. Show me the forecast for the next {days} days."))),
    Tool("get_local_time", "time", "Give the current local time in a city.",
         {"city": _p("string", "Name of the city, for example Lisbon.")},
         ("city",),
         lambda rng: {"city": _city(rng)},
         (Template("What time is it in {city} right now?"),
          Template("Can you check the local time in {city}?"))),
    Tool("convert_currency", "currency",
         "Convert an amount of money from one currency to another at today's rate.",
         {"amount": _p("number", "The amount to convert."),
          "source_currency": _p("string", "ISO 4217 code of the money you have, such as EUR."),
          "target_currency": _p("string", "ISO 4217 code of the money you want, such as JPY.")},
         ("amount", "source_currency", "target_currency"),
         lambda rng: (lambda a: {"amount": _amount(rng, (20, 35, 80, 120, 250, 640, 1500, 12.5)),
                                 "source_currency": a,
                                 "target_currency": _currency(rng, exclude=a.value)})(
             _currency(rng)),
         (Template("How much is {amount} {source_currency} in {target_currency}?"),
          Template("Convert {amount} {source_currency} to {target_currency}, please."),
          Template("I have {amount} {source_currency}. What is that worth in "
                   "{target_currency}?"))),
    Tool("get_exchange_rate", "currency", "Fetch the current exchange rate between two currencies.",
         {"base_currency": _p("string", "ISO 4217 code of the currency being priced."),
          "quote_currency": _p("string", "ISO 4217 code the price is given in.")},
         ("base_currency", "quote_currency"),
         lambda rng: (lambda a: {"base_currency": a,
                                 "quote_currency": _currency(rng, exclude=a.value)})(
             _currency(rng)),
         (Template("What's the exchange rate from {base_currency} to {quote_currency} today?"),
          Template("Look up today's {base_currency} to {quote_currency} rate."))),
    Tool("convert_units", "units", "Convert a measurement between two units of the same kind.",
         {"quantity": _p("number", "The measurement to convert."),
          "source_unit": _p("string", "Unit the measurement is in.",
                            enum=[u for kind in sorted(UNITS) for u in UNITS[kind]]),
          "target_unit": _p("string", "Unit to express the result in.",
                            enum=[u for kind in sorted(UNITS) for u in UNITS[kind]])},
         ("quantity", "source_unit", "target_unit"),
         _units,
         (Template("Convert {quantity} {source_unit} to {target_unit}."),
          Template("How many {target_unit} are in {quantity} {source_unit}?"))),
    Tool("search_flights", "travel",
         "Search scheduled flights between two airports on a given day.",
         {"origin": _p("string", "IATA code of the departure airport, e.g. LIS."),
          "destination": _p("string", "IATA code of the arrival airport, e.g. OSL."),
          "departure_date": _p("string", "Departure date in YYYY-MM-DD format."),
          "passengers": _p("integer", "Number of travellers.", default=1),
          "cabin_class": _p("string", "Cabin to search.", default="economy",
                            enum=["economy", "premium_economy", "business", "first"])},
         ("origin", "destination", "departure_date"),
         lambda rng: (lambda od: {
             "origin": od[0], "destination": od[1], "departure_date": _date(rng)[1],
             "passengers": _count(rng, 2, 6),
             "cabin_class": _choice(rng, ("economy", ("economy",)),
                                    ("premium_economy", ("premium economy",)),
                                    ("business", ("business class", "business")),
                                    ("first", ("first class",)))})(_airports(rng)),
         (Template("Find flights from {origin} to {destination} on {departure_date}."),
          Template("I need {passengers} seats in {cabin_class} from {origin} to {destination} on "
                   "{departure_date}."),
          Template("Search for a {cabin_class} flight from {origin} to {destination} departing "
                   "{departure_date}."))),
    Tool("search_hotels", "lodging", "List available hotels in a city for a stay.",
         {"city": _p("string", "Name of the city, for example Lisbon."),
          "check_in": _p("string", "Arrival date in YYYY-MM-DD format."),
          "check_out": _p("string", "Departure date in YYYY-MM-DD format."),
          "guests": _p("integer", "Number of guests.", default=1),
          "max_price_per_night": _p("number", "Upper price limit per night, in euros.")},
         ("city", "check_in", "check_out"),
         lambda rng: (lambda d_in: {
             "city": _city(rng), "check_in": d_in[1], "check_out": _date(rng, after=d_in[0])[1],
             "guests": _count(rng, 2, 5), "max_price_per_night": _amount(rng, (90, 120, 150, 200)),
         })(_date(rng)),
         (Template("Find a hotel in {city} from {check_in} to {check_out}."),
          Template("Look for hotels in {city} for {guests} guests, checking in {check_in} and out "
                   "{check_out}."),
          Template("I need a room in {city} between {check_in} and {check_out}, under "
                   "{max_price_per_night} euros a night."))),
    Tool("get_travel_time", "maps", "Estimate how long a trip between two places takes.",
         {"origin": _p("string", "Starting place or address."),
          "destination": _p("string", "Place or address to reach."),
          "mode": _p("string", "How the trip is made.", default="driving",
                     enum=["driving", "walking", "cycling", "transit"])},
         ("origin", "destination"),
         lambda rng: (lambda od: {
             "origin": Val(od[0], od[0]), "destination": Val(od[1], od[1]),
             "mode": _choice(rng, ("driving", ("car", "driving")), ("walking", ("on foot",)),
                             ("cycling", ("bike", "cycling")),
                             ("transit", ("public transport", "transit")))})(
             rng.choice((("Porto", "Braga"), ("Madrid", "Toledo"), ("Lyon", "Grenoble"),
                         ("Munich", "Salzburg"), ("Vienna", "Bratislava"), ("Oslo", "Drammen"),
                         ("Seattle", "Tacoma"), ("Osaka", "Kyoto"), ("Melbourne", "Geelong")))),
         (Template("How long does it take to get from {origin} to {destination} by {mode}?"),
          Template("Estimate the travel time from {origin} to {destination}."))),
    Tool("find_restaurants", "dining", "Search restaurants in a city, optionally by cuisine.",
         {"city": _p("string", "Name of the city, for example Lisbon."),
          "cuisine": _p("string", "Kind of food, e.g. Thai."),
          "open_now": _p("boolean", "Only places open at the moment.", default=False)},
         ("city",),
         lambda rng: {"city": _city(rng), "cuisine": _quoted(rng, (
             "Portuguese", "Thai", "Ethiopian", "Georgian", "Peruvian", "Lebanese", "Korean"))},
         (Template("Find {cuisine} restaurants in {city}."),
          Template("Are there any {cuisine} places open right now in {city}?",
                   fixed={"open_now": True}),
          Template("Show me some restaurants in {city}."))),
    Tool("book_table", "dining", "Reserve a table at a restaurant.",
         {"restaurant": _p("string", "Name of the restaurant."),
          "date": _p("string", "Date in YYYY-MM-DD format."),
          "time": _p("string", "Time in 24-hour HH:MM format."),
          "party_size": _p("integer", "Number of people.")},
         ("restaurant", "date", "time", "party_size"),
         lambda rng: {"restaurant": _quoted(rng, ("Casa Verde", "The Copper Pot", "Sakura Garden",
                                                  "Trattoria Luna", "Blue Fig Bistro")),
                      "date": _date(rng)[1], "time": _time(rng), "party_size": _count(rng, 2, 8)},
         (Template("Book a table for {party_size} at {restaurant} on {date} at {time}."),
          Template("Can you reserve {restaurant} for {party_size} people on {date}, {time}?"))),
    Tool("find_recipes", "cooking", "Suggest recipes that use the given ingredients.",
         {"ingredients": {"type": "array", "items": {"type": "string"},
                          "description": "Ingredients the recipe should use."},
          "max_minutes": _p("integer", "Longest acceptable total cooking time."),
          "diet": _p("string", "Dietary restriction.", default="none",
                     enum=["none", "vegetarian", "vegan", "gluten_free"])},
         ("ingredients",),
         lambda rng: (lambda xs: {
             "ingredients": Val(list(xs), f"{xs[0]}, {xs[1]} and {xs[2]}"),
             "max_minutes": _choice(rng, (20, ("20",)), (30, ("30",)), (45, ("45",))),
             "diet": _choice(rng, ("vegetarian", ("vegetarian",)), ("vegan", ("vegan",)),
                             ("gluten_free", ("gluten-free",)))})(
             rng.choice((("chickpeas", "spinach", "feta"), ("salmon", "potatoes", "dill"),
                         ("tofu", "broccoli", "ginger"), ("mushrooms", "barley", "thyme"),
                         ("eggplant", "tomatoes", "garlic"), ("lentils", "carrots", "cumin")))),
         (Template("What can I cook with {ingredients}?"),
          Template("Give me a {diet} recipe using {ingredients} that takes under {max_minutes} "
                   "minutes."))),
    Tool("create_calendar_event", "scheduling", "Add an event to the user's calendar.",
         {"title": _p("string", "Short name of the event."),
          "date": _p("string", "Date in YYYY-MM-DD format."),
          "start_time": _p("string", "Start time in 24-hour HH:MM format."),
          "duration_minutes": _p("integer", "Length of the event.", default=60),
          "location": _p("string", "Where the event takes place.")},
         ("title", "date", "start_time"),
         lambda rng: {"title": _quoted(rng, ("Team retrospective", "Dentist appointment",
                                             "Piano lesson", "Quarterly budget review",
                                             "Lunch with Marta", "Book club")),
                      "date": _date(rng)[1], "start_time": _time(rng),
                      "duration_minutes": _choice(rng, (30, ("30",)), (45, ("45",)),
                                                  (90, ("90",))),
                      "location": _quoted(rng, ("Room 4B", "the Alfama office", "Cafe Nova"))},
         (Template("Add \"{title}\" to my calendar on {date} at {start_time}."),
          Template("Schedule \"{title}\" on {date} at {start_time} for {duration_minutes} minutes, "
                   "at {location}."))),
    Tool("set_reminder", "scheduling", "Create a reminder that fires at a given date and time.",
         {"message": _p("string", "What to be reminded of."),
          "date": _p("string", "Date in YYYY-MM-DD format."),
          "time": _p("string", "Time in 24-hour HH:MM format.")},
         ("message", "date", "time"),
         lambda rng: {"message": _quoted(rng, ("water the plants", "renew my passport",
                                               "call the plumber", "back up my laptop",
                                               "pick up the dry cleaning")),
                      "date": _date(rng)[1], "time": _time(rng)},
         (Template("Set a reminder for {date} at {time}: \"{message}\"."),
          Template("Please remind me on {date} at {time} to \"{message}\"."))),
    Tool("send_email", "messaging", "Send an email message.",
         {"to": _p("string", "Recipient email address."),
          "subject": _p("string", "Subject line."),
          "body": _p("string", "Text of the message.")},
         ("to", "subject", "body"),
         lambda rng: {"to": _quoted(rng, ("ana.silva@example.com", "j.novak@example.org",
                                          "priya.raman@example.net", "lea.fischer@example.org")),
                      "subject": _quoted(rng, ("Quarterly report", "Friday's workshop",
                                               "Moving the kickoff", "Draft for review")),
                      "body": _quoted(rng, ("The report is ready for your review.",
                                            "Can we move our call to Thursday?",
                                            "Thanks again for your help last week."))},
         (Template("Email {to} with the subject \"{subject}\" and the message \"{body}\""),
          Template("Send an email to {to}. Subject: \"{subject}\". Body: \"{body}\""))),
    Tool("send_text_message", "messaging", "Send an SMS to a phone number.",
         {"phone_number": _p("string", "Number in international format, e.g. +351912345678."),
          "message": _p("string", "Text to send.")},
         ("phone_number", "message"),
         lambda rng: {"phone_number": _phone(rng),
                      "message": _quoted(rng, ("Running ten minutes late.",
                                               "The keys are under the mat.",
                                               "See you at the station at six."))},
         (Template("Text {phone_number}: \"{message}\""),
          Template("Send an SMS to {phone_number} saying \"{message}\""))),
    Tool("translate_text", "language", "Translate text into another language.",
         {"text": _p("string", "The text to translate."),
          "target_language": _p("string", "ISO 639-1 code of the language to translate into.")},
         ("text", "target_language"),
         lambda rng: {"text": _quoted(rng, ("Where is the train station?",
                                            "The museum opens at nine.",
                                            "Could you recommend a good bakery?",
                                            "I would like a table by the window.")),
                      "target_language": (lambda code_name: Val(code_name[0], code_name[1]))(
                          rng.choice(LANGUAGES))},
         (Template("Translate \"{text}\" into {target_language}."),
          Template("How do you say \"{text}\" in {target_language}? Use the translator."))),
    Tool("define_word", "language", "Return the dictionary definition of an English word.",
         {"word": _p("string", "The word to define.")},
         ("word",),
         lambda rng: {"word": _quoted(rng, ("serendipity", "ephemeral", "laconic", "quixotic",
                                            "meticulous", "gregarious"))},
         (Template("Look up the definition of \"{word}\"."),
          Template("Check the dictionary entry for the word \"{word}\"."))),
    Tool("get_stock_quote", "finance", "Return the latest trading price for a stock ticker.",
         {"ticker": _p("string", "Exchange ticker symbol, e.g. MSFT.")},
         ("ticker",),
         lambda rng: {"ticker": _quoted(rng, ("ASML", "SAP", "NVDA", "TSM", "KO", "NKE", "SHOP"))},
         (Template("What's {ticker} trading at right now?"),
          Template("Get me the latest quote for {ticker}."))),
    Tool("calculate_loan_payment", "finance",
         "Compute the fixed monthly payment of an amortising loan.",
         {"principal": _p("number", "Amount borrowed."),
          "annual_rate_percent": _p("number", "Yearly interest rate in percent, e.g. 4.5."),
          "years": _p("integer", "Length of the loan in years.")},
         ("principal", "annual_rate_percent", "years"),
         lambda rng: {"principal": _amount(rng, (15000, 24000, 180000, 250000, 320000)),
                      "annual_rate_percent": _amount(rng, (3.2, 4.5, 5.75, 6.1)),
                      "years": _choice(rng, *((y, (str(y),)) for y in (5, 10, 15, 20, 25, 30)))},
         (Template("What would the monthly payment be on a {principal} loan at "
                   "{annual_rate_percent}% over {years} years?"),
          Template("Calculate the monthly installment for borrowing {principal} at an annual rate "
                   "of {annual_rate_percent} percent for {years} years."))),
    Tool("track_package", "shipping", "Get the delivery status of a parcel.",
         {"tracking_number": _p("string", "The parcel's tracking number."),
          "carrier": _p("string", "Carrier that ships it.",
                        enum=["dhl", "ups", "fedex", "usps", "gls"])},
         ("tracking_number", "carrier"),
         lambda rng: {"tracking_number": _tracking(rng),
                      "carrier": _choice(rng, ("dhl", ("DHL",)), ("ups", ("UPS",)),
                                         ("fedex", ("FedEx",)), ("usps", ("USPS",)),
                                         ("gls", ("GLS",)))},
         (Template("Where is my {carrier} package {tracking_number}?"),
          Template("Track the parcel {tracking_number}. It was sent with {carrier}."))),
    Tool("set_thermostat", "home", "Set the target temperature of a thermostat.",
         {"room": _p("string", "Room the thermostat is in."),
          "temperature": _p("number", "Target temperature."),
          "unit": _p("string", "Scale of the temperature.", enum=["celsius", "fahrenheit"],
                     default="celsius")},
         ("room", "temperature"),
         lambda rng: (lambda unit: {
             "room": _quoted(rng, ("living room", "bedroom", "office", "kitchen", "nursery")),
             "temperature": _amount(rng, (19, 20, 21, 22.5) if unit.value == "celsius"
                                    else (66, 68, 70, 72)),
             "unit": unit})(_choice(rng, *_TEMP_UNITS)),
         (Template("Set the {room} thermostat to {temperature} degrees {unit}."),
          Template("Make the {room} {temperature} degrees {unit}, please."))),
    Tool("set_lights", "home", "Switch the lights in a room on or off, optionally dimmed.",
         {"room": _p("string", "Room whose lights to change."),
          "on": _p("boolean", "True to switch on, false to switch off."),
          "brightness_percent": _p("integer", "Brightness from 1 to 100.")},
         ("room", "on"),
         lambda rng: {"room": _quoted(rng, ("living room", "bedroom", "office", "kitchen",
                                            "hallway")),
                      "brightness_percent": _choice(rng, (20, ("20",)), (35, ("35",)),
                                                    (75, ("75",)))},
         (Template("Turn off the lights in the {room}.", fixed={"on": False}),
          Template("Turn on the {room} lights at {brightness_percent}% brightness.",
                   fixed={"on": True}),
          Template("Switch the {room} lights on.", fixed={"on": True}))),
    Tool("log_workout", "fitness", "Record a completed workout.",
         {"activity": _p("string", "Kind of workout.",
                         enum=["running", "cycling", "swimming", "rowing", "yoga", "strength"]),
          "duration_minutes": _p("integer", "How long it lasted."),
          "date": _p("string", "Date in YYYY-MM-DD format."),
          "distance_km": _p("number", "Distance covered, for workouts that have one.")},
         ("activity", "duration_minutes", "date"),
         lambda rng: {"activity": _choice(rng, *((a, (a,)) for a in ("running", "cycling",
                                                                     "swimming", "rowing"))),
                      "duration_minutes": _choice(rng, *((d, (str(d),))
                                                         for d in (20, 30, 45, 60, 90))),
                      "date": _date(rng)[1],
                      "distance_km": _amount(rng, (2.5, 5, 8.5, 10, 21.1))},
         (Template("Log a {duration_minutes}-minute {activity} workout on {date}."),
          Template("Record my {activity} on {date}: {duration_minutes} minutes and {distance_km} "
                   "km."))),
    Tool("get_news_headlines", "news", "Fetch top headlines, optionally for a topic and country.",
         {"topic": _p("string", "Subject to filter by."),
          "country": _p("string", "ISO 3166-1 alpha-2 code of the country, e.g. PT."),
          "limit": _p("integer", "How many headlines to return.", default=5)},
         (),
         lambda rng: {"topic": _quoted(rng, ("renewable energy", "space exploration", "football",
                                             "housing")),
                      "country": _choice(rng, ("PT", ("Portugal",)), ("DE", ("Germany",)),
                                         ("JP", ("Japan",)), ("BR", ("Brazil",)),
                                         ("KE", ("Kenya",))),
                      "limit": _choice(rng, (3, ("3", "three")), (10, ("10", "ten")))},
         (Template("Show me the top {limit} headlines about {topic}."),
          Template("What are today's {topic} headlines in {country}?"),
          Template("Get me the latest news headlines."))),
    Tool("read_file", "files", "Read a text file from the user's workspace.",
         {"path": _p("string", "Path relative to the workspace root.")},
         ("path",),
         lambda rng: {"path": _quoted(rng, ("notes/2027-03-04-standup.md", "data/sales_q3.csv",
                                            "config/settings.yaml", "reports/draft_v2.txt"))},
         (Template("Open {path} and show me what's inside."),
          Template("Read the file {path}."))),
    Tool("list_directory", "files", "List the entries of a directory in the workspace.",
         {"path": _p("string", "Directory path relative to the workspace root."),
          "include_hidden": _p("boolean", "Also list hidden entries.", default=False)},
         ("path",),
         lambda rng: {"path": _quoted(rng, ("data", "reports/archive", "src/pipeline",
                                            "notebooks"))},
         (Template("What files are in {path}?"),
          Template("List everything in {path}, hidden files included.",
                   fixed={"include_hidden": True}))),
    Tool("run_sql_query", "data", "Run a read-only SQL query on a named database and return rows.",
         {"database": _p("string", "Name of the database."),
          "query": _p("string", "The SQL to run.")},
         ("database", "query"),
         lambda rng: {"database": _quoted(rng, ("analytics", "shop", "crm")),
                      "query": (lambda q: Val(q, q, (q + ";",)))(rng.choice((
                          "SELECT region, SUM(revenue) FROM sales GROUP BY region",
                          "SELECT COUNT(*) FROM customers WHERE country = 'PT'",
                          "SELECT order_date, total FROM orders WHERE total > 500 LIMIT 20")))},
         (Template("Run this query on the {database} database: {query}"),
          Template("On {database}, execute: {query}"))),
    Tool("describe_table", "data", "Show the columns and types of a table.",
         {"database": _p("string", "Name of the database."),
          "table": _p("string", "Name of the table.")},
         ("database", "table"),
         lambda rng: {"database": _quoted(rng, ("analytics", "shop", "crm")),
                      "table": _quoted(rng, ("orders", "customers", "invoices", "shipments"))},
         (Template("What columns does the {table} table in the {database} database have?"),
          Template("Show me the structure of the {table} table in {database}."))),
    Tool("search_web", "web", "Search the web and return result snippets.",
         {"query": _p("string", "What to search for."),
          "max_results": _p("integer", "How many results to return.", default=5)},
         ("query",),
         lambda rng: {"query": _quoted(rng, ("best time of year to visit the Azores",
                                             "how to repot an orchid without damaging the roots",
                                             "history of the tram network in Lisbon",
                                             "how do heat pumps work in cold climates")),
                      "max_results": _choice(rng, (3, ("3", "three")), (8, ("8", "eight")))},
         (Template("Search the web for \"{query}\"."),
          Template("Look up \"{query}\" online and give me the top {max_results} results.")),
         broad=True),
)
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# Requests no tool serves, for decline rows. The third field names a domain whose tools the
# request is too close to; such a request is never paired with that domain's tools, so a decline
# stays a decline.
GENERAL: tuple[tuple[str, tuple[str, ...], str | None], ...] = (
    ("Explain in a few sentences how {x} works.",
     ("a rainbow", "a vaccine", "noise-cancelling headphones", "a refrigerator"), None),
    ("Write a short poem about {x}.",
     ("a lighthouse in winter", "the first warm day of spring", "an old bicycle",
      "a quiet library at night"), None),
    ("Give me three practical tips for {x}.",
     ("keeping houseplants alive", "staying focused while studying",
      "preparing for a job interview", "sleeping better"), None),
    ("What is the difference between {x}?",
     ("weather and climate", "a virus and a bacterium", "a lake and a pond",
      "empathy and sympathy"), None),
    ("Summarize the main ideas of {x} in three bullet points.",
     ("the scientific method", "supply and demand", "the water cycle", "natural selection"), None),
    ("Suggest five names for {x}.",
     ("a bakery that only sells sourdough", "a cat that loves cardboard boxes",
      "a hiking club for beginners"), None),
    ("Rewrite this sentence so it sounds more formal: \"{x}\"",
     ("hey, can u send me the file asap", "we gotta push the meeting to next week",
      "thx for helping out yesterday!"), None),
    ("Write a three-question quiz about {x}, with the answers.",
     ("the solar system", "the human skeleton", "the printing press"), None),
    ("Tell me an interesting fact about {x}.", ("octopuses", "honeybees", "volcanoes"), None),
    ("What are the pros and cons of {x}?",
     ("working from home", "learning two languages at once", "living in a small town"), None),
    ("How would you explain {x} to a ten-year-old?",
     ("gravity", "electricity", "why we need sleep"), None),
    ("Continue a short story that begins: \"{x}\"",
     ("The last train had already left.", "Nobody had opened the attic in twenty years."), None),
    ("What would happen if {x}?",
     ("the moon were twice as close to Earth", "a city banned cars from its centre"), None),
    ("Is {x} a good idea? Give me a balanced answer.",
     ("taking a gap year before university", "switching careers at forty",
      "keeping a daily journal"), None),
    ("Help me word a polite note to my neighbour about {x}.",
     ("their dog barking at night", "a parcel left at my door by mistake"), "messaging"),
    ("Plan a simple weekly routine for someone who wants to {x}.",
     ("start running", "get stronger at home"), "fitness"),
    ("What should I keep in mind when cooking {x} for the first time?",
     ("risotto", "a whole fish", "fresh pasta"), "cooking"),
    ("Why is {x}?", ("the sky blue during the day", "sea water salty", "ice slippery"), None),
)


# --- generation ------------------------------------------------------------------------------

_SLOT = re.compile(r"{(\w+)}")


def slots(template: Template) -> list[str]:
    return _SLOT.findall(template.text)


def gold_call(tool: Tool, vals: dict[str, Val], template: Template) -> dict:
    """{name, arguments: {param: accepted values}}; "" accepts leaving an optional one out."""
    stated = set(slots(template))
    args: dict[str, list] = {}
    for name, spec in tool.properties.items():
        if name in template.fixed:
            args[name] = [template.fixed[name]]
        elif name in stated:
            args[name] = [vals[name].value, *vals[name].also]
        else:
            args[name] = [""] + ([spec["default"]] if "default" in spec else [])
    return {"name": tool.name, "arguments": args}


def render(tool: Tool, vals: dict[str, Val], template: Template) -> str:
    return template.text.format(**{name: vals[name].say for name in slots(template)})


def _tool_list(rng: random.Random, target: Tool) -> list[Tool]:
    k = rng.choices(list(LIST_SIZES), weights=list(LIST_SIZES.values()))[0]
    others = [t for t in TOOLS if t is not target]
    rng.shuffle(others)
    listed = [target, *others[: k - 1]]
    rng.shuffle(listed)
    return listed


def _decline_request(rng: random.Random, listed: list[Tool]) -> tuple[str, dict]:
    domains = {t.domain for t in listed}
    if rng.random() < OTHER_TOOL_SHARE:
        tool = rng.choice([t for t in TOOLS if t.domain not in domains and not t.broad])
        template = rng.choice(tool.templates)
        return (render(tool, tool.draw(rng), template),
                {"decline": "other_tool", "asks_for": tool.name})
    text, fills, near = rng.choice([g for g in GENERAL if g[2] not in domains])
    return text.format(x=rng.choice(fills)), {"decline": "no_tool"}


def generate(*, seed: int = SEED, n_fit: int | None = None, n_decline: int | None = None,
             gate: Callable[[dict], str | None] | None = None) -> tuple[list[dict], dict]:
    """(items, report). `gate(item)` names why an item is rejected, or returns None.

    Fit targets cycle through the tools, so each is asked for about as often. Decline row i offers
    fit row i's tool list, skipping lists with a broad tool: the same tools, called and not.
    """
    n_fit = n_fit if n_fit is not None else math.ceil(FIT_ROWS / FIT_KEEP)
    n_decline = n_decline if n_decline is not None else math.ceil(DECLINE_ROWS / DECLINE_KEEP)
    rng = random.Random(seed)
    order = list(TOOLS)
    rng.shuffle(order)
    rejected: Counter[str] = Counter()
    budget = 20 * (n_fit + n_decline) + 100  # a gate that rejects everything must not hang
    fit: list[dict] = []
    for target in itertools.cycle(order):
        if len(fit) == n_fit:
            break
        budget -= 1
        if budget < 0:
            raise RuntimeError(f"the gate rejected too much: {dict(rejected)}")
        listed = _tool_list(rng, target)
        template = rng.choice(target.templates)
        vals = target.draw(rng)
        item = {
            "id": f"tool_fit:{len(fit):04d}", "pool": "tool_fit", "bucket": "tools",
            "messages": [{"role": "user", "content": render(target, vals, template)}],
            "tools": [t.schema() for t in listed],
            "verify": {"kind": "call", "gold": gold_call(target, vals, template)},
            "meta": {"tool": target.name, "domain": target.domain,
                     "template": target.templates.index(template),
                     "listed": [t.name for t in listed], **_PROVENANCE},
        }
        reason = gate(item) if gate else None
        if reason:
            rejected[f"fit {target.name}: {reason}"] += 1
            continue
        fit.append(item)

    decline: list[dict] = []
    lists = [item["meta"]["listed"] for item in fit
             if not any(TOOLS_BY_NAME[n].broad for n in item["meta"]["listed"])]
    if n_decline and not lists:
        raise ValueError("no fit row's list is free of broad tools to decline with")
    for listed_names in itertools.cycle(lists or [[]]):
        if len(decline) == n_decline:
            break
        budget -= 1
        if budget < 0:
            raise RuntimeError(f"the gate rejected too much: {dict(rejected)}")
        listed = [TOOLS_BY_NAME[n] for n in listed_names]
        text, kind = _decline_request(rng, listed)
        item = {
            "id": f"tool_decline:{len(decline):04d}", "pool": "tool_decline", "bucket": "tools",
            "messages": [{"role": "user", "content": text}],
            "tools": [t.schema() for t in listed],
            "verify": {"kind": "no_call"},
            "meta": {**kind, "listed": listed_names, **_PROVENANCE},
        }
        reason = gate(item) if gate else None
        if reason:
            rejected[f"decline {kind['decline']}: {reason}"] += 1
            continue
        decline.append(item)

    items = fit + decline
    report = {
        "seed": seed, "fit": len(fit), "decline": len(decline), "tools": len(TOOLS),
        "domains": len({t.domain for t in TOOLS}),
        "fit_by_tool": dict(sorted(Counter(i["meta"]["tool"] for i in fit).items())),
        "decline_by_kind": dict(Counter(i["meta"]["decline"] for i in decline)),
        "list_sizes": dict(sorted(Counter(len(i["tools"]) for i in items).items())),
        "rejected_by_gate": dict(rejected.most_common()),
    }
    return items, report


_PROVENANCE = {"source_key": "tool_rows", "licence": "Apache-2.0 (generated here)",
               "teacher": "none (templated)"}


# --- the check -------------------------------------------------------------------------------

def standardize(text: str) -> str:
    """As BFCL's checker compares strings: case, spaces and , . / - _ * ^ don't count."""
    return re.sub(r"[ ,./\-_*^]", "", text).lower().replace('"', "'")


def _same(value: Any, accepted: Any, spec: dict) -> bool:
    kind = spec.get("type")
    if kind == "string":
        return isinstance(value, str) and standardize(value) == standardize(str(accepted))
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool) and value == accepted
    if kind == "number":  # an integer is a number, as BFCL accepts an int for a float
        return (isinstance(value, int | float) and not isinstance(value, bool)
                and float(value) == float(accepted))
    if kind == "boolean":
        return isinstance(value, bool) and value == accepted
    if kind == "array":  # the lists here are sets (ingredients): order doesn't count
        item = spec.get("items") or {}
        return (isinstance(value, list) and len(value) == len(accepted)
                and all(any(_same(v, a, item) for v in value) for a in accepted))
    return value == accepted


def decode_calls(tool_calls: list[dict]) -> tuple[list[tuple[str, dict]] | None, str]:
    """OpenAI-style tool calls -> [(name, arguments)], or (None, why) if one doesn't parse."""
    calls = []
    for call in tool_calls or []:
        fn = call.get("function") or {}
        raw = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc:
            return None, f"arguments are not JSON: {exc.msg}"
        if not fn.get("name") or not isinstance(args, dict):
            return None, "a call without a name or with non-object arguments"
        calls.append((fn["name"], args))
    return calls, ""


def check(item: dict, tool_calls: list[dict]) -> tuple[bool, str]:
    """Does the reply's tool use pass the item's check? (ok, why not)."""
    calls, why = decode_calls(tool_calls)
    if calls is None:
        return False, f"format: {why}"
    if item["verify"]["kind"] == "no_call":
        return (not calls), ("" if not calls else f"called {calls[0][0]}")
    gold = item["verify"]["gold"]
    if len(calls) != 1:
        return False, f"expected one call, got {len(calls)}"
    name, args = calls[0]
    if name != gold["name"]:
        return False, f"called {name}, not {gold['name']}"
    params = next(t["function"]["parameters"] for t in item["tools"]
                  if t["function"]["name"] == name)
    for key in args:
        if key not in params["properties"]:
            return False, f"unexpected parameter {key}"
    for key in params["required"]:
        if key not in args:
            return False, f"missing required {key}"
    for key, accepted in gold["arguments"].items():
        if key not in args:
            if "" not in accepted:
                return False, f"left out {key}, which the request states"
            continue
        spec = params["properties"][key]
        if not any(a != "" and _same(args[key], a, spec) for a in accepted):
            return False, f"{key}: {args[key]!r} is not one of {accepted}"
    return True, ""


def verify(items_path: Path, gen_path: Path, out_path: Path) -> dict:
    """Check each generated reply; a row also needs a finished reply with reasoning."""
    items = {i["id"]: i for i in map(json.loads, items_path.open())}
    passed: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    with out_path.open("w") as fh:
        for rec in map(json.loads, gen_path.open()):
            item = items[rec["id"]]
            if rec.get("error") or rec.get("finish_reason") not in ("stop", "tool_calls"):
                ok, why = False, f"unfinished: {rec.get('error') or rec.get('finish_reason')}"
            elif not (rec.get("reasoning") or "").strip():
                ok, why = False, "no reasoning"
            else:
                ok, why = check(item, rec.get("tool_calls") or [])
            (passed if ok else failed)[item["pool"]] += 1
            fh.write(json.dumps({**rec, "check": {"ok": ok, "why": why}}, ensure_ascii=False)
                     + "\n")
    return {"passed": dict(passed), "failed": dict(failed)}


# --- CLI -------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--battery-items", required=True, help="a battery run's items/ dir")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--report", type=Path, required=True)
    g.add_argument("--seed", type=int, default=SEED)
    g.add_argument("--fit", type=int, default=math.ceil(FIT_ROWS / FIT_KEEP))
    g.add_argument("--decline", type=int, default=math.ceil(DECLINE_ROWS / DECLINE_KEEP))
    v = sub.add_parser("verify")
    v.add_argument("--items", type=Path, required=True)
    v.add_argument("--gen", type=Path, required=True)
    v.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.cmd == "verify":
        print(json.dumps(verify(args.items, args.gen, args.out)))
        return
    items, report = generate(seed=args.seed, n_fit=args.fit, n_decline=args.decline,
                             gate=strict_gate(args.battery_items))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    params = {"fit_rows": FIT_ROWS, "fit_keep": FIT_KEEP, "decline_rows": DECLINE_ROWS,
              "decline_keep": DECLINE_KEEP, "list_sizes": LIST_SIZES,
              "other_tool_share": OTHER_TOOL_SHARE, "battery_items": args.battery_items}
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.report.write_text(json.dumps({"params": params, **report, "out": str(args.out),
                                       "out_sha256": digest}, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("fit", "decline", "rejected_by_gate")}))


if __name__ == "__main__":
    main()
