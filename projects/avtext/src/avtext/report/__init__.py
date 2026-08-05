"""Reporting layer — collect every run into one benchmark table and render the dashboard.

Kept separate from `harness` (which produces per-run reports) because this is the *cross-run*
view: it reads all the frozen run.json files and rolls them into the single comparison the
project's findings are read from.
"""
