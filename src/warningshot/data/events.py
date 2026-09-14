"""Events, pages and windows under study.

All analytical choices that a reviewer could accuse of being fitted after the
fact live here, in one file, stated once. Nothing downstream invents a window.

Dates come from tracks/_shared/incident-record/00-incident-timeline.md.
"""


class Event:
    def __init__(self, key, label, t0, entity_pages, note=""):
        self.key = key
        self.label = label
        self.t0 = t0                    # the day attention could first respond
        self.entity_pages = entity_pages
        self.note = note


# The July incident. t0 is OpenAI's 21 Jul acknowledgement, NOT Hugging Face's
# 16 Jul disclosure: the disclosure produced no measurable movement (856 views
# against a 702 baseline), so anchoring on it would measure a non-event. This
# choice is itself one of the study's findings and is reported, not hidden.
JULY_INCIDENT = Event(
    key="july_incident",
    label="OpenAI/Hugging Face intrusion (attribution 21 Jul 2026)",
    t0="20260721",
    entity_pages={"victim": "Hugging_Face", "developer": "OpenAI"},
    note="HF disclosed 16 Jul; OpenAI acknowledged 21 Jul; reports 26 Aug.",
)

# The matched control: an access/policy story about the same model class,
# five weeks earlier, where the protagonist is a developer rather than a victim.
JUNE_BAN = Event(
    key="june_ban",
    label="US directive suspending Fable 5 / Mythos 5 access (13 Jun 2026)",
    t0="20260613",
    entity_pages={"developer": "Anthropic", "product": "Claude_(language_model)"},
    note="CeSIA's own comparison event.",
)

EVENTS = {e.key: e for e in (JULY_INCIDENT, JUNE_BAN)}

# Hazard-concept pages: the abstractions a warning shot is supposed to move.
CONCEPT_PAGES = [
    "AI_safety",
    "AI_alignment",
    "Existential_risk_from_artificial_intelligence",
    "Artificial_general_intelligence",
]

# Generic control. If this moves with the concept pages, the effect is ordinary
# AI curiosity rather than warning-shot conversion. It is what licenses any
# claim that a concept-page rise is risk-specific.
GENERIC_CONTROL = "Machine_learning"

# Conversion horizons. 48h is the window CeSIA used; the longer two exist to
# test whether that window was the right instrument.
HORIZONS = {"48h": 3, "7d": 7, "30d": 30}

# Baseline windows for the sensitivity sweep, as (label, days_before_t0).
# The sweep exists because a single window would not survive review: the
# Hugging_Face figure is baseline-invariant but the OpenAI figure moves 4.6x.
BASELINE_LENGTHS = [("14d", 14), ("30d", 30), ("60d", 60), ("90d", 90)]

# Gap between the end of a baseline window and t0, so the run-up to the event
# never contaminates its own baseline.
BASELINE_GAP_DAYS = 7

# Placebo search span: a quiet stretch before the June ban, used to build the
# null distribution for 30-day concept-page drift.
PLACEBO_SPAN = ("20260115", "20260531")
PLACEBO_STEP_DAYS = 5

# Spans excluded from the placebo null because they contain a known event.
PLACEBO_EXCLUSIONS = [("20260601", "20260901")]


# ===================================================================
# Review response C2: the entity channel is a COMPANY page, not an incident
# page, so it absorbs traffic from unrelated product news.
#
# Confirmed contamination inside the July fit window: on 2026-07-27 the story
# "Kimi-K3 on HuggingFace" scored 1,382 points on Hacker News and "Kimi-K3
# Technical Report" a further 391. `Hugging_Face` pageviews break monotone
# decay on exactly that day (3,062 -> 4,463, +46%) and stay elevated the next.
#
# These dates are declared here rather than discovered inside an analysis, so
# the exclusion is a stated pre-analysis choice with a reason attached and can
# be audited or reversed. Excluding them is reported as a sensitivity arm; the
# headline is NOT silently switched to the excluded fit.
# ===================================================================

CONTAMINATED_DATES = {
    "Hugging_Face": {
        "20260727": "Kimi-K3 release on Hugging Face (HN 1,382 + 391 pts)",
        "20260728": "Kimi-K3 coverage continues; views still elevated",
    },
}


def contaminated_for(page):
    """Dates to optionally exclude for a page, or an empty dict."""
    return CONTAMINATED_DATES.get(page, {})


# Season matching for the placebo null (review response M3). The original null
# was drawn from 15 Jan - 31 May 2026, a period when the concept pages ran
# 6-47% busier than the pre-event June-July baseline. Busier pages produce
# larger absolute excess, inflating the null and making the observed value rank
# artificially LOW - the direction that flatters this project's own conclusion.
# Prior-year same-season windows remove that bias.
SEASON_MATCHED_SPANS = [
    ("20240615", "20240915"),
    ("20250615", "20250915"),
]
