"""Synthetic temporal and provenance corpus, labeled by construction (spec section 20.3).

    corpus = generate()                 # seed 20261026, 40 families
    render_document(family, version)    # the text of one version

A *family* is one document with two or three versions. It has 6 to 10 typed facts (numbers,
durations, dates, names, booleans) drawn from a per-domain template table. Version 2 changes
2 to 4 facts, removes one and adds one; a quarter of the families get a version 3 that changes
1 to 2 more. Every claim is a fact sentence with a value, and its gold verdict at each version
follows from the stored values alone:

- ``supported`` when the fact exists at that version with the claim's value,
- ``contradicted`` when the fact exists with another value,
- ``insufficient`` when the fact is absent at that version.

Nothing here reads a model or the network: the only randomness is ``random.Random(seed)``, and
the fixed tables below are the whole vocabulary. Each family also has a *distractor* (another
entity, the same templates, other values, same collection) and a tenth of the corpus (up to 10
families) has a *foreign* copy for a second tenant that conflicts with every value.
"""

import hashlib
import random
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from itertools import product
from typing import Literal

SEED = 20261026
DOMAINS = ("product_policy", "hr_policy", "api_docs", "service_terms")
DOMAIN_LABELS = {
    "product_policy": "product policy",
    "hr_policy": "HR policy",
    "api_docs": "documentation",
    "service_terms": "service terms",
}

FactKind = Literal["number", "duration", "date", "name", "boolean"]
ClaimKind = Literal["true_v1", "true_latest", "never_true", "absent"]

SUPPORTED = "supported"
CONTRADICTED = "contradicted"
INSUFFICIENT = "insufficient"

BASE_DATE = datetime(2026, 1, 1, tzinfo=UTC)  # family i starts at BASE_DATE + i days
V2_OFFSET = timedelta(days=90)
V3_OFFSET = timedelta(days=180)
MAX_FOREIGN = 10  # families copied into the second tenant
MAX_FAMILIES = 128  # the entity tables hold 64 names per domain (a family and its distractor)
MIN_FACTS, MAX_FACTS = 6, 10


@dataclass(frozen=True)
class FactSpec:
    key: str
    kind: FactKind
    template: str  # one sentence with one "{entity}" and one "{value}"


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    domain: str
    entity: str
    # Per version, fact key -> rendered value; None when the fact is absent from that version.
    # Every version names every key the family ever uses.
    versions: tuple[dict[str, str | None], ...]
    effective: tuple[datetime, ...]


@dataclass(frozen=True)
class TemporalClaim:
    claim_id: str
    family_id: str
    fact_key: str
    text: str
    kind: ClaimKind
    gold: tuple[str, ...]  # the gold verdict at each version index


@dataclass(frozen=True)
class SyntheticCorpus:
    families: tuple[FamilySpec, ...]
    distractors: tuple[FamilySpec, ...]  # one per family, one version
    foreign: tuple[FamilySpec, ...]  # conflicting copies of the first families
    claims: tuple[TemporalClaim, ...]  # of the families only


DOMAIN_FACTS: dict[str, tuple[FactSpec, ...]] = {
    "product_policy": (
        FactSpec(
            "return_window", "duration", "The {entity} can be returned within {value} of delivery."
        ),
        FactSpec("warranty_period", "duration", "The warranty on the {entity} lasts {value}."),
        FactSpec(
            "restocking_fee",
            "number",
            "Returning the {entity} carries a restocking fee of {value} percent.",
        ),
        FactSpec(
            "max_units_per_order",
            "number",
            "A single order may include at most {value} units of the {entity}.",
        ),
        FactSpec(
            "support_hours",
            "number",
            "Phone support for the {entity} is staffed {value} hours per week.",
        ),
        FactSpec(
            "replacement_lead_time",
            "duration",
            "A replacement for the {entity} ships within {value}.",
        ),
        FactSpec(
            "recall_notice_date",
            "date",
            "The most recent recall notice for the {entity} is dated {value}.",
        ),
        FactSpec("product_owner", "name", "The product owner of the {entity} is {value}."),
        FactSpec("gift_returns", "boolean", "Gift returns of the {entity} are {value}."),
        FactSpec("price_matching", "boolean", "Price matching for the {entity} is {value}."),
        FactSpec(
            "spare_parts_years",
            "number",
            "Spare parts for the {entity} are stocked for {value} years.",
        ),
        FactSpec(
            "battery_cycles",
            "number",
            "The battery of the {entity} is rated for {value} charge cycles.",
        ),
    ),
    "hr_policy": (
        FactSpec(
            "notice_period",
            "duration",
            "Employees in the {entity} must give notice of {value} before resigning.",
        ),
        FactSpec(
            "probation_period",
            "duration",
            "New hires in the {entity} serve a probation period of {value}.",
        ),
        FactSpec(
            "annual_leave_days",
            "number",
            "Staff in the {entity} receive {value} days of annual leave.",
        ),
        FactSpec("sick_days", "number", "The {entity} grants {value} paid sick days each year."),
        FactSpec(
            "remote_days",
            "number",
            "Members of the {entity} may work remotely {value} days per month.",
        ),
        FactSpec(
            "overtime_rate",
            "number",
            "Overtime in the {entity} is paid at {value} percent of the base rate.",
        ),
        FactSpec(
            "review_date",
            "date",
            "The next performance review cycle of the {entity} begins on {value}.",
        ),
        FactSpec("hr_contact", "name", "The human resources contact for the {entity} is {value}."),
        FactSpec(
            "relocation_support", "boolean", "Relocation support for the {entity} is {value}."
        ),
        FactSpec(
            "parental_leave_weeks", "number", "Parental leave in the {entity} lasts {value} weeks."
        ),
        FactSpec(
            "training_budget",
            "number",
            "Each member of the {entity} has a training budget of {value} dollars.",
        ),
        FactSpec("casual_dress", "boolean", "Casual dress in the {entity} is {value}."),
    ),
    "api_docs": (
        FactSpec("rate_limit", "number", "The {entity} allows {value} requests per minute."),
        FactSpec(
            "max_payload", "number", "The {entity} accepts payloads of up to {value} kilobytes."
        ),
        FactSpec(
            "token_lifetime", "duration", "Access tokens for the {entity} expire after {value}."
        ),
        FactSpec("request_timeout", "duration", "Requests to the {entity} time out after {value}."),
        FactSpec(
            "deprecation_date",
            "date",
            "The current version of the {entity} will be deprecated on {value}.",
        ),
        FactSpec("maintainer", "name", "The maintainer of the {entity} is {value}."),
        FactSpec("page_size", "number", "The {entity} returns {value} items per page by default."),
        FactSpec(
            "batch_limit",
            "number",
            "A single batch call to the {entity} may contain {value} operations.",
        ),
        FactSpec("cors", "boolean", "Cross origin requests to the {entity} are {value}."),
        FactSpec("idempotency_keys", "boolean", "Idempotency keys for the {entity} are {value}."),
        FactSpec("log_retention", "duration", "The {entity} keeps request logs for {value}."),
        FactSpec(
            "retry_attempts",
            "number",
            "Failed calls to the {entity} are retried up to {value} times.",
        ),
    ),
    "service_terms": (
        FactSpec(
            "planned_downtime",
            "number",
            "The {entity} permits {value} minutes of planned downtime each month.",
        ),
        FactSpec(
            "service_credit",
            "number",
            "Service credits for the {entity} equal {value} percent of the monthly fee.",
        ),
        FactSpec(
            "response_time",
            "duration",
            "Critical incidents on the {entity} receive a first response within {value}.",
        ),
        FactSpec(
            "termination_notice",
            "duration",
            "Either party may end the agreement for the {entity} with {value} of notice.",
        ),
        FactSpec(
            "payment_terms",
            "number",
            "Invoices for the {entity} are due within {value} days of issue.",
        ),
        FactSpec("renewal_date", "date", "The current term of the {entity} renews on {value}."),
        FactSpec("account_manager", "name", "The account manager for the {entity} is {value}."),
        FactSpec(
            "data_residency",
            "boolean",
            "Storing data of the {entity} outside the home region is {value}.",
        ),
        FactSpec("sublicensing", "boolean", "Sublicensing of the {entity} is {value}."),
        FactSpec("backup_retention", "duration", "Backups of the {entity} are kept for {value}."),
        FactSpec(
            "seat_limit",
            "number",
            "A standard contract for the {entity} covers up to {value} users.",
        ),
        FactSpec(
            "liability_cap",
            "number",
            "The liability cap for the {entity} is {value} thousand dollars.",
        ),
    ),
}

# Entity names are the product of two word lists: 64 per domain.
# fmt: off
_ENTITY_WORDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "product_policy": (
        ("Aurora", "Borealis", "Cobalt", "Dynamo", "Ember", "Falcon", "Glacier", "Harbor"),
        ("Router", "Speaker", "Monitor", "Camera", "Drone", "Keyboard", "Tablet", "Printer"),
    ),
    "hr_policy": (
        ("Remote", "Field", "Campus", "Regional", "Global", "Shift", "Contract", "Graduate"),
        ("Engineering Group", "Sales Group", "Support Group", "Finance Group",
         "Design Group", "Operations Group", "Research Group", "Legal Group"),
    ),
    "api_docs": (
        ("Orders", "Billing", "Catalog", "Inventory", "Shipping", "Search", "Payments", "Identity"),
        ("Core API", "Admin API", "Partner API", "Events API",
         "Reports API", "Webhooks API", "Sync API", "Batch API"),
    ),
    "service_terms": (
        ("Cloud Storage", "Managed Database", "Edge Network", "Message Queue",
         "Analytics", "Identity", "Backup", "Monitoring"),
        ("Standard Plan", "Premium Plan", "Enterprise Plan", "Starter Plan",
         "Business Plan", "Team Plan", "Academic Plan", "Nonprofit Plan"),
    ),
}

_NAMES = (
    "Priya Nair", "Tomas Weber", "Amelia Okafor", "Mateo Silva", "Hana Kobayashi",
    "Liam Doyle", "Sofia Rossi", "Noah Fischer", "Yara Haddad", "Ethan Brooks",
    "Zainab Khan", "Lucas Moreau", "Isla Campbell", "Omar Aziz", "Elena Petrova",
    "Jonas Berg", "Mei Tanaka", "Diego Alvarez", "Fatima Noor", "Henrik Larsen",
    "Aisha Bello", "Victor Lang", "Nadia Costa", "Samuel Ortiz",
)
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_UNITS = ("days", "weeks", "months", "hours")
_BOOLEAN = ("permitted", "not permitted")

# Neutral sentences without figures: two follow every fact sentence in a rendered document.
FILLERS = (
    "This section is reviewed regularly by the policy team.",
    "Exceptions require written approval from a manager.",
    "Changes are announced through the usual notice channels.",
    "A quick reference guide summarizes these points.",
    "Questions can be sent to the help desk at any time.",
    "Staff are trained on this topic every year.",
    "Records of requests are kept for audit purposes.",
    "The wording here follows the standard template.",
    "Local rules may add further requirements.",
    "Earlier versions of this page remain available on request.",
    "Please read the related notes before acting.",
    "This text is provided for general guidance.",
)
# fmt: on


def document_title(family: FamilySpec) -> str:
    return f"{family.entity} {DOMAIN_LABELS[family.domain]}"


def _fact(domain: str, key: str) -> FactSpec:
    return next(fact for fact in DOMAIN_FACTS[domain] if fact.key == key)


def fact_sentence(fact: FactSpec, entity: str, value: str) -> str:
    return fact.template.format(entity=entity, value=value)


def claim_value(claim: TemporalClaim, family: FamilySpec) -> str:
    """The value a claim states, read back out of its sentence."""
    prefix, _, suffix = _fact(family.domain, claim.fact_key).template.partition("{value}")
    prefix, suffix = prefix.format(entity=family.entity), suffix.format(entity=family.entity)
    if not (claim.text.startswith(prefix) and claim.text.endswith(suffix)):
        raise ValueError(f"claim {claim.claim_id} does not follow its template")
    return claim.text[len(prefix) : len(claim.text) - len(suffix)]


def _filler(family_id: str, key: str, slot: int) -> str:
    """One of the two fillers of a fact: fixed by the ids, different for the two slots."""
    digest = int(hashlib.sha256(f"{family_id}|{key}".encode()).hexdigest(), 16)
    first = digest % len(FILLERS)
    second = (first + 1 + (digest // len(FILLERS)) % (len(FILLERS) - 1)) % len(FILLERS)
    return FILLERS[(first, second)[slot]]


def render_document(family: FamilySpec, version_index: int) -> str:
    """Title line, then per present fact (in key order) its sentence and two filler sentences."""
    version = family.versions[version_index]
    blocks = [document_title(family)]
    for key in sorted(version):
        value = version[key]
        if value is None:
            continue
        sentence = fact_sentence(_fact(family.domain, key), family.entity, value)
        blocks.append(
            " ".join(
                (sentence, _filler(family.family_id, key, 0), _filler(family.family_id, key, 1))
            )
        )
    return "\n\n".join(blocks) + "\n"


# --- values ------------------------------------------------------------------------------------


def _draw(kind: FactKind, rng: random.Random) -> str:
    if kind == "number":
        return str(rng.randint(2, 120))
    if kind == "duration":
        return f"{rng.randint(2, 60)} {rng.choice(_UNITS)}"
    if kind == "date":
        return f"{rng.choice(_MONTHS)} {rng.randint(1, 28)}, {rng.choice((2026, 2027))}"
    if kind == "name":
        return rng.choice(_NAMES)
    return rng.choice(_BOOLEAN)


def _draw_other(kind: FactKind, rng: random.Random, avoid: Collection[str | None]) -> str | None:
    """A value not in ``avoid``; None when a boolean has no value left."""
    if kind == "boolean":
        options = [value for value in _BOOLEAN if value not in avoid]
        return rng.choice(options) if options else None
    while True:
        value = _draw(kind, rng)
        if value not in avoid:
            return value


def _gold(versions: tuple[Mapping[str, str | None], ...], key: str, value: str) -> tuple[str, ...]:
    """The gold verdict of a claim of ``value`` for fact ``key`` at each version."""
    verdicts = []
    for version in versions:
        stored = version.get(key)
        if stored is None:
            verdicts.append(INSUFFICIENT)
        elif stored == value:
            verdicts.append(SUPPORTED)
        else:
            verdicts.append(CONTRADICTED)
    return tuple(verdicts)


# --- families ----------------------------------------------------------------------------------


def _make_versions(
    rng: random.Random, domain: str, with_v3: bool
) -> tuple[dict[str, str | None], ...]:
    facts = DOMAIN_FACTS[domain]
    kinds = {fact.key: fact.kind for fact in facts}
    first = rng.sample(facts, rng.randint(MIN_FACTS, MAX_FACTS))
    v1: dict[str, str | None] = {fact.key: _draw(fact.kind, rng) for fact in first}
    added = rng.choice([fact for fact in facts if fact.key not in v1])

    present = sorted(v1)
    changed = rng.sample(present, rng.randint(2, 4))
    dropped = rng.choice([key for key in present if key not in changed])
    v2: dict[str, str | None] = dict(v1)
    for key in changed:
        v2[key] = _draw_other(kinds[key], rng, {v1[key]})
    v2[dropped] = None
    v2[added.key] = _draw(added.kind, rng)
    v1[added.key] = None
    versions = [v1, v2]

    if with_v3:
        v3 = dict(v2)
        alive = sorted(key for key, value in v2.items() if value is not None)
        for key in rng.sample(alive, rng.randint(1, 2)):
            v3[key] = _draw_other(kinds[key], rng, {v2[key]})
        versions.append(v3)
    return tuple(dict(sorted(version.items())) for version in versions)


def _effective(index: int, n_versions: int) -> tuple[datetime, ...]:
    start = BASE_DATE + timedelta(days=index)
    return (start, start + V2_OFFSET, start + V3_OFFSET)[:n_versions]


def _distractor(rng: random.Random, family: FamilySpec, entity: str) -> FamilySpec:
    """Same templates and keys as the family's first version, other entity, other values."""
    first = family.versions[0]
    values = {
        key: _draw_other(_fact(family.domain, key).kind, rng, {value})
        for key, value in first.items()
        if value is not None
    }
    return FamilySpec(
        family_id=f"{family.family_id}-distractor",
        domain=family.domain,
        entity=entity,
        versions=(values,),
        effective=family.effective[:1],
    )


def _foreign(rng: random.Random, family: FamilySpec) -> FamilySpec:
    """A copy whose every present value conflicts with the original's value at that version.

    A value that stays the same across versions stays the same in the copy, so the copy changes
    where the original does.
    """
    mapping: dict[tuple[str, str], str] = {}

    def conflicting(key: str, value: str) -> str:
        if (key, value) not in mapping:
            used = {other for (k, _), other in mapping.items() if k == key}
            mapped = _draw_other(_fact(family.domain, key).kind, rng, {value} | used)
            if mapped is None:  # a boolean whose both values are already taken
                raise RuntimeError("no conflicting value left")
            mapping[(key, value)] = mapped
        return mapping[(key, value)]

    versions = tuple(
        {key: None if value is None else conflicting(key, value) for key, value in v.items()}
        for v in family.versions
    )
    return FamilySpec(family.family_id, family.domain, family.entity, versions, family.effective)


def _claim(family: FamilySpec, fact: FactSpec, kind: ClaimKind, value: str) -> TemporalClaim:
    return TemporalClaim(
        claim_id=f"{family.family_id}:{fact.key}:{kind}",
        family_id=family.family_id,
        fact_key=fact.key,
        text=fact_sentence(fact, family.entity, value),
        kind=kind,
        gold=_gold(family.versions, fact.key, value),
    )


def _claims(rng: random.Random, family: FamilySpec) -> list[TemporalClaim]:
    """Per fact: true at v1, true at the latest version (when that differs), never true.

    Plus one claim about a fact of the domain that the family never mentions.
    """
    claims: list[TemporalClaim] = []
    first, last = family.versions[0], family.versions[-1]
    for key in sorted(first):
        fact = _fact(family.domain, key)
        if first[key] is not None:
            claims.append(_claim(family, fact, "true_v1", first[key]))
        if last[key] is not None and last[key] != first[key]:
            claims.append(_claim(family, fact, "true_latest", last[key]))
        ever = {version[key] for version in family.versions if version[key] is not None}
        never = _draw_other(fact.kind, rng, ever)
        if never is not None:
            claims.append(_claim(family, fact, "never_true", never))
    unused = [fact for fact in DOMAIN_FACTS[family.domain] if fact.key not in first]
    absent = rng.choice(unused)
    claims.append(_claim(family, absent, "absent", _draw(absent.kind, rng)))
    return claims


def generate(seed: int = SEED, n_families: int = 40) -> SyntheticCorpus:
    """The corpus for ``seed``: ``n_families`` families spread over the four domains in turn."""
    if not 1 <= n_families <= MAX_FAMILIES:
        raise ValueError(f"n_families must be between 1 and {MAX_FAMILIES}")
    rng = random.Random(seed)
    entities: dict[str, list[str]] = {}
    for position, domain in enumerate(DOMAINS):
        needed = 2 * len(range(position, n_families, len(DOMAINS)))
        names = [f"{a} {b}" for a, b in product(*_ENTITY_WORDS[domain])]
        entities[domain] = rng.sample(names, needed)
    with_v3 = set(rng.sample(range(n_families), n_families // 4))

    families: list[FamilySpec] = []
    distractors: list[FamilySpec] = []
    claims: list[TemporalClaim] = []
    for index in range(n_families):
        domain = DOMAINS[index % len(DOMAINS)]
        slot = index // len(DOMAINS)
        versions = _make_versions(rng, domain, index in with_v3)
        family = FamilySpec(
            family_id=f"{domain}-{slot:02d}",
            domain=domain,
            entity=entities[domain][2 * slot],
            versions=versions,
            effective=_effective(index, len(versions)),
        )
        families.append(family)
        distractors.append(_distractor(rng, family, entities[domain][2 * slot + 1]))
        claims.extend(_claims(rng, family))
    foreign = tuple(_foreign(rng, family) for family in families[:MAX_FOREIGN])
    return SyntheticCorpus(tuple(families), tuple(distractors), foreign, tuple(claims))
