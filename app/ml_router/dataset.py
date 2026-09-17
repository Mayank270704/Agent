"""Versioned, local, reproducible training data for the intent
classifier (Milestone 22, Phase 3).

Generation is entirely template x slot-filling, in Python, with a fixed
random seed — NOT an LLM call, NOT a network request, NOT a paid API.
Every example is deterministic given this module's source: running
`build_dataset()` twice produces byte-identical output, which is what
lets `data/intent_dataset_v1.jsonl` be regenerated and diffed rather than
trusted as an opaque blob.

--------------------------------------------------------------------------
Why templates instead of hand-typing 500 sentences
--------------------------------------------------------------------------
Hand-typing hundreds of examples one at a time produces enormous lexical
overlap by accident (the author's own habitual phrasing) and makes
train/test leakage hard to audit. Templates with explicit topic/date/
concept slots make the generation process itself the leakage control:
the SAME template instantiated with a topic held out of training (see
`build_dataset(..., holdout_topics=...)`) is a clean, auditable way to
keep the Milestone 20 challenge queries out of the training set even
though several training templates are structurally similar to them.

--------------------------------------------------------------------------
Hard negatives (Phase 2/3 requirement)
--------------------------------------------------------------------------
`_DIRECT_HARD_NEGATIVES` exists specifically so the classifier cannot
succeed by memorizing "current"/"latest"/"recent"/"time"/"date" as
WEB/TIME/DATE keywords — each one contains such a word in a sense that
does NOT require a tool. See app/ml_router/contract.py for the reasoning
behind each one.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from app.ml_router.contract import Intent

DATASET_VERSION = "v1"
DATA_DIR = Path(__file__).resolve().parent / "data"
DATASET_PATH = DATA_DIR / f"intent_dataset_{DATASET_VERSION}.jsonl"

# A fixed seed makes every draw below (sampling, shuffling, splitting)
# reproducible across runs and across machines.
_SEED = 20260922


@dataclass(frozen=True)
class IntentExample:
    text: str
    label: str  # Intent.value — a plain string so JSONL stays human-readable

    def __post_init__(self) -> None:
        if not self.text or not self.text.strip():
            raise ValueError("IntentExample.text cannot be blank.")
        if self.label not in {intent.value for intent in Intent}:
            raise ValueError(f"Unknown label {self.label!r}.")


# ===========================================================================
# WEB — templates x topics
# ===========================================================================

_WEB_TEMPLATES = [
    "What is the latest news about {topic}?",
    "What are the latest developments in {topic}?",
    "What are the recent developments in {topic}?",
    "Tell me about recent developments in {topic}.",
    "What is happening with {topic} right now?",
    "What is the current state of {topic}?",
    "Search the web for {topic}.",
    "Search for the latest {topic} news.",
    "Look up current information about {topic}.",
    "Find online what's new with {topic}.",
    "Find current information about {topic}.",
    "What's new with {topic} this week?",
    "Give me today's news about {topic}.",
    "What is currently happening in {topic}?",
    "Can you look up the latest {topic} updates?",
    "search the web for {topic}",
    "whats the latest on {topic}",
    "any recent news on {topic}?",
    "What are people saying about {topic} today?",
    "Can you find the most recent information on {topic}?",
]

_WEB_TOPICS = [
    "AI", "OpenAI", "SpaceX", "the stock market", "Bitcoin", "climate change",
    "the NBA", "quantum computing", "Tesla", "the iPhone", "Google DeepMind",
    "the World Cup", "inflation", "electric vehicles", "the housing market",
    "Apple", "Microsoft", "Nvidia", "renewable energy", "self-driving cars",
    "cybersecurity", "gene editing", "space exploration", "cryptocurrency regulation",
    "artificial general intelligence", "chip manufacturing", "vaccine research",
    "5G networks", "the semiconductor industry", "remote work trends",
]

_WEB_ENTITY_TEMPLATES = [
    "Who is the current CEO of {company}?",
    "What is the current price of {stock}?",
    "What is the latest version of {software}?",
    "Has {company} released anything new recently?",
    "What is {company} working on currently?",
]

_WEB_COMPANIES = ["Google", "Amazon", "Meta", "Netflix", "Intel", "Samsung", "IBM", "Adobe"]
_WEB_STOCKS = ["Nvidia", "Apple", "Tesla", "Amazon", "Microsoft", "Gold"]
_WEB_SOFTWARE = ["Node.js", "Java", "Chrome", "Windows", "Android", "macOS", "TensorFlow", "PyTorch", "Docker", "React"]

_WEB_COMPOUND_TEMPLATES = [
    "Explain {concept} and then tell me about recent developments in {concept}.",
    "What is {concept}, and what are the latest breakthroughs in it?",
    "Explain what {concept} is, then give me today's news about it.",
    "Can you describe {concept} and also find the latest research on it?",
]

_WEB_COMPOUND_CONCEPTS = [
    "RAG", "reinforcement learning", "quantum computing", "fusion energy",
    "large language models", "gene therapy", "autonomous vehicles",
]

# ===========================================================================
# DIRECT — concept explanations, math, casual conversation, hard negatives
# ===========================================================================

_DIRECT_CONCEPT_TEMPLATES = [
    "What is {concept}?",
    "Explain {concept}.",
    "How does {concept} work?",
    "Define {concept}.",
    "Can you explain {concept} to me?",
    "What does {concept} mean?",
    "Give me a simple explanation of {concept}.",
    "I'm confused about {concept}, can you clarify?",
]

_DIRECT_CONCEPTS = [
    "machine learning", "a transformer in deep learning", "gradient descent",
    "a neural network", "recursion", "object-oriented programming",
    "photosynthesis", "the Pythagorean theorem", "supply and demand",
    "DNA replication", "a black hole", "quantum entanglement", "a linked list",
    "TCP/IP", "compound interest", "natural selection", "the water cycle",
    "a binary search tree", "REST APIs", "the central limit theorem",
    "Newton's laws of motion", "a hash table", "blockchain technology",
    "reinforcement learning", "big-O notation", "inheritance in programming",
    "the greenhouse effect", "an algorithm", "a database index", "polymorphism",
    "the stock market as a concept", "how compilers work", "osmosis",
]

_DIRECT_MATH_TEMPLATES = [
    "What is {a} + {b}?",
    "What is {a} times {b}?",
    "What is {a} minus {b}?",
    "If I have {a} apples and give away {b}, how many remain?",
    "What is the square root of {a}?",
    "Convert {a} kilometers to miles.",
]

_DIRECT_MATH_PAIRS = [(25, 17), (12, 8), (9, 4), (100, 37), (7, 6), (64, 1), (45, 22), (13, 13), (50, 9), (81, 1)]

_DIRECT_CASUAL = [
    "Hello, how are you?",
    "Tell me a joke.",
    "What's your favorite color?",
    "Can you help me write a poem about the ocean?",
    "Write a short story about a robot.",
    "Give me some tips for staying productive.",
    "How do I improve my writing skills?",
    "What are some good books to read?",
    "Can you summarize the plot of Romeo and Juliet?",
    "What's a good recipe for pasta?",
    "Can you help me draft a polite email declining a meeting?",
    "What are some tips for public speaking?",
    "Write a haiku about autumn.",
    "How should I structure a cover letter?",
    "What's a fun fact about octopuses?",
    "Give me a workout routine for beginners.",
    "How do I make my code more readable?",
    "What's the difference between a list and a tuple in Python?",
    "Can you proofread this sentence for grammar?",
    "What are some good icebreaker questions?",
]

# Deliberate hard negatives: each contains a recency/time/date-looking word
# in a sense that does NOT require a tool. See contract.py.
_DIRECT_HARD_NEGATIVES = [
    "What is current in programming, like functional versus object-oriented style?",
    "What is a current in physics?",
    "Explain the concept of electrical current.",
    "What is the current ratio in accounting?",
    "How do ocean currents work?",
    "What is today's lesson about, generally speaking, in a typical algebra class?",
    "Explain what a leap year is.",
    "What is the history of the Gregorian calendar?",
    "How is UTC time calculated?",
    "What is Daylight Saving Time?",
    "What time complexity does quicksort have?",
    "What is Big O time complexity?",
    "How do I convert local time to UTC manually?",
    "What is the most recent common ancestor in genetics?",
    "What does 'current' mean in electrical engineering?",
    "What time signature is used in a waltz?",
    "Explain what a fiscal year is.",
    "What is the current tense in grammar?",
    "How does a current transformer work in electronics?",
    "What is a leap second?",
]

# ===========================================================================
# TIME
# ===========================================================================

_TIME_TEMPLATES = [
    "What time is it?",
    "What's the current time?",
    "Can you tell me the time?",
    "What is the time right now?",
    "Tell me the current time please.",
    "what time is it",
    "WHAT TIME IS IT",
    "whats the time",
    "do you know what time it is",
    "current time?",
    "what's the time right now",
    "give me the current time",
    "I need to know the time.",
    "what time is it now",
    "Do you have the time?",
    "What's the time?",
    "Could you tell me what time it is right now?",
    "time please",
    "What time do you have?",
    "Please tell me the current local time.",
    "what's the time",
    "Time check, please.",
    "Can you give me a time check?",
    "What's the exact time right now?",
    "May I know the current time?",
    "I'd like to know what time it is.",
    "What is the local time right now?",
    "Tell me what time it is.",
    "Could you check the time for me?",
    "what is the time now",
    "Time, please?",
    "hey what time is it",
    "quick, what time is it",
    "what's the current local time",
    "can u tell me the time",
    "yo what time is it rn",
    "Do you know the exact time at the moment?",
    "What's the clock reading right now?",
    "Just checking, what time is it?",
    "I need the current time, please.",
]

# ===========================================================================
# DATE
# ===========================================================================

_DATE_TEMPLATES = [
    "What is today's date?",
    "What's the date today?",
    "What day is it?",
    "What day is today?",
    "Tell me today's date.",
    "What is the current date?",
    "what's today's date",
    "whats the date",
    "What's today's date?",
    "Can you tell me today's date?",
    "What day of the week is it today?",
    "today's date?",
]

_DATE_SPECIFIC_TEMPLATES = [
    "What day was {date}?",
    "What weekday is {date}?",
    "Which day of the week was {date}?",
    "What day of the week is {date}?",
]

_DATE_SPECIFIC_DATES = [
    "25 December 2026", "1 January 2027", "4 July 2025", "31 October 2026",
    "14 February 2027", "15 August 2026", "11 September 2025", "1 May 2026",
    "23 June 2027", "9 November 2025",
]


def _dedupe_preserve_order(examples: list[IntentExample]) -> list[IntentExample]:
    seen: set[str] = set()
    out: list[IntentExample] = []
    for example in examples:
        key = example.text.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(example)
    return out


def build_dataset(*, holdout_topics: frozenset[str] = frozenset()) -> list[IntentExample]:
    """Deterministically build the full labeled dataset.

    `holdout_topics` (matched case-insensitively as substrings of the
    generated text) lets a caller exclude specific topics/phrasings from
    generation entirely — used by tests and by `train.py` to guarantee
    zero overlap with the Milestone 20 external challenge set, at the
    SOURCE (generation), which is a stronger guarantee than filtering
    after the fact.
    """
    rng = random.Random(_SEED)
    examples: list[IntentExample] = []

    def _keep(text: str) -> bool:
        lowered = text.lower()
        return not any(topic.lower() in lowered for topic in holdout_topics)

    # -- WEB ----------------------------------------------------------------
    web_texts: list[str] = []
    for template in _WEB_TEMPLATES:
        for topic in _WEB_TOPICS:
            web_texts.append(template.format(topic=topic))
    rng.shuffle(web_texts)
    web_texts = web_texts[:140]  # sample down from the full cross-product

    for template in _WEB_ENTITY_TEMPLATES:
        pool = _WEB_COMPANIES if "{company}" in template else (_WEB_STOCKS if "{stock}" in template else _WEB_SOFTWARE)
        slot = "company" if "{company}" in template else ("stock" if "{stock}" in template else "software")
        for value in pool:
            web_texts.append(template.format(**{slot: value}))

    for template in _WEB_COMPOUND_TEMPLATES:
        for concept in _WEB_COMPOUND_CONCEPTS:
            web_texts.append(template.format(concept=concept))

    examples += [IntentExample(text=t, label=Intent.WEB.value) for t in web_texts if _keep(t)]

    # -- DIRECT ---------------------------------------------------------------
    direct_texts: list[str] = []
    for template in _DIRECT_CONCEPT_TEMPLATES:
        for concept in _DIRECT_CONCEPTS:
            direct_texts.append(template.format(concept=concept))
    rng.shuffle(direct_texts)
    direct_texts = direct_texts[:140]

    for template in _DIRECT_MATH_TEMPLATES:
        for a, b in _DIRECT_MATH_PAIRS:
            direct_texts.append(template.format(a=a, b=b))

    direct_texts += _DIRECT_CASUAL
    direct_texts += _DIRECT_HARD_NEGATIVES

    examples += [IntentExample(text=t, label=Intent.DIRECT.value) for t in direct_texts if _keep(t)]

    # -- TIME -----------------------------------------------------------------
    examples += [IntentExample(text=t, label=Intent.TIME.value) for t in _TIME_TEMPLATES if _keep(t)]
    # Punctuation/case variants for volume + robustness, generated
    # deterministically from the base templates rather than hand-typed.
    for base in _TIME_TEMPLATES:
        variant = base.rstrip("?.").capitalize() + "?"
        if variant != base and _keep(variant):
            examples.append(IntentExample(text=variant, label=Intent.TIME.value))
        upper_variant = base.upper()
        if upper_variant != base and _keep(upper_variant):
            examples.append(IntentExample(text=upper_variant, label=Intent.TIME.value))

    # -- DATE -----------------------------------------------------------------
    examples += [IntentExample(text=t, label=Intent.DATE.value) for t in _DATE_TEMPLATES if _keep(t)]
    for base in _DATE_TEMPLATES:
        variant = base.rstrip("?.").capitalize() + "?"
        if variant != base and _keep(variant):
            examples.append(IntentExample(text=variant, label=Intent.DATE.value))
    for template in _DATE_SPECIFIC_TEMPLATES:
        for date in _DATE_SPECIFIC_DATES:
            text = template.format(date=date)
            if _keep(text):
                examples.append(IntentExample(text=text, label=Intent.DATE.value))

    examples = _dedupe_preserve_order(examples)
    rng.shuffle(examples)
    return examples


def save_dataset(examples: list[IntentExample], path: Path = DATASET_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps({"text": example.text, "label": example.label}, ensure_ascii=True) + "\n")


def load_dataset(path: Path = DATASET_PATH) -> list[IntentExample]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            examples.append(IntentExample(text=row["text"], label=row["label"]))
    return examples


def class_distribution(examples: list[IntentExample]) -> dict[str, int]:
    counts: dict[str, int] = {intent.value: 0 for intent in Intent}
    for example in examples:
        counts[example.label] += 1
    return counts


def split_dataset(
    examples: list[IntentExample],
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = _SEED,
) -> tuple[list[IntentExample], list[IntentExample], list[IntentExample]]:
    """Stratified split: each class is shuffled and split independently
    so train/val/test each keep roughly the same class proportions as
    the full dataset, then the three pools are concatenated and
    re-shuffled. Deterministic given `seed`.
    """
    if not (0 < train_frac < 1) or not (0 < val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be in (0, 1) and sum to < 1.")

    rng = random.Random(seed)
    by_label: dict[str, list[IntentExample]] = {intent.value: [] for intent in Intent}
    for example in examples:
        by_label[example.label].append(example)

    train: list[IntentExample] = []
    val: list[IntentExample] = []
    test: list[IntentExample] = []

    for label, group in by_label.items():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train += shuffled[:n_train]
        val += shuffled[n_train : n_train + n_val]
        test += shuffled[n_train + n_val :]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def assert_no_text_overlap(*groups: list[IntentExample]) -> None:
    """Raises AssertionError if any two groups share an exact
    (case/whitespace-normalized) text — the leakage check Phase 3/7
    requires. Used by train.py and by the test suite."""
    seen: dict[str, int] = {}
    for group_index, group in enumerate(groups):
        for example in group:
            key = example.text.strip().lower()
            if key in seen and seen[key] != group_index:
                raise AssertionError(f"Text leaked across splits: {example.text!r}")
            seen[key] = group_index
