"""Generate the synthetic product-manual corpus for the chunk_rescorer demo.

The corpus is three clusters of three same-brand models (pressure washers, garage door
openers, portable generators) plus three short manuals. Within a cluster the first model is
the head; its siblings reuse the head's safety section verbatim and adapt the head's other
sections, so the first several thousand tokens of the three manuals read alike. Each model's
Troubleshooting section carries exactly one distinguishing fault from assets/data/queries.json
and none of its siblings' faults. Text is written through the Gemini CLI (`gemini -p`).

    uv run python src/chunk_rescorer/generate_manuals.py --dry-run       # show the plan, no calls
    uv run python src/chunk_rescorer/generate_manuals.py                 # generate missing manuals
    uv run python src/chunk_rescorer/generate_manuals.py --only RG-4000P --force
    uv run python src/chunk_rescorer/generate_manuals.py --postprocess   # re-clean files, rebuild manifest

Outputs: assets/data/manuals/<MODEL>.md and assets/data/manuals/manifest.json.
Requires the Gemini CLI on PATH and already authenticated.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

TOK_PER_WORD = 1.326  # Qwen3 tokenizer on English prose; the notebook computes exact counts
CAP_TOKENS = 8192  # jina-reranker-v3.5 per-document cap on EIS

DATA_DIR = Path(__file__).resolve().parents[2] / "assets" / "data"
OUT_DIR = DATA_DIR / "manuals"
QUERIES = DATA_DIR / "queries.json"

BRANDS = {
    "Ridgeline Outdoor Power": (
        "Outdoor power equipment maker. Formal, safety-heavy register in the style of "
        "engine-driven equipment manuals: California Proposition 65 notice, carbon monoxide "
        "warnings, fuel handling, spark arrestor and emissions statements."
    ),
    "Kestrel Home Systems": (
        "Home comfort and access-control equipment. Installer-oriented register: code "
        "compliance references (NFPA, UL, local codes), qualified-installer language, "
        "entrapment warnings, diagnostic-code tables."
    ),
    "Northfield Tools": (
        "Hand and bench power tools and accessories. Terse workshop register: eye protection, "
        "dust, battery pack handling, compact numbered steps."
    ),
}


@dataclass
class Product:
    model: str
    brand: str
    category: str
    name: str
    target_tokens: int
    install_word: str = "Assembly"  # "Installation" for fixed equipment
    features: list[str] = field(default_factory=list)
    head: bool = False  # first model of a cluster; siblings adapt its sections

    @property
    def target_words(self) -> int:
        return round(self.target_tokens / TOK_PER_WORD)


RIDGE, KESTREL, NORTH = "Ridgeline Outdoor Power", "Kestrel Home Systems", "Northfield Tools"

CLUSTERS: list[list[Product]] = [
    [
        Product("RG-3200P", RIDGE, "pressure washer", "3,200 PSI 2.5 GPM Gas Pressure Washer", 16000, head=True,
                features=["208cc OHV engine", "axial cam pump with thermal relief valve", "five quick-connect nozzles",
                          "onboard detergent tank", "25-foot high-pressure hose", "never-flat wheels"]),
        Product("RG-2600P", RIDGE, "pressure washer", "2,600 PSI 2.3 GPM Gas Pressure Washer", 16000,
                features=["196cc OHV engine", "axial cam pump with thermal relief valve", "four quick-connect nozzles",
                          "detergent siphon tube", "25-foot high-pressure hose", "10-inch never-flat wheels",
                          "garden hose inlet filter screen"]),
        Product("RG-4000P", RIDGE, "pressure washer", "4,000 PSI 3.5 GPM Gas Pressure Washer", 16000,
                features=["306cc OHV engine with low-oil shutdown", "triplex plunger pump with oil sight glass",
                          "five quick-connect nozzles", "onboard detergent tank", "50-foot steel-braided high-pressure hose",
                          "13-inch pneumatic wheels"]),
    ],
    [
        Product("KG-1200B", KESTREL, "garage door opener", "1-1/4 HP Belt-Drive Garage Door Opener with Battery Backup",
                22000, install_word="Installation", head=True,
                features=["DC motor with soft start and stop", "steel-reinforced belt rail", "battery backup",
                          "Wi-Fi module and smartphone app", "safety reversing sensors", "wireless keypad",
                          "integrated LED lighting", "rolling-code remotes"]),
        Product("KG-800C", KESTREL, "garage door opener", "1/2 HP Chain-Drive Garage Door Opener", 22000,
                install_word="Installation",
                features=["1/2 HP AC motor", "chain-drive rail with chain tension bolt", "safety reversing sensors",
                          "two rolling-code remotes", "wall control with lock button", "LED bulb socket"]),
        Product("KG-1500W", KESTREL, "garage door opener",
                "1-1/4 HP Belt-Drive Garage Door Opener with Camera and Battery Backup", 22000, install_word="Installation",
                features=["DC motor with soft start and stop", "steel-reinforced belt rail", "battery backup",
                          "integrated camera with night vision and privacy shutter", "motion-activated LED lighting",
                          "Wi-Fi module and smartphone app", "safety reversing sensors", "wireless keypad"]),
    ],
    [
        Product("RG-7500E", RIDGE, "portable generator", "7,500-Watt Electric-Start Portable Generator", 20000, head=True,
                features=["420cc OHV engine with low-oil shutdown", "electric start with recoil backup",
                          "CO Sentinel carbon monoxide shutdown", "four 120V GFCI duplex outlets and one 120/240V twist-lock outlet",
                          "digital hour meter", "wheel kit and folding handle", "8-gallon fuel tank"]),
        Product("RG-4500E", RIDGE, "portable generator", "4,500-Watt Recoil-Start Portable Generator", 20000,
                features=["224cc OHV engine with low-oil shutdown", "recoil start",
                          "two 120V GFCI duplex outlets and one 120/240V twist-lock outlet", "fuel shutoff valve with detent",
                          "CO Sentinel carbon monoxide shutdown", "4-gallon fuel tank", "wheel kit"]),
        Product("RG-10000D", RIDGE, "portable generator", "10,000-Watt Dual-Fuel Electric-Start Portable Generator", 20000,
                features=["459cc OHV engine with low-oil shutdown", "dual fuel: gasoline or propane with fuel selector",
                          "electric start with remote start fob", "propane regulator with excess-flow safety",
                          "CO Sentinel carbon monoxide shutdown",
                          "four 120V GFCI duplex outlets, one 120/240V twist-lock outlet, one 50-amp outlet",
                          "digital multimeter display", "8.5-gallon fuel tank"]),
    ],
]

SHORT: list[Product] = [
    Product("NF-D20B", NORTH, "cordless drill/driver", "20V Brushless 1/2-inch Drill/Driver", 4500,
            features=["brushless motor", "two-speed gearbox", "24-position clutch", "1/2-inch ratcheting chuck",
                      "LED work light", "belt hook", "2.0 Ah battery and charger"]),
    Product("KT-7W", KESTREL, "smart thermostat", "7-Day Programmable Wi-Fi Thermostat", 5000, install_word="Installation",
            features=["touchscreen", "7-day scheduling", "Wi-Fi app control", "C-wire or battery power",
                      "conventional and heat pump systems up to 2H/2C", "filter change reminder"]),
    Product("NF-HR100", NORTH, "hose reel", "100-ft Wall-Mount Retractable Hose Reel", 2500,
            features=["automatic slow retraction", "180-degree swivel bracket", "brass inlet fitting",
                      "hose stopper", "5/8-inch hose included", "lockable at any length"]),
]

PRODUCTS: list[Product] = [p for c in CLUSTERS for p in c] + SHORT

# Section template: (title, share of the word budget). Troubleshooting lands around 75% of the
# manual, past the 8,192-token cap in every long manual.
SECTIONS: list[tuple[str, float]] = [
    ("Important Safety Instructions", 0.18),
    ("Specifications", 0.05),
    ("{install} and Setup", 0.15),
    ("Operation", 0.22),
    ("Maintenance and Storage", 0.18),
    ("Troubleshooting", 0.12),
    ("Parts, Warranty, and Service", 0.10),
]
MIN_SECTION_WORDS = 120

GUIDELINES = """You write owner's manuals for a hardware retailer's product catalog. Every manual you write
is for a fictional product from a fictional brand; do not reference real manufacturers, real
standards-body document numbers you are unsure of, or real phone numbers or URLs. Invent
consistent ones for the brand and reuse them.

Write in the register of a real printed owner's manual: formal, procedural, safety-heavy.
Use numbered steps for procedures, "WARNING:" and "CAUTION:" lines with a short hazard
statement, specification tables written as plain "Label: value" lines, and consistent part
names and control names throughout. Refer to earlier sections by title when appropriate.

Format rules:
- Output plain markdown. Section heading as given, subsections as "### " headings. No bold.
- No HTML, no images, no placeholders such as [insert]. No preamble and no closing remarks.
- Write only the section requested. Do not write other sections' content or headings.
- Hit the requested word count within about 10 percent.
- Respond with the section text only. Do not use any tools, do not read or write files,
  and do not describe what you are about to do.

Troubleshooting sections are written as a list of entries, each entry exactly:
Problem: <symptom as a customer would notice it>
Possible cause: <one or more causes, separated by semicolons>
Corrective action: <specific steps, referencing part and control names used earlier>
Cover 15 to 25 distinct problems for a long manual, 6 to 10 for a short one. Group them under
"### " subheadings by subsystem."""

SECTION_RE = re.compile(r"^## (\d+)\. (.+)$", re.M)
ENTRY_RE = re.compile(r"Problem: (.*)\nPossible cause: .*\nCorrective action: .*\n?")


# ----------------------------------------------------------------------------- helpers

def count_words(text: str) -> int:
    return len(text.split())


def section_plan(p: Product) -> list[tuple[str, int]]:
    return [(title.format(install=p.install_word), max(MIN_SECTION_WORDS, round(p.target_words * share)))
            for title, share in SECTIONS]


def split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """(header, [(title, body)]) where each body starts with its '## N. Title' line."""
    starts = [m.start() for m in SECTION_RE.finditer(text)]
    if not starts:
        return text, []
    bodies = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
    return text[: starts[0]], [(SECTION_RE.match(b).group(2).strip(), b) for b in bodies]


def brand_context(p: Product) -> str:
    return (f"Brand: {p.brand}\nBrand profile: {BRANDS[p.brand]}\nProduct: {p.name}\n"
            f"Model number: {p.model}\nCategory: {p.category}\nKey features:\n"
            + "\n".join(f"- {f}" for f in p.features))


def load_faults() -> tuple[dict[str, str], dict[str, list[str]]]:
    """entry per model, and the Problem lines of every other model in the same cluster."""
    q = json.loads(QUERIES.read_text())
    entry = {s["target"]: s["entry"] for s in q["symptom_queries"]}
    exclude: dict[str, list[str]] = {}
    for cl in q["clusters"].values():
        for m in cl["models"]:
            exclude[m] = [entry[o].split("\n")[0][len("Problem: "):] for o in cl["models"] if o != m and o in entry]
    return entry, exclude


def postprocess(text: str, entry: str | None = None) -> str:
    """Strip bold; drop troubleshooting entries that leaked into other sections; make sure the
    model's distinguishing entry is present in Troubleshooting; tidy blank lines."""
    text = text.replace("**", "")
    header, sections = split_sections(text)
    if not sections:
        return text
    ts_problems = {m.group(1) for t, b in sections if t == "Troubleshooting" for m in ENTRY_RE.finditer(b)}
    out = []
    for title, body in sections:
        if title == "Troubleshooting":
            if entry and entry.split("\n")[0][len("Problem: "):] not in ts_problems:
                body = body.rstrip() + "\n\n" + entry + "\n"
        elif ts_problems:
            body = ENTRY_RE.sub(lambda m: "" if m.group(1) in ts_problems else m.group(0), body)
            body = re.sub(r"^### [^\n]*\n(?:\s*\n)*(?=(?:#{2,3} |\Z))", "", body, flags=re.M)
        out.append(re.sub(r"\n{3,}", "\n\n", body).rstrip() + "\n")
    return header + "\n".join(out)


def describe(p: Product, text: str) -> dict:
    header, sections = split_sections(text)
    offset = count_words(header)
    secs, ts_start = [], None
    for title, body in sections:
        w = count_words(body)
        secs.append({"title": title, "words": w, "start_word": offset, "start_est_tokens": round(offset * TOK_PER_WORD)})
        if title == "Troubleshooting":
            ts_start = round(offset * TOK_PER_WORD)
        offset += w
    words = count_words(text)
    est = round(words * TOK_PER_WORD)
    return {"model": p.model, "brand": p.brand, "category": p.category, "name": p.name, "head": p.head,
            "file": f"{p.model}.md", "target_tokens": p.target_tokens, "words": words, "est_tokens": est,
            "over_cap": est > CAP_TOKENS, "troubleshooting_start_est_tokens": ts_start,
            "troubleshooting_past_cap": ts_start is not None and ts_start > CAP_TOKENS, "sections": secs}


def write_manifest(generator: str) -> dict:
    by_model = {p.model: p for p in PRODUCTS}
    manuals = [describe(by_model[f.stem], f.read_text()) for f in OUT_DIR.glob("*.md") if f.stem in by_model]
    order = {p.model: i for i, p in enumerate(PRODUCTS)}
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": generator,
        "tokens_per_word_estimate": TOK_PER_WORD,
        "cap_tokens": CAP_TOKENS,
        "clusters": [[p.model for p in c] for c in CLUSTERS],
        "manuals": sorted(manuals, key=lambda m: order[m["model"]]),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


# ----------------------------------------------------------------------------- generation

class Gemini:
    """Drives `gemini -p` headlessly. Context goes in on stdin; the task goes in via -p."""

    def __init__(self, model: str | None, attempts: int = 5, timeout: int = 900):
        if not shutil.which("gemini"):
            raise SystemExit("gemini CLI not found on PATH; install and authenticate it first")
        self.model, self.attempts, self.timeout = model, attempts, timeout
        self.calls = self.in_tokens = self.out_tokens = 0

    def ask(self, context: str, task: str) -> str:
        cmd = ["gemini", "-p", task, "-o", "json"] + (["-m", self.model] if self.model else [])
        env = {**os.environ, "GEMINI_CLI_TRUST_WORKSPACE": "true"}
        last = ""
        for attempt in range(1, self.attempts + 1):
            try:
                proc = subprocess.run(cmd, input=context, capture_output=True, text=True, timeout=self.timeout, env=env)
            except subprocess.TimeoutExpired:
                proc, last = None, "timeout"
            if proc is not None and proc.returncode == 0:
                try:
                    payload = json.loads(proc.stdout)
                except json.JSONDecodeError:
                    payload = None
                if payload and payload.get("response", "").strip():
                    self.calls += 1
                    for m in payload.get("stats", {}).get("models", {}).values():
                        self.in_tokens += m.get("tokens", {}).get("prompt", 0)
                        self.out_tokens += m.get("tokens", {}).get("candidates", 0)
                    return payload["response"].strip()
                last = f"empty or non-JSON response: {proc.stderr[-300:]!r}"
            elif proc is not None:
                last = f"exit {proc.returncode}: {proc.stderr[-400:]!r}"
            wait = min(120, 10 * 2 ** (attempt - 1))
            print(f"  gemini attempt {attempt}/{self.attempts} failed ({last}); retry in {wait}s", file=sys.stderr, flush=True)
            time.sleep(wait)
        raise RuntimeError(last)


def header_for(p: Product) -> str:
    return (f"# {p.brand}\n\n# {p.name}\n\nOwner's Manual · Model {p.model}\n\n"
            f"Read this manual completely before assembling, installing, or operating the product. "
            f"Keep it for future reference.\n")


def adapt_verbatim(section: str, head: Product, p: Product) -> str:
    """Head's section reused word for word, with the model identity swapped."""
    return section.replace(head.name, p.name).replace(head.model, p.model)


def write_section(g: Gemini, p: Product, n: int, title: str, words: int, previous: str,
                  reference: str | None, entry: str | None, exclude: list[str]) -> str:
    task = (f"Write section {n} of the owner's manual for the {p.name} (model {p.model}). "
            f'Heading: "## {n}. {title}". Target length: {words} words. '
            "Follow the writing guidelines and use the product sheet and the manual so far, all provided on stdin.")
    if reference:
        task += (" A reference version of this section from a sibling model of the same brand is provided. "
                 "Keep its structure, subheadings, and shared wording; change only what differs for this model "
                 "(specifications, controls, features, part numbers).")
    if title == "Troubleshooting":
        if entry:
            task += " Include the entry given under '=== Required troubleshooting entry ===' word for word, under a fitting subheading."
        if exclude:
            task += (" Do not include any entry about these symptoms; they belong to other models: "
                     + " | ".join(exclude))
    parts = ["=== Writing guidelines ===", GUIDELINES, "", "=== Product sheet ===", brand_context(p), ""]
    if reference:
        parts += ["=== Reference section from sibling model ===", reference, ""]
    if title == "Troubleshooting" and entry:
        parts += ["=== Required troubleshooting entry ===", entry, ""]
    if previous:
        parts += ["=== Manual so far ===", previous, ""]
    text = g.ask("\n".join(parts), task)
    return text if text.startswith("## ") else f"## {n}. {title}\n\n{text}"


def build_manual(g: Gemini, p: Product, head: Product | None, head_text: str | None,
                 entry: str | None, exclude: list[str]) -> str:
    head_secs = dict(split_sections(head_text)[1]) if head_text else {}
    parts = [header_for(p)]
    for n, (title, words) in enumerate(section_plan(p), start=1):
        ref = head_secs.get(title)
        if n == 1 and ref and head:
            text = adapt_verbatim(ref, head, p)  # shared safety boilerplate, no call
        else:
            text = write_section(g, p, n, title, words, "\n\n".join(parts[1:]), ref, entry, exclude)
        parts.append(text)
        print(f"  {p.model} §{n} {title}: {count_words(text)} words (asked {words}){' [copied]' if n == 1 and ref else ''}", flush=True)
    return postprocess("\n\n".join(parts) + "\n", entry)


def dry_run(products: list[Product]) -> None:
    print(f"{'model':10} {'category':19} {'head':5} {'target_tok':>10} {'words':>6}  troubleshooting ~starts")
    for p in products:
        plan = section_plan(p)
        ts = round(sum(w for _, w in plan[:5]) * TOK_PER_WORD)
        print(f"{p.model:10} {p.category:19} {str(p.head):5} {p.target_tokens:>10} {p.target_words:>6}  ~{ts} tok")
    print(f"\nlong-manual target total: {sum(p.target_tokens for p in products if p.target_tokens > CAP_TOKENS):,} tokens "
          f"(generated text has run ~1.5x target)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, help="Gemini model id passed to `gemini -m` (default: CLI default)")
    ap.add_argument("--only", action="append", help="Generate only this model number (repeatable)")
    ap.add_argument("--force", action="store_true", help="Regenerate manuals that already exist")
    ap.add_argument("--dry-run", action="store_true", help="Print the plan; make no calls")
    ap.add_argument("--postprocess", action="store_true", help="Re-clean manuals on disk and rebuild manifest.json; no calls")
    args = ap.parse_args()

    known = {p.model for p in PRODUCTS}
    if args.only and set(args.only) - known:
        print(f"unknown model(s): {set(args.only) - known}", file=sys.stderr)
        return 2
    selected = [p for p in PRODUCTS if not args.only or p.model in args.only]
    if args.dry_run:
        dry_run(selected)
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entry, exclude = load_faults()

    if args.postprocess:
        for p in PRODUCTS:
            f = OUT_DIR / f"{p.model}.md"
            if f.exists():
                f.write_text(postprocess(f.read_text(), entry.get(p.model)))
        m = write_manifest("gemini-cli")
        print(f"post-processed; manifest rebuilt with {len(m['manuals'])} entries")
        return 0

    def wanted(p: Product) -> bool:
        return p in selected and (args.force or not (OUT_DIR / f"{p.model}.md").exists())

    g = Gemini(args.model)
    t0 = time.time()
    for cluster in CLUSTERS:
        head = cluster[0]
        head_path = OUT_DIR / f"{head.model}.md"
        if wanted(head) or not head_path.exists():
            print(f"generating {head.model} ({head.name}, head, target {head.target_tokens} tokens)", flush=True)
            head_path.write_text(build_manual(g, head, None, None, entry.get(head.model), exclude.get(head.model, [])))
        head_text = head_path.read_text()
        for p in cluster[1:]:
            if not wanted(p):
                continue
            print(f"generating {p.model} ({p.name}, sibling of {head.model}, target {p.target_tokens} tokens)", flush=True)
            (OUT_DIR / f"{p.model}.md").write_text(build_manual(g, p, head, head_text, entry.get(p.model), exclude.get(p.model, [])))
    for p in SHORT:
        if wanted(p):
            print(f"generating {p.model} ({p.name}, target {p.target_tokens} tokens)", flush=True)
            (OUT_DIR / f"{p.model}.md").write_text(build_manual(g, p, None, None, None, []))

    m = write_manifest("gemini-cli" + (f" ({args.model})" if args.model else ""))
    print(f"\n{'model':10} {'est_tokens':>10} {'ts_start':>9}  over_cap  ts_past_cap")
    for x in m["manuals"]:
        print(f"{x['model']:10} {x['est_tokens']:>10} {str(x['troubleshooting_start_est_tokens']):>9}  "
              f"{str(x['over_cap']):8}  {x['troubleshooting_past_cap']}")
    print(f"gemini calls: {g.calls}; tokens {g.in_tokens:,} in / {g.out_tokens:,} out; {round(time.time() - t0)}s wall")
    return 0


if __name__ == "__main__":
    sys.exit(main())
