#!/usr/bin/env python3
"""Resumable, crash-safe Wikipedia extraction.

Design:
  * Output is append-only. A JSON checkpoint stores how many pages have been
    fully scanned+decoded. On resume we re-decompress from the start (bz2 is
    single-stream, not random access) but we SKIP pages already consumed
    (n <= pages_done) so nothing is duplicated in the output. This makes a
    crash cost only re-decompression, never re-cleanup or duplicate writes.
  * Logs to stderr with flush=True so detached runs show progress.

Usage:
  python3 wiki_extract_resumable.py <input.bz2> -o <articles.txt> [--state FILE]
"""
import argparse, bz2, json, os, re, sys, time
from lxml import etree

KEYWORD_HINTS = [
    "mathematics", "algebra", "geometry", "topology", "analysis",
    "calculus", "number theory", "combinatorics", "probability",
    "statistics", "set theory", "group theory", "linear algebra",
    "differential", "tensor", "manifold", "category theory",
    "logic", "theorem", "lemma", "proof", "graph theory",
    "physics", "quantum", "relativity", "mechanics", "thermodynamics",
    "electromagnetism", "particle", "atomic", "nuclear", "optics",
    "cosmology", "astrophysics", "gravity", "field theory",
    "string theory", "statistical mechanics", "condensed matter",
    "superconduct", "plasma", "semiconductor", "wave", "energy",
    "force", "momentum", "entropy", "hamiltonian", "lagrangian",
    "algorithm", "data structure", "computer science", "programming",
    "compiler", "operating system", "database", "network",
    "software", "kernel", "linux", "unix", "windows", "macos",
    "python", "java", "rust", "c++", "javascript", "typescript",
    "machine learning", "neural network", "deep learning",
    "artificial intelligence", "computer architecture",
    "instruction set", "memory (computing)", "cpu", "gpu",
    "distributed system", "cloud computing", "cryptography",
    "hash function", "encryption", "decryption",
    "regular expression", "automata", "complexity theory",
    "turing machine", "lambda calculus", "type theory",
    "functional programming", "object-oriented",
    "microprocessor", "transistor", "logic gate", "circuit",
    "integrated circuit", "semiconductor device", "diode",
    "mosfet", "cmos", "flip-flop", "register", "alu",
    "bus (computing)", "motherboard", "ram", "rom",
    "hard disk", "ssd", "solid-state drive", "flash memory",
    "graphics card", "sound card", "network card",
    "computer hardware", "electronic engineering",
    "printed circuit", "fpga", "asic", "vlsi",
    "signal processing", "digital signal", "analog",
    "oscilloscope", "multimeter",
    # ── Economy / finance ──────────────────────────────────────────────
    "economy", "economics", "macroeconomic", "microeconomic",
    "gdp", "inflation", "monetary policy", "fiscal policy",
    "supply chain", "commodity", "commodity market", "trade",
    "market", "pricing", "settlement", "ledger", "treasury",
    "tokenomic", "blockchain", "smart contract", "cryptocurrency",
    # ── Electricity / energy generation ─────────────────────────────────
    "electricity", "electric power", "electrical grid", "power grid",
    "electric energy", "electricity generation", "power generation",
    "power plant", "power station", "powerplant", "utility",
    "baseload", "peaker", "load curve", "peak demand", "demand response",
    "net metering", "distributed generation", "microgrid",
    "virtual power plant", "vpp", "grid stability", "frequency regulation",
    "ancillary services", "substation", "transformer", "high-voltage",
    "high voltage", "hvdc", "grid operator", "smart grid",
    "energy market", "electricity market", "grid-tied", "grid tie",
    "power inverter", "inverter", "mppt", "charge controller",
    # ── Solar power plants ─────────────────────────────────────────────
    "solar power", "solar power plant", "solar plant", "solar farm",
    "photovoltaic", "photovoltaics", "pv module", "solar panel", "pv cell",
    "monocrystalline", "polycrystalline", "thin-film", "thin film",
    "perovskite", "bifacial", "het", "topcon", "perc cell",
    "solar tracker", "tracking system", "solar array", "solar string",
    "concentrated solar", "csp", "parabolic trough", "solar tower",
    "heliostat", "solar thermal", "molten salt", "thermal storage",
    "solar receiver", "solar energy", "insolation", "irradiance",
    # ── Water desalination / distillation ───────────────────────────────
    "desalination", "water desalination", "desalination plant",
    "reverse osmosis", "sea water reverse osmosis", "swro",
    "multi-effect distillation", "multi-stage flash", "msf",
    "membrane distillation", "thermal desalination", "electrodialysis",
    "water treatment", "water purification", "water distillation",
    "distillation", "condenser", "evaporator", "brine",
    "potable water", "water supply", "wastewater treatment",
    "cooling tower", "heat exchanger",
    # ── Foundry / smelting / materials ──────────────────────────────────
    "foundry", "metal foundry", "smelting", "smelter", "smelt",
    "casting", "die casting", "sand casting", "investment casting",
    "blast furnace", "electric arc furnace", "basic oxygen furnace",
    "aluminum", "aluminium", "alumina", "bauxite", "hall-heroult",
    "steel", "steelmaking", "steel mill", "pig iron", "wrought iron",
    "silicon", "metallurgical silicon", "silicon metal", "ferrosilicon",
    "silica", "quartz", "sand mining", "silica sand",
    "electrolysis", "electrolytic", "electrolyzer", "refining",
    "hydrometallurgy", "pyrometallurgy", "metallurgy", "metal alloy",
    "alloy", "copper", "nickel", "zinc", "lead", "tin", "titanium",
    "magnesium", "lithium", "rare earth", "manganese", "chromium",
    "cobalt", "tungsten", "molybdenum", "aluminum smelting",
    "potline", "anode", "cathode",
    # ── Mining / geology / mineral deposits ─────────────────────────────
    "mining", "mining industry", "mining engineering", "mine",
    "mineral", "mineral deposit", "ore", "ore deposit", "ore body",
    "ore grade", "geology", "geological", "geomorphology",
    "mineralogy", "petrology", "stratigraphy", "tectonic",
    "plate tectonics", "volcanology", "seismology", "rock",
    "sedimentary", "igneous", "metamorphic", "geochemistry",
    "exploration", "mineral exploration", "prospecting", "reserve",
    "open-pit", "open pit", "underground mining", "shaft mining",
    "strip mining", "placer", "dredging", "tailings", "ore dressing",
    "beneficiation", "flotation", "copper mining", "gold mining",
    "iron ore", "bauxite mining", "lithium mining", "coal mining",
    "uranium mining", "diamond mining", "salt mining", "quarry",
    "quarrying", "granite", "limestone", "marble", "sandstone",
    "continental crust", "subduction", "rift", "fault", "fold",
    "mountain building", "orogeny", "weathering", "erosion",
    "deposit", "ore genesis", "hydrothermal", "magmatic",
    "placer deposit", "sedimentary basin",
    # ── Electric motors / electromechanics ──────────────────────────────
    "electric motor", "electric motors", "induction motor", "motor",
    "electric machine", "electrical machine", "synchronous motor",
    "brushless", "permanent magnet", "stepper motor", "servomotor",
    "linear motor", "dc motor", "motor controller", "variable-frequency",
    "variable frequency drive", "vfd", "motor drive", "generator",
    "electric generator", "alternator", "dynamo", "turbine",
    "wind turbine", "hydraulic turbine", "steam turbine", "gas turbine",
    "hydroelectric", "hydropower", "pumped storage", "water turbine",
    "electromechanic", "actuator",
    # ── Rail / transportation infrastructure ────────────────────────────
    "rail", "railway", "railroad", "rail transport", "railway electrification",
    "overhead line", "catenary", "third rail", "electric locomotive",
    "locomotive", "freight rail", "high-speed rail", "commuter rail",
    "metro", "subway", "tram", "light rail", "rolling stock",
    "railway signalling", "track gauge", "ballast",
    # ── Space industry ──────────────────────────────────────────────────
    "space industry", "spaceflight", "space exploration", "rocket",
    "launch vehicle", "satellite", "orbital", "propulsion",
    "rocket engine", "spacecraft", "space station", "lunar",
    "mars mission", "orbital mechanics", "aerospace", "avionics",
    "satellite communication", "earth observation", "telemetry",
    # ── Engineering disciplines ─────────────────────────────────────────
    "engineering", "civil engineering", "structural engineering",
    "geotechnical engineering", "mechanical engineering",
    "electrical engineering", "electronic engineering",
    "industrial engineering", "process engineering", "chemical engineering",
    "materials engineering", "power engineering", "control engineering",
    "systems engineering", "computer engineering", "software engineering",
    "hardware engineering", "engineering design", "industrial design",
    "manufacturing", "factory", "industrial process", "automation",
    # ── Energy storage (grid infra from axiom-core) ─────────────────────
    "energy storage", "battery", "battery storage", "battery cell",
    "battery pack", "lfp", "li-ion", "lithium-ion", "sodium-ion",
    "lead-acid", "flow battery", "battery management", "bms",
    "state of charge", "state of health", "cycle life", "depth of discharge",
    "energy storage system", "ess", "containerized", "grid battery",
    "flywheel", "supercapacitor", "compressed air", "hydrogen storage",
    "fuel cell", "electrolyzer", "power-to-x",
    # ── Compute / data centers ──────────────────────────────────────────
    "data center", "datacenter", "data centre", "server farm",
    "hyperscale", "colocation", "compute", "high performance computing",
    "gpu cluster", "server", "rack", "blade server", "cooling system",
    "liquid cooling", "immersion cooling", "power distribution unit",
    "pdu", "uninterruptible", "ups", "generator backup", "tier iv",
    # ── Embedded / IoT (ESP32/PoGIE from axiom-core) ────────────────────
    "esp32", "embedded system", "microcontroller", "firmware",
    "internet of things", "iot", "sensor network", "telemetry",
    "modbus", "rs-485", "serial protocol", "energy ip", "edge computing",
]
NEGATIVE_HINTS = [
    "television series", "album by", "song by", "film starring",
    "married to", "born in", "singer", "actor", "actress",
    "footballer", "cricketer", "politician",
    "village in", "town in", "district of",
    # narrow false-positive-prone generic words to their engineering senses
    "music video", "film (", "band", "single (music)", "rock band",
    "rock album", "album", "song", "film director", "television show",
    "racing team", "motorcycle racing", "motorsport", "motor racing",
    "game", "video game", "comic", "manga", "novel", "book series",
    "company (disambiguation)", "people", "person born",
]

WIKI_PATTERNS = [
    (re.compile(r"\{\{[^{}]*\}\}"), ""),
    (re.compile(r"\[\[File:[^\]]*\]\]", re.I), ""),
    (re.compile(r"\[\[Image:[^\]]*\]\]", re.I), ""),
    (re.compile(r"\[\[Category:[^\]]*\]\]", re.I), ""),
    (re.compile(r"\[\[([^|\]]*?\|)?([^\]]+?)\]\]"), r"\2"),
    (re.compile(r"'''([^']+)'''"), r"\1"),
    (re.compile(r"''([^']+)''"), r"\1"),
    (re.compile(r"<ref[^>]*>.*?</ref>", re.S | re.I), ""),
    (re.compile(r"<[^>]+>"), ""),
    (re.compile(r"^=+\s*([^=]+?)\s*=+$", re.M), r"\n\n## \1\n\n"),
    (re.compile(r"^\*+", re.M), ""),
    (re.compile(r"^#+\s*", re.M), ""),
    (re.compile(r"^\{\|.*?\|\}", re.S | re.M), ""),
    (re.compile(r"^---+$", re.M), "\n"),
    (re.compile(r"&amp;"), "&"),
    (re.compile(r"&lt;"), "<"),
    (re.compile(r"&gt;"), ">"),
    (re.compile(r"&nbsp;"), " "),
    (re.compile(r"&quot;"), '"'),
    (re.compile(r"&#\d+;"), ""),
    (re.compile(r"\n{3,}"), "\n\n"),
]

def wikitext_to_text(s):
    for pat, repl in WIKI_PATTERNS:
        s = pat.sub(repl, s)
    return s.strip()

def title_matches(title):
    t = title.lower()
    if any(neg in t for neg in NEGATIVE_HINTS):
        return False
    return any(kw in t for kw in KEYWORD_HINTS)

def load_state(path):
    try:
        with open(path) as f:
            return json.load(f).get("pages_done", 0)
    except Exception:
        return 0

def save_state(path, n):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"pages_done": n}, f)
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--state", default=None, help="state json path")
    ap.add_argument("--progress-every", type=int, default=20000)
    args = ap.parse_args()

    if args.state:
        state_file = args.state
    else:
        state_file = os.path.join(os.path.dirname(args.output), ".extract_state.json")
    pages_done0 = load_state(state_file)

    stats = {"pages": 0, "matched": 0, "emitted": 0, "redirects": 0, "chars": 0}
    t0 = time.time()
    if pages_done0:
        print(f"[resume] starting; {pages_done0} pages already emitted", file=sys.stderr, flush=True)

    src = bz2.open(args.input, "rb")
    # Append: preserve any previously-written clean output.
    with src, open(args.output, "a", encoding="utf-8") as out:
        context = etree.iterparse(src, events=("end",), tag="{*}page")
        for _, page in context:
            stats["pages"] += 1
            n = stats["pages"]

            if n % args.progress_every == 0:
                elapsed = time.time() - t0
                print(f"[extract] pages={n} matched={stats['matched']} "
                      f"emitted={stats['emitted']} chars={stats['chars']/1e6:.1f}MB "
                      f"rate={n/elapsed:.0f}pg/s", file=sys.stderr, flush=True)

            # Resume: skip already-emitted pages entirely (no duplicate append).
            if n <= pages_done0:
                page.clear()
                continue

            title_el = page.find("{*}title")
            ns_el = page.find("{*}ns")
            if title_el is None or ns_el is None:
                page.clear(); continue
            title = title_el.text or ""
            if ns_el.text != "0":
                page.clear(); continue
            if page.find("{*}redirect") is not None:
                stats["redirects"] += 1
                page.clear(); continue
            if not title_matches(title):
                page.clear(); continue
            stats["matched"] += 1
            text_el = page.find(".//{*}text")
            wikitext = text_el.text if text_el is not None else ""
            if not wikitext:
                page.clear(); continue
            clean = wikitext_to_text(wikitext)
            if len(clean) < 400:
                page.clear(); continue
            stats["emitted"] += 1
            out.write(title + "\n---\n" + clean + "\n<<<END>>>\n")
            stats["chars"] += len(clean)
            # free memory
            page.clear()
            while page.getprevious() is not None:
                del page.getparent()[0]

            if n % 5000 == 0:
                save_state(state_file, n)

    save_state(state_file, stats["pages"])
    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed:.1f}s ({stats['pages']/elapsed:.0f} pages/s)", file=sys.stderr)
    print(f"  pages={stats['pages']} matched={stats['matched']} "
          f"emitted={stats['emitted']} chars={stats['chars']/1e6:.1f}MB")

if __name__ == "__main__":
    main()