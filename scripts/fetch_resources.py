"""Download the free resources the project needs (all open licenses / public domain).

- Google Fonts (OFL / Apache / UFL licensed) -> data/fonts/<family>/
- Project Gutenberg texts (public domain, via the NLTK data mirror) -> data/corpus/gutenberg/
- tessdata_best English model (Apache 2.0, float LSTM, fine-tunable) -> data/tessdata/

Usage:
    python scripts/fetch_resources.py
"""

import io
import re
import sys
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data"
GFONTS = "https://raw.githubusercontent.com/google/fonts/main"

FAMILIES = """
lora merriweather playfairdisplay ptserif sourceserif4 crimsontext librebaskerville ebgaramond notoserif
cormorantgaramond spectral cardo oldstandardtt domine bitter robotoslab zillaslab arvo alegreya vollkorn gelasio
frankruhllibre literata sourceserifpro ibmplexserif notoserifdisplay
opensans lato montserrat raleway nunito ptsans sourcesans3 worksans firasans notosans ubuntu poppins mulish karla
rubik manrope barlow dmsans ibmplexsans publicsans cabin oxygen josefinsans quicksand archivo assistant heebo hind
asap exo2 titilliumweb oswald roboto robotocondensed barlowcondensed sairacondensed inter librefranklin
sourcecodepro ibmplexmono spacemono inconsolata firamono robotomono courierprime jetbrainsmono ubuntumono cousine
anonymouspro overpassmono specialelite tinos arimo
caveat patrickhand kalam indieflower architectsdaughter
""".split()


def get(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    except Exception:
        return None


def fetch_family(family: str) -> int:
    out = ROOT / "fonts" / family
    if out.exists() and any(out.glob("*.ttf")):
        return len(list(out.glob("*.ttf")))
    for lic in ("ofl", "apache", "ufl"):
        meta = get(f"{GFONTS}/{lic}/{family}/METADATA.pb")
        if not meta:
            continue
        files = re.findall(r'filename: "([^"]+\.ttf)"', meta.decode())
        out.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in files:
            data = get(f"{GFONTS}/{lic}/{family}/{urllib.request.quote(f)}")
            if data:
                (out / f).write_bytes(data)
                n += 1
        return n
    return 0


def fetch_fonts() -> None:
    with ThreadPoolExecutor(8) as ex:
        counts = dict(zip(FAMILIES, ex.map(fetch_family, FAMILIES)))
    missing = [f for f, n in counts.items() if n == 0]
    print(f"fonts: {sum(counts.values())} files from {len(FAMILIES) - len(missing)} families; missing: {missing}")


def fetch_gutenberg() -> None:
    out = ROOT / "corpus"
    if (out / "gutenberg").exists():
        return
    data = get("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip")
    if not data:
        sys.exit("could not download gutenberg corpus")
    out.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(data)).extractall(out)
    print("corpus: gutenberg ok")


def fetch_tessdata() -> None:
    out = ROOT / "tessdata" / "eng_best.traineddata"
    if out.exists():
        return
    data = get("https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/eng.traineddata")
    if not data:
        sys.exit("could not download tessdata_best")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    print("tessdata: eng_best ok")


if __name__ == "__main__":
    fetch_tessdata()
    fetch_gutenberg()
    fetch_fonts()
