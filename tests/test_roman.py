"""The standalone transliteration helpers, still used for saved metadata and font compatibility."""
import os, sys, tempfile
os.environ["PEAKCUT_DATA"] = tempfile.mkdtemp(prefix="pc-roman-"); os.environ["DATABASE_URL"] = ""
sys.path.insert(0, ".")
from studio import brain, roman
fails = []
def check(n, ok, x=""):
    print(("PASS " if ok else "FAIL ") + n + (f"  {x}" if x and not ok else "")); ok or fails.append(n)
def ws(text):
    return [{"w": w, "s": i * 0.5, "e": i * 0.5 + 0.4} for i, w in enumerate(text.split())]
def text(words):
    return " ".join(w["w"] for w in words)

# ---- the spelling rules on their own
for hi, want in [("चूना जो उसको", "chuna jo usko"), ("कहना करता समझना", "kahna karta samajhna"), ("आज", "aaj"),
                 ("ज़िंदगी", "zindagi"), ("नहीं यार", "nahi yaar"), ("बात", "baat"), ("हमारा", "hamara"),
                 ("GTA गेम में 1000", "GTA gem mein 1000")]:
    check(f"{hi} -> {want}", roman.to_roman(hi) == want, roman.to_roman(hi))
check("English is left alone", roman.to_roman("Hello there, 25 cars!") == "Hello there, 25 cars!")

# ---- what the AI writes for the screen
c = brain._tidy({"hook": "चूना लगा", "title": "उसको चूना", "highlight_words": ["चूना", "boss"], "score": 90}, 1, 20)
check("hook, title and highlighted words are in Roman letters", (c["hook"], c["title"], c["highlight_words"]) == ("CHUNA LAGA", "usko chuna", ["chuna", "boss"]), str(c))

print("ALL PASSED" if not fails else f"{len(fails)} failed")
import shutil; shutil.rmtree(os.environ["PEAKCUT_DATA"], ignore_errors=True)
sys.exit(1 if fails else 0)
