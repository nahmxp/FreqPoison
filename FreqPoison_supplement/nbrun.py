"""Execute the original notebook's code cells verbatim (pip line removed, plotting off)."""
import json, os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: plt.close("all")
NB = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "FreqPoison_Experiment.ipynb")))
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
def load(upto=None, quiet=True):
    import io, contextlib
    ns = {}
    cells = CODE if upto is None else CODE[:upto]
    for src in cells:
        src = "\n".join(l for l in src.splitlines() if not l.strip().startswith("!"))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(src, "<nb>", "exec"), ns)
        if not quiet: print(buf.getvalue())
    return ns
if __name__ == "__main__":
    ns = load(upto=8, quiet=False)   # through results tables
