"""Auto-save every matplotlib figure into ../figures/<prefix>_NN.png.

Why this exists: many analysis notebooks call plt.show() without an
accompanying fig.savefig(), so plots only live inside the .ipynb and
never reach the figures/ folder. Calling `autosave("12_uniform")` once
at the top of a notebook patches plt.show() so each subsequent show()
also writes the current figure to disk under the chosen prefix.

Existing explicit fig.savefig() calls keep working — this only adds
a save before show, it does not remove anything.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

_FIGS_DIR = Path(__file__).resolve().parent.parent / "figures"
_state = {"prefix": "fig", "counter": 0, "patched": False, "orig_show": None, "dpi": 130}


def autosave(prefix: str, dpi: int = 130, figs_dir: str | None = None) -> Path:
    """Patch plt.show() so it also writes <prefix>_NN.png to figures/.

    Safe to call multiple times — re-patching just resets the counter
    and prefix for the new notebook.
    """
    if figs_dir is not None:
        global _FIGS_DIR
        _FIGS_DIR = Path(figs_dir).resolve()
    _FIGS_DIR.mkdir(parents=True, exist_ok=True)

    _state["prefix"]  = prefix
    _state["counter"] = 0
    _state["dpi"]     = int(dpi)

    if not _state["patched"]:
        _state["orig_show"] = plt.show

        def _show_and_save(*args, **kwargs):
            # Save every open figure (not just gcf) so multi-figure cells
            # don't lose any. Filenames stay deterministic per show() call.
            for num in plt.get_fignums():
                _state["counter"] += 1
                fig = plt.figure(num)
                out = _FIGS_DIR / f"{_state['prefix']}_{_state['counter']:02d}.png"
                try:
                    fig.savefig(out, dpi=_state["dpi"], bbox_inches="tight")
                except Exception as e:
                    print(f"[figsave] failed to save {out.name}: {e}")
            return _state["orig_show"](*args, **kwargs)

        plt.show = _show_and_save
        _state["patched"] = True

    print(f"[figsave] autosave active: {_FIGS_DIR}/{prefix}_NN.png")
    return _FIGS_DIR
