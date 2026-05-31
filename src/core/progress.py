"""Barre de progression simple pour le CLI (sans dépendance externe)."""

import sys
from typing import Optional


class ProgressBar:
    """Affiche la progression current/total avec barre ASCII."""

    def __init__(
        self,
        total: int,
        prefix: str = "",
        width: int = 32,
        enabled: bool = True,
    ):
        self.total = max(total, 1)
        self.prefix = prefix
        self.width = width
        self.enabled = enabled and total > 0
        self.use_bar = self.enabled and sys.stdout.isatty()
        self.current = 0
        self._last_logged_pct = -1

    def update(self, current: Optional[int] = None, suffix: str = "") -> None:
        if not self.enabled:
            return

        if current is not None:
            self.current = min(current, self.total)
        else:
            self.current = min(self.current + 1, self.total)

        pct = int(100 * self.current / self.total)

        if self.use_bar:
            filled = int(self.width * self.current / self.total)
            bar = "=" * filled
            if filled < self.width:
                bar += ">"
                bar += " " * (self.width - filled - 1)
            line = f"\r  {self.prefix} [{bar}] {self.current}/{self.total} ({pct}%)"
            if suffix:
                line += f" {suffix}"
            sys.stdout.write(line.ljust(100))
            sys.stdout.flush()
        elif pct >= self._last_logged_pct + 10 or self.current == self.total:
            self._last_logged_pct = pct
            msg = f"  {self.prefix}: {self.current}/{self.total} ({pct}%)"
            if suffix:
                msg += f" {suffix}"
            print(msg)

    def finish(self, suffix: str = "") -> None:
        if not self.enabled:
            return
        self.update(self.total, suffix=suffix)
        if self.use_bar:
            sys.stdout.write("\n")
            sys.stdout.flush()
