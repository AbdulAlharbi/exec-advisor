"""advisor — run the multi-agent executive advisor from the command line."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .audit import AuditLog
from .chief_of_staff import BriefStore, ChiefOfStaff
from .config import Settings
from .llm.base import make_brain
from .models import Brief
from .sources import SourceManifest

app = typer.Typer(add_completion=False, help="Multi-agent executive advisor with verifiable grounding.")
console = Console()


def _settings(brain: Optional[str], out: Optional[Path], site: Optional[Path]) -> Settings:
    s = Settings()
    if brain:
        s.brain = brain
    if out:
        s.out_dir = out
    if site:
        s.site_dir = site
    return s


def _show(b: Brief) -> None:
    console.print(Panel(f"[bold]{b.headline}[/bold]\n\n{b.summary}",
                        title=f"{b.run_id} · health {b.health_score}/100 · {b.posture} · {b.brain}", border_style="blue"))
    if b.changes:
        t = Table(title=f"changes since {b.previous_run_id}", show_lines=False)
        t.add_column("type"); t.add_column("kind"); t.add_column("finding"); t.add_column("note")
        for c in b.changes:
            style = {"new": "red", "escalated": "red", "resolved": "green", "improved": "green"}.get(c.type, "dim")
            t.add_row(f"[{style}]{c.type}[/{style}]", c.kind, c.title, c.note)
        console.print(t)
    for kind, items, label in (("risk", b.risks, "severity"), ("opportunity", b.opportunities, "impact")):
        t = Table(title=f"{kind}s", show_lines=True)
        t.add_column("id"); t.add_column("finding", max_width=48); t.add_column(label); t.add_column("status"); t.add_column("evidence"); t.add_column("contrarian", max_width=48)
        for f in items:
            ev = "\n".join(f'"{b.claim(c).quote[:70]}…"' if b.claim(c) and len(b.claim(c).quote) > 70 else f'"{b.claim(c).quote}"' for c in f.evidence if b.claim(c))
            obj = "\n".join(o.replace("contrarian ", "") for o in f.objections)
            colour = {"critical": "bold red", "high": "red", "medium": "yellow", "low": "dim"}[f.level]
            t.add_row(f.id, f.title, f"[{colour}]{f.level}[/{colour}]", f.status, ev, obj)
        console.print(t)
    if b.assumptions:
        t = Table(title="assumptions worth challenging", show_lines=True)
        t.add_column("assumption", max_width=40); t.add_column("why questionable", max_width=60); t.add_column("test", max_width=48)
        for a in b.assumptions:
            t.add_row(a.statement, a.why_questionable, a.test)
        console.print(t)
    t = Table(title="recommendations", show_lines=True)
    t.add_column("#"); t.add_column("action", max_width=60); t.add_column("owner"); t.add_column("when"); t.add_column("success looks like", max_width=44)
    for i, r in enumerate(b.recommendations, 1):
        t.add_row(str(i), r.action, r.owner, r.timeline, r.expected_outcome)
    console.print(t)
    g = b.grounding
    console.print(f"[dim]grounding: claims {g.claims_extracted} extracted · {g.claims_verified} verified · {g.claims_dropped} dropped · "
                  f"findings {g.findings_proposed} proposed · {g.findings_dropped} dropped · {g.findings_weakened} weakened · "
                  f"decided by {', '.join(f'{k}={v}' for k, v in b.decided_by.items())}[/dim]")


@app.command()
def run(
    sources: Optional[Path] = typer.Option(None, "--sources", "-s", help="sources.yaml manifest"),
    brain: Optional[str] = typer.Option(None, "--brain", "-b", help="auto | claude | heuristic"),
    out: Optional[Path] = typer.Option(None, "--out", help="output directory"),
    site: Optional[Path] = typer.Option(None, "--site", help="site directory (docs/) to publish into"),
    publish: bool = typer.Option(True, help="write docs/data for the dashboard"),
    previous: Optional[str] = typer.Option(None, help="run id to diff against (default: latest run)"),
    show: bool = typer.Option(True, help="print the brief"),
) -> None:
    """Analyse the sources once, diff against the previous run, write the report and publish the site data."""
    s = _settings(brain, out, site)
    if sources:
        s.sources_file = sources
    kind = s.resolve_brain()
    manifest = SourceManifest.load(s.sources_file)
    cos = ChiefOfStaff(make_brain(kind, s.model), s)
    prev = cos.store.load(previous) if previous else cos.store.latest()
    console.print(f"[dim]brain={kind} sources={s.sources_file} previous={prev.run_id if prev else 'none'}[/dim]")
    brief = cos.run(manifest, prev)
    if publish:
        cos.publish(brief)
    if show:
        _show(brief)
    console.print(f"[green]done[/green] {brief.run_id} → {s.runs_dir / (brief.run_id + '.json')}")


@app.command()
def show(run_id: Optional[str] = typer.Argument(None), out: Optional[Path] = typer.Option(None, "--out")) -> None:
    """Print a past brief (default: the latest)."""
    s = _settings(None, out, None)
    store = BriefStore(s.runs_dir)
    b = store.load(run_id) if run_id else store.latest()
    if b is None:
        raise typer.Exit("no runs yet")
    _show(b)


@app.command()
def history(out: Optional[Path] = typer.Option(None, "--out")) -> None:
    """List past runs."""
    s = _settings(None, out, None)
    t = Table()
    t.add_column("run"); t.add_column("at"); t.add_column("brain"); t.add_column("health"); t.add_column("posture"); t.add_column("headline", max_width=80)
    for b in BriefStore(s.runs_dir).list():
        t.add_row(b.run_id, f"{b.at:%Y-%m-%d %H:%M}", b.brain, str(b.health_score), b.posture, b.headline)
    console.print(t)


@app.command()
def audit(run_id: Optional[str] = typer.Argument(None), out: Optional[Path] = typer.Option(None, "--out")) -> None:
    """Print the audit trail and verify the hash chain."""
    s = _settings(None, out, None)
    log = AuditLog(s.audit_path)
    t = Table()
    t.add_column("at"); t.add_column("run"); t.add_column("event"); t.add_column("payload", max_width=90)
    for e in log.entries():
        if run_id and e.get("run_id") != run_id:
            continue
        t.add_row(e["at"][11:19], e.get("run_id") or "", e["event"], str(e["payload"])[:300])
    console.print(t)
    ok, n, msg = log.verify()
    console.print(f"[{'green' if ok else 'red'}]{msg}[/{'green' if ok else 'red'}]")


@app.command()
def sources(path: Optional[Path] = typer.Option(None, "--sources", "-s")) -> None:
    """List the sources in the manifest and check they load."""
    s = _settings(None, None, None)
    m = SourceManifest.load(path or s.sources_file)
    t = Table()
    t.add_column("id"); t.add_column("title"); t.add_column("kind"); t.add_column("chars"); t.add_column("status")
    for src in m.materialise(s.fetch_timeout, s.max_source_chars):
        t.add_row(src.id, src.title, src.kind, f"{src.chars:,}", src.error or "[green]ok[/green]")
    console.print(t)


if __name__ == "__main__":
    app()
