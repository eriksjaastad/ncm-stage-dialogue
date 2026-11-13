"""Build the static public research publication from committed report artifacts."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min"


def _sample(results: dict[str, Any], model: str) -> dict[str, Any]:
    return next(
        sample
        for sample in results["generation"]["samples"]
        if sample["model"] == model and sample["boundary"] == "early"
    )


def _snippet(sample: dict[str, Any], limit: int = 420) -> str:
    text = sample["continuation"].strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + "…"
    return html.escape(text)


def _metric_card(label: str, value: str, note: str, tone: str = "neutral") -> str:
    return (
        f'<article class="metric metric--{tone}"><p>{html.escape(label)}</p>'
        f'<strong>{html.escape(value)}</strong><span>{html.escape(note)}</span></article>'
    )


def _result_row(label: str, textnca: str, transformer: str, winner: str = "") -> str:
    return (
        "<tr>"
        f"<th>{html.escape(label)}</th>"
        f'<td class="{("winner" if winner == "textnca" else "")}">{html.escape(textnca)}</td>'
        f'<td class="{("winner" if winner == "transformer" else "")}">{html.escape(transformer)}</td>'
        "</tr>"
    )


def _safe_results(results: dict[str, Any], model_names: tuple[str, str]) -> dict[str, Any]:
    """Return publication evidence without local paths or checkpoint identifiers."""
    return {
        "schema_version": results["schema_version"],
        "identity": {
            "run_id": results["identity"]["run_id"],
            "training_commit": results["identity"]["training_commit"],
        },
        "dataset": {
            key: results["dataset"][key]
            for key in (
                "block_length",
                "corpus_sha256",
                "record_count",
                "split_counts",
                "split_token_counts",
                "turn_count",
                "vocab_size",
                "work_count",
            )
        },
        "protocol": results["protocol"],
        "training": {
            "comparison": results["training"]["comparison"],
            "models": {name: results["training"]["models"][name] for name in model_names},
        },
        "generation": {"summary": results["generation"]["summary"]},
        **(
            {
                "v1_to_v2": results["v1_to_v2"],
                "textnca_investment_decision": results["textnca_investment_decision"],
            }
            if "v1_to_v2" in results
            else {}
        ),
    }


def _safe_config(config: dict[str, Any]) -> dict[str, Any]:
    return {"model": config["model"], "training": config["training"]}


def _page_shell(title: str, description: str, body: str, canonical_url: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <meta name="description" content="{html.escape(description)}">
  <meta name="robots" content="noindex,nofollow,noarchive">
  <link rel="canonical" href="{html.escape(canonical_url)}">
  <link rel="stylesheet" href="assets/style.css">
</head>
<body>
{body}
</body>
</html>
"""


def build_publication(
    repository: Path, output_dir: Path, site_url: str, repository_url: str
) -> None:
    """Render a static, evidence-linked account of the V1 and V2 experiments."""
    repository = repository.resolve()
    output_dir = output_dir.resolve()
    v1 = _load(repository / "reports/v1/results.json")
    v2 = _load(repository / "reports/v2/results.json")
    corpus_v2 = _load(repository / "reports/corpus-v2/results.json")
    sources = _load(repository / "sources/plays.json")["sources"]
    configs = {
        name: _load(repository / f"configs/{name}.json")
        for name in ("v1-textnca", "v1-transformer", "v2-textnca", "v2-transformer")
    }

    v1_tn = v1["training"]["models"]["v1-textnca"]
    v1_tr = v1["training"]["models"]["v1-transformer"]
    v2_tn = v2["training"]["models"]["v2-textnca"]
    v2_tr = v2["training"]["models"]["v2-transformer"]
    v1_tn_gen = v1["generation"]["summary"]["v1-textnca"]
    v1_tr_gen = v1["generation"]["summary"]["v1-transformer"]
    v2_tn_gen = v2["generation"]["summary"]["v2-textnca"]
    v2_tr_gen = v2["generation"]["summary"]["v2-transformer"]
    tn_sample = _sample(v2, "v2-textnca")
    tr_sample = _sample(v2, "v2-transformer")
    decision = v2["textnca_investment_decision"]["decision"]
    train_steps = configs["v2-textnca"]["training"]["train_steps"]
    architecture_windows = configs["v2-textnca"]["model"]["windows"]
    transformer_speedup = v2["training"]["comparison"]["transformer_training_speedup"]
    textnca_loss = v2_tn["test"]["loss"]
    transformer_loss = v2_tr["test"]["loss"]
    textnca_unique = v2_tn_gen["cross_prompt"]["unique_continuation_rate"]
    transformer_unique = v2_tr_gen["cross_prompt"]["unique_continuation_rate"]

    metrics = "".join(
        [
            _metric_card(
                "Final decision",
                f"{decision.title()} TextNCA",
                "Preserve it as a negative result",
                "bad",
            ),
            _metric_card(
                "V2 test loss",
                f"{transformer_loss:.3f} vs {textnca_loss:.3f}",
                "Transformer vs TextNCA",
                "good",
            ),
            _metric_card(
                "Training speed",
                f"{transformer_speedup:.2f}× faster",
                "Transformer end-to-end",
                "good",
            ),
            _metric_card(
                "Unique outputs",
                f"{transformer_unique:.0%} vs {textnca_unique:.0%}",
                "Transformer vs TextNCA",
                "good",
            ),
        ]
    )

    v1_rows = "".join(
        [
            _result_row("Test loss ↓", _fmt(v1_tn["test"]["loss"]), _fmt(v1_tr["test"]["loss"]), "transformer"),
            _result_row("Training time ↓", _minutes(v1_tn["elapsed_seconds"]), _minutes(v1_tr["elapsed_seconds"]), "transformer"),
            _result_row(
                "Unique continuations ↑",
                f'{v1_tn_gen["cross_prompt"]["unique_continuation_rate"]:.1%}',
                f'{v1_tr_gen["cross_prompt"]["unique_continuation_rate"]:.1%}',
                "transformer",
            ),
            _result_row(
                "Cross-work similarity ↓",
                _fmt(v1_tn_gen["cross_prompt"]["mean_cross_work_character_trigram_jaccard"]),
                _fmt(v1_tr_gen["cross_prompt"]["mean_cross_work_character_trigram_jaccard"]),
                "transformer",
            ),
            _result_row(
                "Token repeated trigrams ↓",
                f'{v1_tn_gen["macro_average_across_works"]["token_repeated_trigram_rate"]:.1%}',
                f'{v1_tr_gen["macro_average_across_works"]["token_repeated_trigram_rate"]:.1%}',
                "transformer",
            ),
            _result_row(
                "Character repeated trigrams ↓",
                f'{v1_tn_gen["macro_average_across_works"]["character_repeated_trigram_rate"]:.1%}',
                f'{v1_tr_gen["macro_average_across_works"]["character_repeated_trigram_rate"]:.1%}',
                "transformer",
            ),
            _result_row(
                "Allowed held-out speaker ↑",
                f'{v1_tn_gen["macro_average_across_works"]["speaker_consistency"]:.1%}',
                f'{v1_tr_gen["macro_average_across_works"]["speaker_consistency"]:.1%}',
            ),
        ]
    )
    v2_rows = "".join(
        [
            _result_row("Test loss ↓", _fmt(v2_tn["test"]["loss"]), _fmt(v2_tr["test"]["loss"]), "transformer"),
            _result_row("Training time ↓", _minutes(v2_tn["elapsed_seconds"]), _minutes(v2_tr["elapsed_seconds"]), "transformer"),
            _result_row(
                "Unique continuations ↑",
                f'{v2_tn_gen["cross_prompt"]["unique_continuation_rate"]:.1%}',
                f'{v2_tr_gen["cross_prompt"]["unique_continuation_rate"]:.1%}',
                "transformer",
            ),
            _result_row(
                "Cross-work similarity ↓",
                _fmt(v2_tn_gen["cross_prompt"]["mean_cross_work_character_trigram_jaccard"]),
                _fmt(v2_tr_gen["cross_prompt"]["mean_cross_work_character_trigram_jaccard"]),
                "transformer",
            ),
            _result_row(
                "Token repeated trigrams ↓",
                f'{v2_tn_gen["macro_average_across_works"]["token_repeated_trigram_rate"]:.1%}',
                f'{v2_tr_gen["macro_average_across_works"]["token_repeated_trigram_rate"]:.1%}',
                "textnca",
            ),
            _result_row(
                "Character repeated trigrams ↓",
                f'{v2_tn_gen["macro_average_across_works"]["character_repeated_trigram_rate"]:.1%}',
                f'{v2_tr_gen["macro_average_across_works"]["character_repeated_trigram_rate"]:.1%}',
                "textnca",
            ),
            _result_row(
                "Prefix-valid continuation ↑",
                f'{v2_tn_gen["macro_average_across_works"]["prefix_valid"]:.1%}',
                f'{v2_tr_gen["macro_average_across_works"]["prefix_valid"]:.1%}',
                "textnca",
            ),
            _result_row(
                "Allowed held-out speaker ↑",
                f'{v2_tn_gen["macro_average_across_works"]["speaker_consistency"]:.1%}',
                f'{v2_tr_gen["macro_average_across_works"]["speaker_consistency"]:.1%}',
            ),
        ]
    )

    body = f"""
<div class="draft-banner"><strong>Review draft</strong><span>Not indexed · awaiting editorial changes</span><a href="{html.escape(repository_url)}/issues/new">Suggest a change</a></div>
<header class="hero" id="top">
  <nav><a class="wordmark" href="#top">NCM Stage Dialogue</a><div><a href="#stages">Stages</a><a href="#results">Results</a><a href="#artifacts">Artifacts</a><a href="{html.escape(repository_url)}">Edit on GitHub</a></div></nav>
  <div class="hero__grid">
    <div>
      <p class="eyebrow">Final experiment report · V1 → V2</p>
      <h1>We tested whether a TextNCA could learn stage dialogue. It did not.</h1>
      <p class="lede">A controlled, reproducible comparison against a parameter-matched Transformer—two corpora, four {train_steps:,}-step runs, structured generation probes, and one clear stopping decision.</p>
      <div class="hero__actions"><a class="button" href="#conclusion">Read the conclusion</a><a class="button button--ghost" href="data/v2-evidence.json">Download V2 data</a></div>
    </div>
    <aside class="verdict"><span>Verdict</span><strong>Transformer wins</strong><p>TextNCA improved with more data, but remained slower, less accurate, and severely prompt-insensitive.</p></aside>
  </div>
</header>
<main>
  <section class="metric-grid" aria-label="Headline results">{metrics}</section>

  <section class="section prose" id="question">
    <p class="eyebrow">Research question</p>
    <h2>Can iterative local updates compete with a Transformer on structured dramatic dialogue?</h2>
    <p>We implemented the primary TextNCA architecture with local-attention windows <code>{html.escape(str(architecture_windows))}</code> and compared it with a conventional Transformer under the same byte vocabulary, context length, optimizer, seed, batch size, data stream, and training budget. Parameter counts differed by only {v2["training"]["comparison"]["parameter_difference_percent"]:.3f}%.</p>
    <p>The experiment was not designed to produce a commercial chatbot. It was designed to find out whether this architecture deserved another round of investment.</p>
  </section>

  <section class="section" id="stages">
    <p class="eyebrow">Every stage</p><h2>From corpus design to stopping rule</h2>
    <ol class="timeline">
      <li><span>01</span><div><h3>Corpus V1</h3><p>{v1["dataset"]["work_count"]} public-domain plays, {v1["dataset"]["record_count"]} scenes, and {v1["dataset"]["turn_count"]:,} turns. Complete works were isolated by split.</p></div></li>
      <li><span>02</span><div><h3>Locked V1 comparison</h3><p>Both ~27M-parameter models trained for 10,000 steps. The Transformer won on loss and speed; TextNCA collapsed across prompts.</p></div></li>
      <li><span>03</span><div><h3>Corpus V2</h3><p>Expanded to {v2["dataset"]["work_count"]} works, {v2["dataset"]["record_count"]} scenes, and {v2["dataset"]["turn_count"]:,} turns—{corpus_v2["delta"]["tokens_percent"]:.1f}% more tokens—with author and coauthor isolation.</p></div></li>
      <li><span>04</span><div><h3>Locked V2 comparison</h3><p>Architecture and training settings stayed fixed. Only the frozen corpus and run destinations changed.</p></div></li>
      <li><span>05</span><div><h3>Generation-quality probes</h3><p>Early, middle, and late held-out boundaries measured repetition, exact diversity, cross-prompt collapse, grammar state, speaker formatting, and held-out speaker identity.</p></div></li>
      <li><span>06</span><div><h3>Stop decision</h3><p>TextNCA lost on held-out loss and speed and showed no advantage on output diversity or prompt sensitivity. Further TextNCA training was stopped.</p></div></li>
    </ol>
  </section>

  <section class="section provenance" id="provenance">
    <p class="eyebrow">Provenance and locked protocol</p><h2>Every stage is tied to a run, revision, corpus hash, and checked-in configuration.</h2>
    <p class="section-intro">The corpus contains {v2["dataset"]["work_count"]} works parsed from {len(sources)} hash-pinned Project Gutenberg source files. Every source records retrieval date, byte count, SHA-256, United States public-domain basis, parser, and split assignment. Raw source text and checkpoints are not published.</p>
    <div class="provenance-grid">
      <article><span>V1 run</span><strong>{html.escape(v1["identity"]["run_id"])}</strong><p>Commit <code>{v1["identity"]["training_commit"]}</code><br>Corpus <code>{v1["dataset"]["corpus_sha256"]}</code><br>{v1_tn["parameters"]:,} / {v1_tr["parameters"]:,} parameters<br>{configs["v1-textnca"]["training"]["train_steps"]:,} steps · seed {configs["v1-textnca"]["training"]["seed"]}</p></article>
      <article><span>V2 run</span><strong>{html.escape(v2["identity"]["run_id"])}</strong><p>Commit <code>{v2["identity"]["training_commit"]}</code><br>Corpus <code>{v2["dataset"]["corpus_sha256"]}</code><br>{v2_tn["parameters"]:,} / {v2_tr["parameters"]:,} parameters<br>{train_steps:,} steps · seed {configs["v2-textnca"]["training"]["seed"]}</p></article>
    </div>
  </section>

  <section class="section" id="results">
    <p class="eyebrow">Results</p><h2>More data helped both models. It helped the Transformer much more.</h2>
    <div class="result-grid">
      <article class="panel"><div class="panel__head"><span>V1</span><strong>{v1["dataset"]["work_count"]} works</strong></div><img src="assets/v1-training-curves.svg" alt="V1 validation loss curves"><table><thead><tr><th>Metric</th><th>TextNCA</th><th>Transformer</th></tr></thead><tbody>{v1_rows}</tbody></table></article>
      <article class="panel"><div class="panel__head"><span>V2</span><strong>{v2["dataset"]["work_count"]} works</strong></div><img src="assets/v2-training-curves.svg" alt="V2 validation loss curves"><table><thead><tr><th>Metric</th><th>TextNCA</th><th>Transformer</th></tr></thead><tbody>{v2_rows}</tbody></table></article>
    </div>
    <div class="callout"><strong>The widening gap matters.</strong><p>From V1 to V2, TextNCA test loss improved {abs(v2["v1_to_v2"]["textnca"]["test_loss_change_percent"]):.1f}%. Transformer test loss improved {abs(v2["v1_to_v2"]["transformer"]["test_loss_change_percent"]):.1f}%.</p></div>
  </section>

  <section class="section" id="samples">
    <p class="eyebrow">Representative output</p><h2>The metrics are clearer when you see the text.</h2>
    <p class="section-intro">These are predeclared early-boundary samples, not hand-picked successes or failures. Both models still generate poor dialogue. TextNCA's repeated shape across unrelated plays is the important failure.</p>
    <div class="sample-grid">
      <article class="sample sample--bad"><div><span>TextNCA · V2</span><strong>{html.escape(tn_sample["title"])}</strong></div><pre>{_snippet(tn_sample)}</pre></article>
      <article class="sample"><div><span>Transformer · V2</span><strong>{html.escape(tr_sample["title"])}</strong></div><pre>{_snippet(tr_sample)}</pre></article>
    </div>
    <p class="fineprint">Neither V2 model generated a correct held-out speaker in these probes, and neither completed a full serialized work. The Transformer is the stronger research baseline, not a production-ready dialogue model.</p>
  </section>

  <section class="section conclusion" id="conclusion">
    <p class="eyebrow">Conclusion</p><h2>The experiment succeeded by telling us to stop.</h2>
    <p>TextNCA is not worth another training round in this project. It was {v2["training"]["comparison"]["transformer_training_speedup"]:.2f}× slower to train, had {v2["training"]["comparison"]["textnca_test_loss_relative_percent"]:.1f}% higher V2 test loss, produced only {v2_tn_gen["cross_prompt"]["unique_continuation_rate"]:.0%} unique continuations, and retained {v2_tn_gen["cross_prompt"]["mean_cross_work_character_trigram_jaccard"]:.3f} cross-work character-trigram similarity.</p>
    <p>If convincing stage dialogue remains the goal, the next rational experiment is fine-tuning a capable pretrained language model—not V3 of this TextNCA.</p>
  </section>

  <section class="section" id="limitations">
    <p class="eyebrow">Limitations</p><h2>What this result does—and does not—say</h2>
    <ul class="limitations"><li>One training seed and one generation seed.</li><li>English-language public-domain drama only.</li><li>Near-parameter matching, not exact matching.</li><li>No architecture or optimizer ablations.</li><li>Generation probes are descriptive, not human-preference evaluations.</li><li>This rejects this implementation and scale; it does not prove every neural cellular automaton is inferior.</li></ul>
  </section>

  <section class="section artifacts" id="artifacts">
    <p class="eyebrow">Evidence</p><h2>Committed artifacts, not reconstructed claims</h2>
    <div class="artifact-grid"><a href="data/v1-evidence.json"><strong>V1 evidence</strong><span>Sanitized metrics, protocol, hashes</span></a><a href="data/v2-evidence.json"><strong>V2 evidence</strong><span>Final metrics and decision rule</span></a><a href="data/corpus-v2-evidence.json"><strong>Corpus V2</strong><span>Scale, rationale, limitations</span></a><a href="data/experiment-configs.json"><strong>Locked configs</strong><span>All four model/training configs</span></a><a href="{html.escape(repository_url)}"><strong>Edit the draft</strong><span>Source, issues, and revision history</span></a></div>
    <p class="metric-definition"><strong>Metric definitions:</strong> repeated-trigram rates measure within-sample repetition; exact unique-continuation rate measures duplicate outputs; cross-work character-trigram Jaccard measures similarity between prompts from different held-out plays. Lower repetition and cross-work similarity are better; higher uniqueness is better.</p>
    <p class="hashes">V2 training commit <code>{v2["identity"]["training_commit"]}</code><br>Corpus SHA-256 <code>{v2["dataset"]["corpus_sha256"]}</code></p>
  </section>
</main>
<footer><p>NCM Stage Dialogue · Review draft · Negative results are results.</p><div><a href="privacy.html">Privacy</a><a href="sitemap.xml">Sitemap</a><a href="{html.escape(repository_url)}">GitHub</a><a href="#top">Back to top</a></div></footer>
"""

    privacy_body = f"""
<div class="draft-banner"><strong>Review draft</strong><span>Not indexed · awaiting editorial changes</span><a href="{html.escape(repository_url)}/issues/new">Suggest a change</a></div>
<header class="subhero"><nav><a class="wordmark" href="index.html">NCM Stage Dialogue</a></nav><p class="eyebrow">Privacy</p><h1>A static research publication with no tracking.</h1></header>
<main><section class="section prose"><h2>What this site collects</h2><p>This publication does not use analytics, cookies, accounts, forms, advertising, or a database. GitHub Pages may process standard request data such as IP address, browser information, requested URL, and timestamp for security and delivery.</p><h2>Who processes it</h2><p>GitHub Pages hosts and delivers the static files. Its infrastructure may use subprocessors described in GitHub’s privacy documentation.</p><h2>Retention and contact</h2><p>This site does not maintain its own visitor records. Hosting logs are governed by GitHub’s retention practices. Review notes and corrections can be filed in the public repository.</p><p><a class="button button--ghost" href="index.html">Return to the report</a></p></section></main>
<footer><p>NCM Stage Dialogue · Review draft</p><div><a href="index.html">Report</a><a href="{html.escape(repository_url)}">GitHub</a></div></footer>
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    assets = output_dir / "assets"
    data = output_dir / "data"
    assets.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    (output_dir / "index.html").write_text(
        _page_shell(
            "NCM Stage Dialogue — Final TextNCA Experiment Results",
            "A controlled V1-to-V2 comparison of TextNCA and Transformer models on public-domain stage dialogue.",
            body,
            site_url.rstrip("/") + "/",
        ),
        encoding="utf-8",
    )
    (output_dir / "privacy.html").write_text(
        _page_shell(
            "Privacy — NCM Stage Dialogue",
            "Privacy details for the static NCM Stage Dialogue research publication.",
            privacy_body,
            site_url.rstrip("/") + "/privacy.html",
        ),
        encoding="utf-8",
    )
    shutil.copyfile(repository / "site/style.css", assets / "style.css")
    shutil.copyfile(repository / "reports/v1/training-curves.svg", assets / "v1-training-curves.svg")
    shutil.copyfile(repository / "reports/v2/training-curves.svg", assets / "v2-training-curves.svg")
    evidence = {
        "v1": _safe_results(v1, ("v1-textnca", "v1-transformer")),
        "v2": _safe_results(v2, ("v2-textnca", "v2-transformer")),
    }
    corpus_evidence = {
        key: corpus_v2[key]
        for key in (
            "corpus_version",
            "delta",
            "manifest_sha256",
            "rationale",
            "remaining_limitations",
            "schema_version",
            "v1",
            "v2",
        )
    }
    corpus_evidence["source_provenance"] = {
        "source_file_count": len(sources),
        "provider": "Project Gutenberg",
        "jurisdiction": "United States",
        "all_sources_hash_pinned": all(source.get("source_sha256") for source in sources),
        "all_sources_record_public_domain_basis": all(
            source.get("public_domain_basis") for source in sources
        ),
    }
    config_evidence = {name: _safe_config(config) for name, config in configs.items()}
    for destination, payload in (
        (data / "v1-evidence.json", evidence["v1"]),
        (data / "v2-evidence.json", evidence["v2"]),
        (data / "corpus-v2-evidence.json", corpus_evidence),
        (data / "experiment-configs.json", config_evidence),
    ):
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    sitemap = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>{html.escape(site_url.rstrip('/') + '/')}</loc></url>
  <url><loc>{html.escape(site_url.rstrip('/') + '/privacy.html')}</loc></url>
</urlset>
"""
    (output_dir / "sitemap.xml").write_text(sitemap, encoding="utf-8")
    (output_dir / "robots.txt").write_text(
        "User-agent: *\nDisallow: /\n# Review draft: indexing remains disabled until approval.\n",
        encoding="utf-8",
    )
    (output_dir / ".nojekyll").write_text("", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the static NCM research publication.")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("docs"))
    parser.add_argument(
        "--site-url", default="https://eriksjaastad.github.io/ncm-stage-dialogue"
    )
    parser.add_argument(
        "--repository-url", default="https://github.com/eriksjaastad/ncm-stage-dialogue"
    )
    args = parser.parse_args()
    build_publication(args.repository, args.output_dir, args.site_url, args.repository_url)
    print(json.dumps({"output": str(args.output_dir), "site_url": args.site_url}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
