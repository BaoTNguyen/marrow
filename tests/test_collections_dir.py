from marrow import cli
from marrow.collect import collect_cli_run, default_collections_dir


def test_default_collections_dir_follows_cwd(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    want = tmp_path / ".vascular" / "marrow" / "collections"
    assert default_collections_dir().resolve() == want.resolve()
    out, _ = collect_cli_run("codex", "x", dry_run=True)
    assert out.parent.resolve() == want.resolve()
    cli.main(["collect", "cli-run", "claude", "x", "--dry-run"])
    assert str(want.resolve()) in capsys.readouterr().out
    assert not want.exists()
