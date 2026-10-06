"""Tests for marrow.vascular_paths."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from marrow.vascular_paths import KINDS, home, journal_dir, path, repo_dir


class TestKinds:
    def test_kinds_is_tuple_of_strings(self):
        assert isinstance(KINDS, tuple)
        assert KINDS == ("config", "secrets", "state", "spool", "log", "cache", "data", "backups")


class TestHome:
    def test_home_default(self, monkeypatch):
        monkeypatch.delenv("VASCULAR_HOME", raising=False)
        h = home()
        assert h == Path.home() / ".vascular"

    def test_home_overridden(self, monkeypatch):
        monkeypatch.setenv("VASCULAR_HOME", "/custom/vascular")
        h = home()
        assert h == Path("/custom/vascular")


class TestPath:
    def test_valid_kind(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path))
        p = path("config", "heart", "settings.json")
        assert p == tmp_path / "config" / "heart" / "settings.json"

    def test_multiple_parts(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path))
        p = path("state", "heart", "events", "2024", "log.txt")
        assert p == tmp_path / "state" / "heart" / "events" / "2024" / "log.txt"

    def test_new_kinds(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path))
        assert path("secrets", "x") == tmp_path / "secrets" / "x"
        assert path("spool", "x") == tmp_path / "spool" / "x"
        assert path("log", "x") == tmp_path / "log" / "x"

    def test_invalid_kind_raises(self, monkeypatch):
        monkeypatch.delenv("VASCULAR_HOME", raising=False)
        with pytest.raises(ValueError, match="unknown kind"):
            path("unknown", "heart")

    def test_does_not_create_directory(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path))
        p = path("config", "heart", "deep", "nested")
        # path() creates nothing
        assert not p.exists()


class TestJournalDir:
    def test_journal_dir_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path))
        monkeypatch.delenv("EVENT_JOURNAL_DIR", raising=False)
        j = journal_dir()
        assert j == tmp_path / "spool" / "events"

    def test_journal_dir_overridden(self, monkeypatch, tmp_path):
        monkeypatch.setenv("EVENT_JOURNAL_DIR", "/custom/events")
        j = journal_dir()
        assert j == Path("/custom/events")

    def test_journal_dir_ignores_vascular_home_when_env_set(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VASCULAR_HOME", str(tmp_path))
        monkeypatch.setenv("EVENT_JOURNAL_DIR", "/override/events")
        j = journal_dir()
        assert j == Path("/override/events")


class TestRepoDir:
    def test_repo_dir_string_root(self, tmp_path):
        d = repo_dir(str(tmp_path), "heart")
        assert d == tmp_path / ".vascular" / "heart"

    def test_repo_dir_path_root(self, tmp_path):
        d = repo_dir(tmp_path, "heart")
        assert d == tmp_path / ".vascular" / "heart"

    def test_repo_dir_does_not_create(self, tmp_path):
        d = repo_dir(str(tmp_path), "heart")
        assert not d.exists()

    def test_repo_dir_different_components(self, tmp_path):
        assert repo_dir(str(tmp_path), "capillaries") == tmp_path / ".vascular" / "capillaries"
        assert repo_dir(str(tmp_path), "arteries") == tmp_path / ".vascular" / "arteries"

    def test_repo_dir_preserves_nested_root(self, tmp_path):
        nested = tmp_path / "sub" / "repo"
        nested.mkdir(parents=True)
        d = repo_dir(str(nested), "heart")
        assert d == nested / ".vascular" / "heart"
