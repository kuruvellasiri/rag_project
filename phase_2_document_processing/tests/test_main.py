from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import main


def test_clean_text_keeps_paragraphs_and_removes_extra_spaces():
    assert main.clean_text("  A   title  \n\n\n B\t  line ") == "A title\n\nB line"


def test_json_loader_returns_consistent_record(tmp_path):
    path = tmp_path / "sample.json"
    path.write_text('{"topic":"rag"}', encoding="utf-8")
    record = main.load_document(path)[0]
    assert set(record) == {"document_id", "source", "file_type", "page", "text"}
    assert record["file_type"] == "json"
    assert record["page"] is None
    assert '"topic": "rag"' in record["text"]


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "sample.bin"
    path.write_bytes(b"x")
    try:
        main.load_document(path)
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("Expected unsupported extension to fail")


def test_batch_continues_after_bad_file(tmp_path):
    (tmp_path / "bad.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "good.txt").write_text("hello", encoding="utf-8")
    records, failures = main.process_documents(tmp_path)
    assert len(records) == 1 and records[0]["text"] == "hello"
    assert len(failures) == 1


def test_handbook_samples_load_and_are_labeled_as_examples():
    records, failures = main.process_documents(main.DOCUMENTS_DIR)
    assert failures == []
    assert {Path(record["source"]).suffix.lower() for record in records} == {
        ".txt", ".json", ".html", ".csv", ".pdf", ".docx", ".xlsx"
    }
    # Every format should carry the explicit example-policy warning.
    for suffix in {".txt", ".json", ".html", ".csv", ".pdf", ".docx", ".xlsx"}:
        text = "\n".join(r["text"] for r in records if Path(r["source"]).suffix.lower() == suffix)
        assert "Example polic" in text or "fictional examples" in text
    all_text = "\n".join(r["text"] for r in records)
    for section in ("Attendance", "Examinations", "Grading", "Student conduct",
                    "ID cards", "Library rules", "Fees", "Leave", "Campus safety"):
        assert section.lower() in all_text.lower()
