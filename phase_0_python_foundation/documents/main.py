from pathlib import Path
import json


def load_document(file_path: str) -> dict:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    extension = path.suffix.lower()

    if extension == ".txt":
        text = path.read_text(encoding="utf-8")

    elif extension == ".json":
        content = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps(content, indent=2)

    else:
        raise ValueError(f"Unsupported file format: {extension}")

    return {
        "source": path.name,
        "file_type": extension,
        "text": text,
        "character_count": len(text)
    }


def main():
    base_dir = Path(__file__).parent

    files = [
        base_dir / "rules.txt",
        base_dir / "rules.json"
    ]

    for file_path in files:
        try:
            document = load_document(str(file_path))

            print("\nSource:", document["source"])
            print("Type:", document["file_type"])
            print("Characters:", document["character_count"])
            print("Content:\n", document["text"])

        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            print("Error:", error)


if __name__ == "__main__":
    main()
