
from pathlib import Path


def write_file(file_path: str, content: str) -> None:
    path = Path(file_path)

    path.write_text(content, encoding="utf-8")
    print("File written successfully!")


def read_file(file_path: str) -> str:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return path.read_text(encoding="utf-8")


def append_file(file_path: str, content: str) -> None:
    path = Path(file_path)

    with path.open("a", encoding="utf-8") as file:
        file.write(content)

    print("Content appended successfully!")


def main():
    file_path = "notes.txt"

    # 1. Write content
    write_file(file_path, "Hello! I am learning Python.\n")

    # 2. Read content
    print("\nReading file:")
    print(read_file(file_path))

    # 3. Append new content
    append_file(file_path, "I am learning RAG fundamentals.\n")

    # 4. Read the updated content
    print("\nUpdated file content:")
    print(read_file(file_path))


if __name__ == "__main__":
    main()
