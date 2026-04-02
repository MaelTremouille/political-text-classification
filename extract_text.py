import zipfile
from pathlib import Path

TEXT_FOLDER = Path("text_files")
YEARS = ['1973', '1978', '1981', '1988', '1993']
ELECTIONS = ['legislatives', 'presidentielle']

# Extract all zip archives and print statistics
total_files = 0
for year in YEARS:
    for e_type in ELECTIONS:
        zip_path = TEXT_FOLDER / year / f"{e_type}.zip"
        if not zip_path.exists():
            continue
        output_dir = TEXT_FOLDER / year / e_type
        output_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            txt_members = [m for m in zf.namelist() if m.endswith(".txt")]
            zf.extractall(TEXT_FOLDER / year)
            count = len(txt_members)
            total_files += count
            print(f"{year}/{e_type}: {count} text files extracted")

print(f"\nTotal text files: {total_files}")

# Read all extracted text files
documents = {}
for year in YEARS:
    for e_type in ELECTIONS:
        folder = TEXT_FOLDER / year / e_type
        if not folder.exists():
            continue
        for txt_file in sorted(folder.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8")
            if text.strip():
                documents[(year, e_type, txt_file.stem)] = text

print(f"Loaded {len(documents)} non-empty documents")
