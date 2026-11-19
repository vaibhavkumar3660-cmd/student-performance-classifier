from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/grugg1233/StudentPerformanceFactors/main/StudentPerformanceFactors.csv"
DEST = Path(__file__).resolve().parent / "data" / "StudentPerformanceFactors.csv"

def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the Student Performance Factors dataset...")
    with urlopen(URL, timeout=60) as response:
        data = response.read()
    DEST.write_bytes(data)
    print(f"Saved {len(data):,} bytes to: {DEST}")
    print("Dataset download complete.")

if __name__ == "__main__":
    main()
