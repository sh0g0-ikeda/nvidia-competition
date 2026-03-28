from __future__ import annotations

import argparse

from baseline_domain import PackageConfig
from baseline_services import AdapterPackager


def parse_args() -> PackageConfig:
    parser = argparse.ArgumentParser(description="Package a LoRA adapter directory into submission.zip.")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--zip-path", default="submission.zip")
    args = parser.parse_args()
    return PackageConfig(adapter_dir=args.adapter_dir, zip_path=args.zip_path)


def main() -> None:
    config = parse_args()
    file_names = AdapterPackager().package(config)
    print(f"Created: {config.output_path}")
    print("Contents:")
    for name in file_names:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
