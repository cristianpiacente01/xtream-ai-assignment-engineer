"""
Wrapper script designed to start the FastAPI application.
It accepts an optional custom model path via command line argument.

Cristian Piacente
"""

import argparse
import subprocess


def main():
    # Initialize argparser
    parser = argparse.ArgumentParser(description="Start the FastAPI application.")
    parser.add_argument("--model-path", help="Custom path to the model file.", required=False)
    args = parser.parse_args()

    # Start the application, eventually passing a custom model path
    if args.model_path is None:
        # No custom path was passed
        subprocess.run(["python", "-m", "app.main"], check=True)
    else:
        # Start the application and pass the custom model path
        subprocess.run(["python", "-m", "app.main", args.model_path], check=True)



if __name__ == "__main__":
    main()