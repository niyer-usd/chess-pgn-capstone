# Chess Score Sheet Error Correction Using AI-Assisted PGN Reconstruction

**AAI-590 Capstone Project — University of San Diego, MSAAI**
**Student:** Narendra Iyer (niyer@sandiego.edu)
**Project Duration:** June 23 – August 11, 2025

## Project Overview

This project builds a five-layer hybrid AI pipeline that detects and corrects transcription errors in chess score sheets converted to PGN (Portable Game Notation) format via OCR. The system combines rule-based chess validation, two trained neural network components, and Stockfish engine beam search to identify and fix four realistic error types.

## Repository Structure

| File | Description |
|---|---|
| `00_environment_test.ipynb` | Verifies all required Python libraries are installed in the SageMaker environment |
| `01_download_lichess.ipynb` | Downloads 121,332 games from Lichess Open Database, filters to 97,160 games, applies synthetic corruption pipeline across four error types, and uploads dataset to Amazon S3 |
| `02_EDA.ipynb` | Exploratory data analysis — six analyses covering class balance, game length/Elo distributions, error injection point distribution, Type 4 cascade length analysis, total moves distribution, and relative injection position uniformity |
| `corruption_metadata.csv` | Original metadata CSV — 97,160 rows, 7 columns, one row per game file |
| `corruption_metadata_v2.csv` | Corrected metadata CSV — 97,160 rows, 8 columns. Fixes: corrupted_move set to NaN for deletion error types (3 and 4); new shift_starts_at column added for Type 4 errors |
| `.gitignore` | Excludes large PGN game folders (hosted on Amazon S3) and system files |

## Dataset

The full dataset is hosted publicly on Amazon S3:

- **Raw source games:** https://chess-pgn-capstone-387100404445-us-east-1-an.s3.us-east-1.amazonaws.com/raw-pgn/
- **Corrupted games:** https://chess-pgn-capstone-387100404445-us-east-1-an.s3.us-east-1.amazonaws.com/corrupted-pgn/
- **Corrected metadata:** https://chess-pgn-capstone-387100404445-us-east-1-an.s3.us-east-1.amazonaws.com/corruption_metadata_v2.csv
- **Original Lichess source:** https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst

## Four Error Types

| Type | Description |
|---|---|
| Type 1 | Illegal move substitution — a syntactically valid but illegal move replaces the correct move |
| Type 2 | OCR character substitution — a character within the move notation is misread (e.g. B→8, Q→O) |
| Type 3 | Move deletion without column shift — a move is removed but the player noticed and continued correctly |
| Type 4 | Move deletion with column shift cascade — a move is removed and all subsequent moves are attributed to the wrong player |

## Five-Layer Pipeline Architecture

| Layer | Component | Description |
|---|---|---|
| 1 | python-chess | Rule-based move legality checker |
| 2 | Bidirectional LSTM | Sequence classifier — predicts error position and type |
| 3 | Stockfish | Beam search candidate correction generator |
| 4 | BERT-style transformer | Bidirectional consistency scorer via HuggingFace fine-tuning |
| 5 | Ranking | Combines Stockfish evaluation and consistency scores |

## Tools and Infrastructure

- **Language:** Python 3.12
- **Libraries:** python-chess, pandas, numpy, matplotlib, seaborn, PyTorch, HuggingFace Transformers
- **Cloud:** AWS SageMaker Studio (ml.t3.medium development, ml.g4dn.xlarge Spot for training)
- **Storage:** Amazon S3 (us-east-1)
- **Dataset source:** Lichess Open Database (CC0 license)
