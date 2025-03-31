# Snake AI - Deep Reinforcement Learning Project

A Snake game implementation with an AI agent trained using Deep Reinforcement Learning. The project includes both a playable version and an AI training system.

## Project Structure

```
SnakeAI/
├── src/
│   ├── environment.py   # Game environment implementation
│   ├── main.py         # AI training loop
│   └── snakeHuman.py   # Playable version of the game
├── requirements.txt    # Project dependencies
└── README.md          # This file
```

## Prerequisites

- Python 3.8 or higher
- PyTorch
- Pygame
- NumPy

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the AI

To train the AI agent from scratch:
```bash
python src/main.py
```

### Playing the Game

To play the game manually:
```bash
python src/snakeHuman.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the classic Snake game
- Implemented for the INF1600 course at NTNU
- Inspired by DeepMind's DQN paper 