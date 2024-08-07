# Pipe Mania Solver

This project, part of the Artificial Intelligence course, aims to utilize AI search techniques to find solutions for the Pipe Mania puzzle, where the goal is to connect pipe pieces on a grid to form a continuous pipeline.

## Provided Material
Implemented:
- search.py: Necessary search algorithms.
- visualizer.py and images/: Puzzle display.
- utils.py: Utility functions.
- tests/: Test cases.

Further Development Needed:
- pipe.py: Basic implementation and class skeletons.

## Input and Output Format
The input grid (*n x n* matrix) consists of pipe pieces represented as two-letter strings separated by tab characters:
- End pieces: FE, FD, FC, FB
- Junction pieces: BE, BD, BC, BB
- Corner pieces: VE, VD, VC, VB
- Straight pieces: LH, LV
> You can view each piece in the images folder.

Input example:
```
VC	BE	VD
BB	VE	LH
VE	LV	VB
```

Output example:
```
VB	BB	VE
BD	VC	LV
VD	LH	VC
```

## Implementation
- pipe.py: The main program capable of solving Pipe Mania puzzles. DFS is used to find the solution.
  - PipeManiaState: Represents the state of the puzzle.
  - Board: Internal representation of the grid with methods to manipulate and query pipe pieces.
  - PipeMania: Implements methods for actions, result, goal testing, and heuristic.

- runtests.sh: Script to run test cases from the tests folder.

## Usage
To run the main program or the visualizer:
```bash
python3 file.py < input.txt
```
> Replace file.py with pipe.py or visualizer.py based on your needs. Depending on your system configuration, you might need to use python or py instead of python3.
