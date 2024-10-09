"""
Car Puzzle Game

In this game, there are two players: the Puzzle Creator (Player 0) and the Puzzle Solver (Player 1).

**Gameplay:**

- **Phase 1 (Puzzle Creation):**

    - The Puzzle Creator designs a car puzzle by specifying the grid size, placing cars, and providing a solution.
    - The puzzle involves moving cars on a grid to allow the target car (ID 0) to reach the exit.
    - Cars can only move in the direction they are oriented (horizontal or vertical).

- **Phase 2 (Puzzle Solving):**

    - The Puzzle Solver attempts to solve the puzzle by moving the cars to allow the target car to reach the exit.
    - The Puzzle Solver provides a sequence of moves as their solution.

**Key Rules:**

- Cars must be at least two units long and placed in straight lines either horizontally (H) or vertically (V).
- The Puzzle Creator must provide a valid puzzle and a valid solution.
- If the Puzzle Creator provides an invalid puzzle or solution, they lose.
- The Puzzle Solver must provide a valid sequence of moves that solves the puzzle.
- If the Puzzle Solver successfully solves the puzzle, they win; otherwise, they lose.

**Parameters:**

- The game does not have adjustable parameters.

**Game Outcomes:**

- **Puzzle Creator wins**: If the Puzzle Solver fails to solve the puzzle.
- **Puzzle Solver wins**: If they successfully solve the puzzle.
- **Puzzle Creator loses**: If they provide an invalid puzzle or solution.
"""

import random
from typing import Any, Dict, Optional, Tuple

import textarena as ta


class CarPuzzleEnv(ta.Env):
    def __init__(
        self,
    ):
        """
        Initialize the Car Puzzle game.

        Roles:
            - Player 0 is the Puzzle Creator
            - Player 1 is the Puzzle Solver
        """
        self.environment_name = "Car Puzzle"

        # Initialize game state
        self.game_state = ta.State(
            {
                "phase": 1,  # 1: Puzzle Creation, 2: Puzzle Solving
                "game_over": False,
                "board": None,
                "exit": None,
                "cars": {},
                "logs": [],
                "render": ["phase", "game_over"],
            }
        )

    def reset(
        self, seed: Optional[int] = None
    ) -> Tuple[Optional[ta.Observation], ta.Reward]:
        """
        Reset the game to its initial state.

        Returns:
            Tuple[Dict[int, str], Dict[int, Any]]: Initial prompts for both players and additional info.
        """
        if seed is not None:
            random.seed(seed)
        else:
            random.seed()

        self.game_state["phase"] = 1
        self.game_state["game_over"] = False
        self.game_state["board"] = None
        self.game_state["exit"] = None
        self.game_state["cars"] = {}
        self.game_state["logs"] = []

        # Generate initial prompts for both players
        observations = {
            0: [
                self._generate_puzzle_creator_prompt(0)
            ],  # Player 0 is the Puzzle Creator
            1: [
                self._generate_puzzle_solver_prompt(1)
            ],  # Player 1 is the Puzzle Solver
        }

        info = {}

        return observations, info

    def _generate_puzzle_creator_prompt(self, id) -> ta.Message:
        """
        Generate the prompt for the Puzzle Creator.

        Returns:
            str: Initial prompt for the Puzzle Creator.
        """
        prompt = (
            f"You are Player {id}, the Puzzle Creator in the 'Car Puzzle' game.\n"
            "Your task is to create a car puzzle by specifying the grid size, placing cars, and providing a solution.\n"
            "There is a specific target car with ID 0 that needs to reach the exit.\n"
            "Cars are numbered uniquely and can only move in the direction they are oriented (forward and backward).\n"
            "Each car must be at least two squares long and be placed in a straight line (horizontal or vertical).\n"
            "Format your submission using the following formalized language (use square brackets for keywords):\n"
            "\n"
            "1. Define Grid Size:\n"
            "[GridSize] width height\n"
            "Example: [GridSize] 6 6\n"
            "\n"
            "2. Place Cars:\n"
            "[PlaceCar] car_id orientation x y length\n"
            " - car_id: Integer identifier for the car (0 is the target car).\n"
            " - orientation: H for horizontal, V for vertical.\n"
            " - x y: Coordinates of the starting position (top-left for H, top for V).\n"
            " - length: Length of the car (minimum 2).\n"
            "Example: [PlaceCar] 0 H 2 2 2\n"
            "\n"
            "3. Specify Exit Point (optional, default is right edge on the same row as the target car):\n"
            "[Exit] x y\n"
            "Example: [Exit] 5 2\n"
            "\n"
            "4. Provide Solution Steps:\n"
            "List of moves using the following format:\n"
            "[Move] car_id steps\n"
            " - car_id: Integer identifier of the car to move.\n"
            " - steps: Positive or negative integer indicating the number of steps to move (positive for forward, negative for backward).\n"
            "Example: [Move] 0 1\n"
            "\n"
            "Submit your puzzle and solution together.\n"
            "Full Example:\n"
            "[GridSize] 6 6\n"
            "[PlaceCar] 0 H 2 2 2\n"
            "[PlaceCar] 1 V 0 0 3\n"
            "[PlaceCar] 2 V 4 0 2\n"
            "[Exit] 5 2\n"
            "[Move] 1 1\n"
            "[Move] 0 2\n"
        )
        return (id, prompt)

    def _generate_puzzle_solver_prompt(self, id) -> ta.Message:
        """
        Generate the prompt for the Puzzle Solver.

        Returns:
            str: Initial prompt for the Puzzle Solver.
        """
        prompt = (
            f"You are Player {id}, the Puzzle Solver in the 'Car Puzzle' game.\n"
            "You will be given a car puzzle to solve.\n"
            "There is a specific target car with ID 0 that needs to reach the exit.\n"
            "Provide your solution as a list of moves using the formalized language.\n"
            "Format for moves:\n"
            "[Move] car_id steps\n"
            " - car_id: Integer identifier of the car to move.\n"
            " - steps: Positive or negative integer indicating the number of steps to move (positive for forward, negative for backward).\n"
            "Example: [Move] 0 1\n"
            "\n"
            "Wait for the puzzle from the Puzzle Creator."
        )
        return (id, prompt)

    def step(
        self,
        player_id: int,
        action: str,
    ) -> Tuple[
        Optional[ta.Observation],  # observations
        Optional[ta.Reward],  # reward
        bool,  # truncated
        bool,  # terminated
        ta.Info,  # info
    ]:
        """
        Process the player's action.

        Args:
            player_id (int): The player's ID (0 or 1).
            action (str): The player's message or action.

        Returns:
            tuple: (observations, reward, truncated, terminated, info)
        """
        terminated = False
        truncated = False
        reward = None
        info = {}
        observations = {}

        self.game_state["logs"].append((-1, action))

        if self.game_state["game_over"]:
            # Game is already over
            observations = {player_id: None}
            info["reason"] = "Game is already over."
            return observations, reward, truncated, terminated, info

        if self.game_state["phase"] == 1 and player_id == 0:
            # Phase 1: Puzzle Creator provides puzzle and solution
            success, message = self._process_puzzle_creator_submission(action)
            if not success:
                # Puzzle Creator loses
                terminated = True
                reward = {0: -1, 1: 1}
                info["reason"] = message
            else:
                # Proceed to Phase 2
                self.game_state["phase"] = 2
                observations = {1: self._get_puzzle_description()}
                info["info"] = (
                    "Puzzle has been provided. Puzzle Solver, please provide your solution."
                )
        elif self.game_state["phase"] == 2 and player_id == 1:
            # Phase 2: Puzzle Solver provides solution
            success, message = self._process_puzzle_solver_submission(action)
            if success:
                # Puzzle Solver wins
                terminated = True
                reward = {0: -1, 1: 1}
                info["reason"] = "Puzzle Solver successfully solved the puzzle."
            else:
                # Puzzle Solver loses
                terminated = True
                reward = {0: 1, 1: -1}
                info["reason"] = message
        else:
            # Invalid action
            info["reason"] = (
                f"Unexpected action from Player {player_id} in phase {self.game_state['phase']}."
            )
        self.game_state["logs"].append((-1, info["reason"]))

        self.game_state["game_over"] = terminated

        return observations, reward, truncated, terminated, info

    def _process_puzzle_creator_submission(self, submission: str) -> Tuple[bool, str]:
        """
        Process the submission from the Puzzle Creator.

        Args:
            submission (str): The submission string from the Puzzle Creator.

        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            puzzle_data, solution_steps = self._parse_puzzle_creator_submission(
                submission
            )
            success, error_message = self._initialize_board(puzzle_data)
            if not success:
                return False, f"Invalid puzzle: {error_message}"
            # Verify the provided solution
            is_valid_solution = self._verify_puzzle_and_solution(solution_steps)
            if not is_valid_solution:
                return False, "Provided solution does not solve the puzzle."
            return True, "Puzzle and solution accepted."
        except Exception as e:
            return False, f"Error processing submission: {str(e)}"

    def _process_puzzle_solver_submission(self, submission: str) -> Tuple[bool, str]:
        """
        Process the submission from the Puzzle Solver.

        Args:
            submission (str): The submission string from the Puzzle Solver.

        Returns:
            Tuple[bool, str]: (success, message)
        """
        try:
            solution_steps = self._parse_puzzle_solver_submission(submission)
            is_solved = self._apply_solution(solution_steps)
            if is_solved:
                return True, "Puzzle solved successfully."
            else:
                return False, "Solution did not solve the puzzle."
        except Exception as e:
            return False, f"Error processing submission: {str(e)}"

    def _parse_puzzle_creator_submission(self, submission: str) -> Tuple[list, list]:
        """
        Parse the submission from the Puzzle Creator.

        Args:
            submission (str): The submission string.

        Returns:
            Tuple[list, list]: (puzzle_data, solution_steps)
        """
        lines = submission.strip().split("\n")
        puzzle_data = []
        solution_steps = []
        in_solution = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("[Move]"):
                in_solution = True
            if in_solution:
                solution_steps.append(line)
            else:
                puzzle_data.append(line)
        if not solution_steps:
            raise ValueError("No solution steps provided.")
        return puzzle_data, solution_steps

    def _parse_puzzle_solver_submission(self, submission: str) -> list:
        """
        Parse the submission from the Puzzle Solver.

        Args:
            submission (str): The submission string.

        Returns:
            list: solution_steps
        """
        lines = submission.strip().split("\n")
        solution_steps = []
        for line in lines:
            line = line.strip()
            if line.startswith("[Move]"):
                solution_steps.append(line)
        if not solution_steps:
            raise ValueError("No solution steps provided.")
        return solution_steps

    def _initialize_board(self, puzzle_data: list) -> Tuple[bool, str]:
        """
        Initialize the game board based on the puzzle data.

        Args:
            puzzle_data (list): List of strings representing the puzzle data.

        Returns:
            Tuple[bool, str]: (success, error_message)
        """
        self.game_state["cars"] = {}
        self.game_state["board"] = None
        self.game_state["exit"] = None
        for line in puzzle_data:
            tokens = line.strip().split()
            if not tokens:
                continue
            if tokens[0] == "[GridSize]":
                if len(tokens) != 3:
                    return False, "Invalid [GridSize] format."
                width, height = int(tokens[1]), int(tokens[2])
                self.game_state["board"] = [
                    ["." for _ in range(width)] for _ in range(height)
                ]
            elif tokens[0] == "[PlaceCar]":
                if len(tokens) != 6:
                    return False, "Invalid [PlaceCar] format."
                car_id = int(tokens[1])
                orientation = tokens[2]
                x, y = int(tokens[3]), int(tokens[4])
                length = int(tokens[5])
                success, error_message = self._place_car(
                    car_id, orientation, x, y, length
                )
                if not success:
                    return False, error_message
            elif tokens[0] == "[Exit]":
                if len(tokens) != 3:
                    return False, "Invalid [Exit] format."
                x, y = int(tokens[1]), int(tokens[2])
                self.game_state["exit"] = (x, y)
        if self.game_state["board"] is None:
            return False, "Board size was not specified."
        if 0 not in self.game_state["cars"]:
            return False, "Target car with car_id=0 was not placed."
        if self.game_state["exit"] is None:
            # Default exit is the right edge on the same row as the target car
            main_car = self.game_state["cars"].get(0)
            if main_car["orientation"] != "H":
                return False, "Target car must be horizontal for default exit position."
            self.game_state["exit"] = (
                len(self.game_state["board"][0]) - 1,
                main_car["positions"][0][1],
            )

        # Ensure all cars are in straight lines and at least two squares long
        for car_id, car in self.game_state["cars"].items():
            if car["length"] < 2:
                return False, f"Car {car_id} must be at least 2 units long."
            if car["orientation"] not in ("H", "V"):
                return (
                    False,
                    f"Car {car_id} has invalid orientation: {car['orientation']}.",
                )
        return True, "Board initialized successfully."

    def _place_car(
        self, car_id: int, orientation: str, x: int, y: int, length: int
    ) -> Tuple[bool, str]:
        """
        Place a car on the board.

        Args:
            car_id (int): The car's ID.
            orientation (str): 'H' for horizontal or 'V' for vertical.
            x (int): Starting x-coordinate.
            y (int): Starting y-coordinate.
            length (int): Length of the car.

        Returns:
            Tuple[bool, str]: (success, error_message)
        """
        board = self.game_state["board"]
        if length < 2:
            return False, f"Car {car_id} must be at least 2 units long."
        if orientation not in ("H", "V"):
            return False, f"Invalid orientation for car {car_id}: {orientation}."
        if car_id in self.game_state["cars"]:
            return False, f"Car {car_id} already placed."
        positions = []
        for i in range(length):
            xi, yi = x + i if orientation == "H" else x, (
                y if orientation == "H" else y + i
            )
            if not (0 <= xi < len(board[0]) and 0 <= yi < len(board)):
                return False, f"Car {car_id} is out of bounds at ({xi}, {yi})."
            if board[yi][xi] != ".":
                return False, f"Cell ({xi}, {yi}) is already occupied."
            positions.append((xi, yi))
        for xi, yi in positions:
            board[yi][xi] = str(car_id)
        self.game_state["cars"][car_id] = {
            "orientation": orientation,
            "positions": positions,
            "length": length,
        }
        return True, "Car placed successfully."

    def _verify_puzzle_and_solution(self, solution_steps: list) -> bool:
        """
        Verify that the provided solution solves the puzzle.

        Args:
            solution_steps (list): List of solution steps.

        Returns:
            bool: True if the solution solves the puzzle, False otherwise.
        """
        # Copy the board and cars for simulation
        board_copy = [row.copy() for row in self.game_state["board"]]
        cars_copy = {cid: car.copy() for cid, car in self.game_state["cars"].items()}
        for car_id in cars_copy:
            cars_copy[car_id]["positions"] = cars_copy[car_id]["positions"].copy()
        for step in solution_steps:
            if not step.startswith("[Move]"):
                return False
            tokens = step.strip().split()
            if len(tokens) != 3:
                return False
            car_id = int(tokens[1])
            steps = int(tokens[2])
            if car_id not in cars_copy:
                return False
            success = self._move_car(cars_copy, board_copy, car_id, steps)
            if not success:
                return False
            if car_id == 0 and self._check_exit(cars_copy[0]):
                return True
        # After all moves, check if target car has reached the exit
        return self._check_exit(cars_copy[0])

    def _apply_solution(self, solution_steps: list) -> bool:
        """
        Apply the solution steps provided by the Puzzle Solver.

        Args:
            solution_steps (list): List of solution steps.

        Returns:
            bool: True if the solution solves the puzzle, False otherwise.
        """
        # Copy the board and cars for simulation
        board_copy = [row.copy() for row in self.game_state["board"]]
        cars_copy = {cid: car.copy() for cid, car in self.game_state["cars"].items()}
        for car_id in cars_copy:
            cars_copy[car_id]["positions"] = cars_copy[car_id]["positions"].copy()
        for step in solution_steps:
            if not step.startswith("[Move]"):
                return False
            tokens = step.strip().split()
            if len(tokens) != 3:
                return False
            car_id = int(tokens[1])
            steps = int(tokens[2])
            if car_id not in cars_copy:
                return False
            success = self._move_car(cars_copy, board_copy, car_id, steps)
            if not success:
                return False
            if car_id == 0 and self._check_exit(cars_copy[0]):
                return True
        # After all moves, check if target car has reached the exit
        return self._check_exit(cars_copy[0])

    def _move_car(self, cars: dict, board: list, car_id: int, steps: int) -> bool:
        """
        Move a car on the board.

        Args:
            cars (dict): Dictionary of cars.
            board (list): The board representation.
            car_id (int): The ID of the car to move.
            steps (int): Number of steps to move.

        Returns:
            bool: True if the move is successful, False otherwise.
        """
        car = cars[car_id]
        orientation = car["orientation"]
        positions = car["positions"]
        dx, dy = 0, 0
        if orientation == "H":
            dx = steps
        else:
            dy = steps
        new_positions = []
        for xi, yi in positions:
            xi_new, yi_new = xi + dx, yi + dy
            if not (0 <= xi_new < len(board[0]) and 0 <= yi_new < len(board)):
                return False  # Move out of bounds
            # Check for collision
            cell_content = board[yi_new][xi_new]
            if cell_content != "." and cell_content != str(car_id):
                return False  # Collision detected
            new_positions.append((xi_new, yi_new))
        # Update board
        for xi, yi in positions:
            board[yi][xi] = "."
        for xi, yi in new_positions:
            board[yi][xi] = str(car_id)
        # Update car positions
        car["positions"] = new_positions
        return True

    def _check_exit(self, car: dict) -> bool:
        """
        Check if the target car has reached the exit.

        Args:
            car (dict): The target car's data.

        Returns:
            bool: True if the car has reached the exit, False otherwise.
        """
        for xi, yi in car["positions"]:
            if (xi, yi) == self.game_state["exit"]:
                return True
        return False

    def _get_puzzle_description(self) -> str:
        """
        Generate a description of the puzzle for the Puzzle Solver.

        Returns:
            str: The puzzle description.
        """
        description = ""
        description += f"[GridSize] {len(self.game_state['board'][0])} {len(self.game_state['board'])}\n"
        for car_id, car in self.game_state["cars"].items():
            orientation = car["orientation"]
            x, y = car["positions"][0]
            length = car["length"]
            description += f"[PlaceCar] {car_id} {orientation} {x} {y} {length}\n"
        exit_x, exit_y = self.game_state["exit"]
        description += f"[Exit] {exit_x} {exit_y}"
        return description

    def render(self):
        """
        Render the current game state.
        """
        print("Game Logs:")
        for i, log in self.game_state["logs"]:
            if i == -1:
                print(f"[GAME] {log}")
            else:
                print(f"Player {i}: {log}")
        print("\n")
