import re
from typing import Optional, Dict, Tuple, List, Any, Set
import textarena as ta

class QuantumTicTacToeEnv(ta.Env):
    """ Quantum Tic Tac Toe: Players place superpositions of moves, resolved when cycles form. """
    def __init__(self):
        super().__init__()
        self.cell_mapping = {i * 3 + j: (i, j) for i in range(3) for j in range(3)}
        self.move_count = 0

    def reset(self, num_players: int, seed: Optional[int] = None):
        self.state = ta.State(num_players=2, min_players=2, max_players=2, max_turns=25)
        self.state.reset(
            game_state={
                "board": [['' for _ in range(3)] for _ in range(3)],
                "superpositions": {},
                "move_log": []
            },
            player_prompt_function=self._generate_player_prompt
        )
        self.move_count = 0
        self._observer_current_state()

    def _render_board(self):
        board = self.state.game_state["board"]
        superpositions = self.state.game_state["superpositions"]

        # Build a dictionary from cell -> list of marks
        cell_marks = {(r, c): [] for r in range(3) for c in range(3)}
        for move_id, (pid, a, b) in superpositions.items():
            mark = f"{move_id}{'A' if pid == 0 else 'B'}"
            cell_marks[a].append(mark)
            cell_marks[b].append(mark)

        def render_cell(r, c):
            cell_idx = f"[{r * 3 + c}]"
            marks = cell_marks[(r, c)]
            solid = board[r][c]

            if solid:
                return [cell_idx, "", f"  {solid}"]
            elif marks:
                lines = [cell_idx]
                if len(marks) <= 2:
                    lines += [" / ".join(marks), ""]
                else:
                    lines += [" / ".join(marks[:2]), " / ".join(marks[2:])]
                return lines
            else:
                return [cell_idx, "", ""]

        # Combine lines
        rendered_rows = []
        for r in range(3):
            row_lines = [""] * 3  # three lines per cell
            for c in range(3):
                cell = render_cell(r, c)
                for i in range(3):
                    row_lines[i] += f"{cell[i]:^10}"
            rendered_rows.append("\n".join(row_lines))

        return "\n" + "\n" + "\n---+----------+----------+---\n".join(rendered_rows) + "\n"


    def _observer_current_state(self):
        board_str = self._render_board()
        message = f"Quantum Tic Tac Toe Board:\n\n{board_str}\n\nSubmit your move as '[a,b]' to place a quantum mark in two locations."
        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message, for_logging=False)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[str, Any]) -> str:
        symbol = 'X' if player_id == 1 else 'O'
        subscript_rule = "odd" if player_id == 1 else "even"

        return (
            f"You are Player {player_id} in Quantum Tic Tac Toe.\n"
            f"Your symbol is '{symbol}', and your move numbers are always {subscript_rule} (e.g., X1, X3 or O2, O4).\n\n"

            "Goal: Win by forming a line of three classical marks (solidified from superpositions).\n\n"

            "How to Play:\n"
            "- On each turn, place a spooky mark in two different empty squares using the format '[a,b]'.\n"
            "- These marks are entangled, and labeled like 'X1 / X1' or 'O4 / O4'.\n"
            "- You cannot place spooky marks in a square that has already collapsed (solidified).\n\n"

            "Collapse Rule:\n"
            "- If your move creates a cycle in the entanglement graph, it collapses automatically.\n"
            "- Each spooky mark in the cycle turns into a classical mark in one of its two positions.\n"
            "- Any dependent spooky marks also collapse.\n\n"

            "Victory:\n"
            "- The game ends when a player has three classical marks in a row.\n"
            "- If both players get a line during the same collapse, the one with the lower max move number wins.\n\n"

            "Example move: '[0,4]' places a spooky mark in cells 0 and 4."
        )

    def step(self, action: str) -> Tuple[bool, ta.Info]:
        player_id = self.state.current_player_id
        symbol = 'X' if player_id == 1 else 'O'
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)

        # match = re.match(r"\[(\d+),(\d+)\]", action.replace(" ", ""))
        match = re.search(r"\[(\d+),(\d+)\]", action.replace(" ", ""))
        if not match:
            self.state.set_invalid_move(player_id, reason="Invalid format. Use '[a,b]'.")
        else:
            a, b = int(match.group(1)), int(match.group(2))
            if a == b or a not in self.cell_mapping or b not in self.cell_mapping:
                self.state.set_invalid_move(player_id=player_id, reason="Invalid or duplicate cell indices.")
            else:
                pos_a, pos_b = self.cell_mapping[a], self.cell_mapping[b]
                board = self.state.game_state["board"]
                if board[pos_a[0]][pos_a[1]] or board[pos_b[0]][pos_b[1]]:
                    self.state.set_invalid_move(player_id=player_id, reason="One of the cells is already solidified.")
                else:
                    move_id = self.move_count
                    self.state.game_state["superpositions"][move_id] = (player_id, pos_a, pos_b)
                    self.state.game_state["move_log"].append((move_id, player_id, pos_a, pos_b))
                    self.move_count += 1

                    self._resolve_cycles()

        self._observer_current_state()
        return self.state.step()

    def _resolve_cycles(self):
        superpositions = self.state.game_state["superpositions"]
        graph = {}
        for move_id, (_, a, b) in superpositions.items():
            graph.setdefault(a, []).append((b, move_id))
            graph.setdefault(b, []).append((a, move_id))

        visited = set()
        def dfs(node, path, seen_ids):
            if node in path:
                cycle = path[path.index(node):]
                involved_ids = seen_ids[path.index(node):]
                return cycle, involved_ids
            if node not in graph:
                return None
            for neighbor, move_id in graph[node]:
                if move_id in seen_ids:
                    continue
                result = dfs(neighbor, path + [node], seen_ids + [move_id])
                if result:
                    return result
            return None

        for start in graph:
            result = dfs(start, [], [])
            if result:
                cycle, involved_ids = result
                self._collapse_superpositions(involved_ids)
                break

    def _collapse_superpositions(self, move_ids: List[int]):
        board = self.state.game_state["board"]
        superpositions = self.state.game_state["superpositions"]
        for move_id in move_ids:
            if move_id not in superpositions:
                continue
            player_id, a, b = superpositions.pop(move_id)
            symbol = 'X' if player_id == 1 else 'O'
            for r, c in [a, b]:
                if board[r][c] == '':
                    board[r][c] = symbol
                    break  # collapse to the first available cell

        # Check for a win
        for pid in range(2):
            symbol = 'X' if pid == 1 else 'O'
            if self._check_winner(symbol):
                self.state.set_winners(player_ids=[pid], reason=f"Player {pid} wins with solidified marks!")

    def _check_winner(self, symbol: str) -> bool:
        board = self.state.game_state["board"]
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] == symbol:
                return True
            if board[0][i] == board[1][i] == board[2][i] == symbol:
                return True
        if board[0][0] == board[1][1] == board[2][2] == symbol:
            return True
        if board[0][2] == board[1][1] == board[2][0] == symbol:
            return True
        return False
