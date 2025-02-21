from textarena.wrappers.RenderWrappers.PrettyRenderWrapper.base import BaseRenderer
import shutil
from pathlib import Path

class TicTacToeRenderer(BaseRenderer):
    """TicTacToe-specific browser renderer"""

    def __init__(self, env, player_names=None, port=8000, host="127.0.0.1"):
        super().__init__(env, player_names, port, host)

    def get_state(self) -> dict:
        """Get TicTacToe-specific state"""
        try:
            board = self.env.board
            return {
                "board": board or [],
                "current_player": self.env.state.current_player_id if self.env.state else "Unknown",
                "move_history": self.env.move_history,
            }
        except Exception as e:
            print(f"Error getting state: {e}")
            return {
                "board": [],
                "current_player": "Unknown",
                "move_history": [],
            }

    def get_custom_js(self) -> str:
        """Get TicTacToe-specific JavaScript code"""
        return """
        console.log('Loading TicTacToe components...');

        const TicTacToeBoard = ({ board }) => {
            return (
                <div className="board-container">
                    <table className="tic-tac-toe-board">
                        <tbody>
                            {board.map((row, rowIndex) => (
                                <tr key={rowIndex}>
                                    {row.map((cell, colIndex) => (
                                        <td
                                            key={colIndex}
                                            className={`cell ${
                                                (rowIndex + colIndex) % 2 === 0 ? 'light' : 'dark'
                                            }`}
                                        >
                                            <div className="cell-content">{cell}</div>
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            );
        };

        const MoveHistory = ({ moveHistory, playerNames }) => {
            const moveHistoryRef = React.useRef(null);

            React.useEffect(() => {
                if (moveHistoryRef.current) {
                    moveHistoryRef.current.scrollTop = moveHistoryRef.current.scrollHeight;
                }
            }, [moveHistory]);

            return (
                <div>
                    <h3>Move History</h3>
                    <div className="move-history" ref={moveHistoryRef}>
                        {moveHistory.map((move, index) => {
                            const player = playerNames[index % 2];
                            return (
                                <div key={index} className="move-entry">
                                    <span className="move-number">{index + 1}. </span>
                                    <span className="move">
                                        {player}: {formatMove(move)}
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                </div>
            );
        };

        const renderTicTacToeGameInfo = (gameState) => {
            const currentPlayerName = gameState.player_names[gameState.current_player];

            return (
                <div>
                    <h3>Current Turn: {currentPlayerName}</h3>
                    <h3>Players</h3>
                    <div className="players">
                        {Object.entries(gameState.player_names).map(([id, name]) => (
                            <div key={id} className={id === '0' ? 'O-player' : 'X-player'}>
                                {name} ({id === '0' ? 'O' : 'X'})
                            </div>
                        ))}
                    </div>

                    <MoveHistory
                        moveHistory={gameState.move_history}
                        playerNames={gameState.player_names}
                    />
                </div>
            );
        };

        const formatMove = (move) => {
            if (!move) return '';
            const [row, col] = move;
            return `Row ${row}, Col ${col}`;
        };

        const TicTacToeGame = () => {
            console.log('Initializing TicTacToeGame');
            const [gameState, setGameState] = React.useState(null);

            React.useEffect(() => {
                const ws = new WebSocket(`ws://${window.location.host}/ws`);
                ws.onopen = () => console.log('WebSocket connected');
                ws.onmessage = (event) => {
                    try {
                        const state = JSON.parse(event.data);
                        console.log('Received state:', state);
                        setGameState(state);
                    } catch (err) {
                        console.error('Error parsing WebSocket message:', err);
                    }
                };
                ws.onclose = () => console.log('WebSocket connection closed');
                return () => ws.close();
            }, []);

            if (!gameState) {
                console.log('Waiting for game state...');
                return <div>Loading game state...</div>;
            }

            return (
                <BaseGameContainer className="TicTacToe-layout" gameState={gameState} renderGameInfo={renderTicTacToeGameInfo}>
                    <div className="main-content">
                        <TicTacToeBoard board={gameState.board} />
                    </div>
                </BaseGameContainer>
            );
        };

        console.log('Initializing React app');
        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<TicTacToeGame />);
        console.log('TicTacToe components loaded');
        """

    def get_custom_css(self) -> str:
        """Get TicTacToe-specific CSS"""
        return """
        .main-content {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }

        .board-container {
            flex: 0 0 400px;
            display: flex;
            justify-content: center;
            align-items: center;
            margin: auto;
            overflow: hidden;
        }

        .tic-tac-toe-board {
            border-collapse: collapse;
            width: 400px;
            height: 400px;
            table-layout: fixed;
        }

        .tic-tac-toe-board .cell {
            border: 2px solid #000;
            position: relative;
            text-align: center;
            vertical-align: middle;
        }

        .cell.light {
            background-color: #d7b899;
        }

        .cell.dark {
            background-color: #a97d55;
        }

        .cell::before {
            content: "";
            display: block;
            padding-top: 100%;
        }

        .cell-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 2rem;
            font-weight: bold;
            color: #333;
        }

        .move-history {
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid #ccc;
            padding: 10px;
            background-color: #222d37;
            color: #fff;
        }

        .move-entry {
            margin-bottom: 5px;
        }

        .move-history::-webkit-scrollbar {
            width: 8px;
        }

        .move-history::-webkit-scrollbar-thumb {
            background-color: #aaa;
            border-radius: 4px;
        }

        .move-history::-webkit-scrollbar-track {
            background: #f0f0f0;
        }
        """
