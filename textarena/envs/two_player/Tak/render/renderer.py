from textarena.wrappers.RenderWrappers.OfflineBrowserWrapper.base import BaseRenderer
import shutil
from pathlib import Path

class TakRenderer(BaseRenderer):
    """Tak-specific browser renderer"""

    def __init__(self, env, player_names=None, port=8000, host="127.0.0.1"):
        super().__init__(env, player_names, port, host)

    def get_state(self) -> dict:
        """Get Tak-specific state"""
        try:
            board = self.env.board
            return {
                "board": board or [],  # Assuming board is a nested list
                "current_player": f"Player {self.env.state.current_player_id}" if self.env.state else "Unknown",
                "move_stack": self.env.state.game_state.get("rendered_board", "") if self.env.state else "",
                "player_pieces": self.env.players or {},
            }
        except Exception as e:
            print(f"Error getting state: {e}")
            return {
                "board": [],
                "current_player": "Unknown",
                "move_stack": [],
                "player_pieces": {},
            }

    def get_custom_js(self) -> str:
        """Get Tak-specific JavaScript code"""
        return """
        console.log('Loading Tak components...');

        const TakBoard = ({ board }) => {
            return (
                <div className="board-container">
                    <table className="tak-board">
                        <tbody>
                            {board.map((row, rowIndex) => (
                                <tr key={rowIndex}>
                                    {row.map((cell, cellIndex) => (
                                        <td
                                            key={cellIndex}
                                            className={`cell ${((rowIndex + cellIndex) % 2 === 0) ? "light" : "dark"}`}
                                        >
                                            <div className="cell-content">
                                                {Array.isArray(cell) && cell.length > 0 ? (
                                                    <div className="cell-list">
                                                        {/* Render the items in the original order */}
                                                        {cell.map((item, itemIndex) => (
                                                            <span key={itemIndex} className="cell-item">
                                                                {item}
                                                            </span>
                                                        ))}
                                                    </div>
                                                ) : (
                                                    <span className="cell-item">
                                                        {cell !== null ? cell : ""}
                                                    </span>
                                                )}
                                            </div>
                                        </td>
                                    ))}
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            );
        };

        function GameInfo({ gameState }) {
            return (
                <div className="info-container">
                    <h2>Game Status</h2>
                    <div className="status">
                        <div>Current Turn: {gameState.current_player}</div>
                    </div>
                    <h3>Player Pieces</h3>
                    <div className="players">
                        {Object.entries(gameState.player_pieces).map(([id, pieces]) => (
                            <div key={id}>
                                Player {id}: Stones - {pieces.stones}, Capstones - {pieces.capstones}
                            </div>
                        ))}
                    </div>
                </div>
            );
        }

        const ChatHistory = ({ gameState }) => {
            const messagesEndRef = React.useRef(null);
            
            React.useEffect(() => {
                messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
            }, [gameState.chat_history]);

            return (
                <div className="chat-container">
                    <h2>Game Chat</h2>
                    <div className="chat-messages">
                        {gameState.chat_history.map((msg, i) => (
                            <div key={i} className={`chat-message ${msg.player_id === 0 ? 'white' : 'black'}`}>
                            
                                <div className="player-name">
                                    {gameState.player_names[msg.player_id]}:
                                </div>
                                <div>{msg.message}</div>
                            </div>
                        ))}
                        <div ref={messagesEndRef} />
                    </div>
                </div>
            );
        };

        const TakGame = () => {
            console.log('Initializing TakGame');
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
                <BaseGameContainer className="tak-layout" gameState={gameState}>
                    <div className="game-header">
                        Tak Game
                    </div>
                    <div className="main-content">
                        <TakBoard board={gameState.board} />
                        <GameInfo gameState={gameState} />
                    </div>
                    <ChatHistory gameState={gameState} />
                </BaseGameContainer>
            );
        };

        // Initialize the app
        console.log('Initializing React app');
        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<TakGame />);
        console.log('Tak components loaded');
        """

    def get_custom_css(self) -> str:
        """Get custom CSS styles for Tak game"""
        return """
        .main-content {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }

        .game-header {
            font-size: 30px;
            text-align: center;
            margin: 0 0 10px 0;
            padding: 0;
            letter-spacing: 2px;
            text-transform: uppercase;
            background: linear-gradient(45deg, #4CAF50, #2196F3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }


        .board-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 20px auto;
            background-color: #2b2b2b;
            padding: 15px;
            border-radius: 8px;
            border: 2px solid #444;
            max-width: 90%;
        }

        .tak-board {
            border-collapse: collapse;
            table-layout: fixed; /* Consistent column widths */
            width: auto;
            aspect-ratio: 1 / 1; /* Makes the grid perfectly square */
        }

        .tak-board td {
            width: 100px; /* Fixed cell width */
            height: 100px; /* Fixed cell height */
            aspect-ratio: 1 / 1; /* Ensures each cell remains square */
            text-align: center;
            vertical-align: middle;
            border: 1px solid #555;
            padding: 0;
            position: relative;
            overflow: hidden;
        }

        .tak-board .light {
            background-color: #d7b899; /* Lighter beige for better contrast */
        }

        .tak-board .dark {
            background-color: #a97d55; /* Darker brown for better contrast */
        }

        .tak-board .cell-content {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100%;
            overflow: hidden;
        }

        .tak-board .cell-list {
            display: flex; /* Arrange items horizontally */
            flex-wrap: nowrap; /* Prevent wrapping */
            gap: 2px; /* Spacing between items */
            justify-content: flex-end; /* Align items to the right */
        }

        .tak-board .cell-item {
            font-size: 0.7rem; /* Smaller font size for more content */
            color: #ffffff; /* White text for better legibility */
            padding: 0 2px;
            background: rgba(0, 0, 0, 0.3); /* Semi-transparent background for better visibility */
            border-radius: 3px; /* Slight rounding for aesthetics */
        }

        .info-container {
            background: linear-gradient(145deg, #2c2f33, #23272a);
            padding: 20px;
            border-radius: 15px;
            color: #ffffff;
            border: 1px solid #4b4b4b;
            max-width: 90%;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
            margin: 0 auto; /* Centers the box */
        }

        .info-container h2 {
            margin-top: 0;
            margin-bottom: 15px;
            text-align: center;
            font-size: 1.8rem;
            color: #ffffff;
            border-bottom: 1px solid #4b4b4b;
            padding-bottom: 10px;
            letter-spacing: 1px;
            text-transform: uppercase;
        }

        .info-container .status,
        .info-container .players {
            margin-bottom: 20px;
            font-size: 1.2rem;
            line-height: 1.6;
        }

        .info-container .status div,
        .info-container .players div {
            margin: 5px 0;
            color: #dcdcdc;
        }


        .status {
            margin-bottom: 20px;
        }

        .players {
            margin-bottom: 20px;
        }

        /* General Layout */
        .chat-container {
            flex: 1;
            background: linear-gradient(145deg, #2c2f33, #23272a);
            padding: 20px;
            border-radius: 15px;
            color: #ffffff;
            border: 1px solid #4b4b4b;
            max-height: 500px;
            overflow-y: auto; /* Enable scrolling */
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
        }

        /* Chat Title */
        .chat-container h2 {
            margin: 0;
            font-size: 1.8rem;
            color: #ffffff;
            text-align: center;
            border-bottom: 1px solid #4b4b4b;
            padding-bottom: 10px;
            margin-bottom: 15px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }

        /* Chat Messages */
        .chat-messages {
            display: flex;
            flex-direction: column;
            gap: 15px;
            padding-bottom: 20px;
        }

        /* Individual Chat Bubble */
        .chat-message {
            padding: 15px 20px;
            border-radius: 20px;
            position: relative;
            max-width: 75%;
            word-wrap: break-word;
            font-size: 1rem;
            line-height: 1.5;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease-in-out;
        }

        /* Player 0's Chat Bubble */
        .chat-message.white {
            background: #f4f4f4;
            color: #1e1e1e;
            align-self: flex-start;
            border: 1px solid #ddd;
            animation: fadeInLeft 0.4s ease-out;
        }

        .chat-message.white::after {
            content: '';
            position: absolute;
            left: -10px;
            top: 50%;
            transform: translateY(-50%);
            border-width: 10px 10px 10px 0;
            border-style: solid;
            border-color: transparent #f4f4f4 transparent transparent;
        }

        /* Opponent's Chat Bubble */
        .chat-message.black {
            background: #2b2f33;
            color: #ffffff;
            align-self: flex-end;
            border: 1px solid #555;
            animation: fadeInRight 0.4s ease-out;
        }

        .chat-message.black::after {
            content: '';
            position: absolute;
            right: -10px;
            top: 50%;
            transform: translateY(-50%);
            border-width: 10px 0 10px 10px;
            border-style: solid;
            border-color: transparent transparent transparent #2b2f33;
        }

        /* Smooth Animation */
        @keyframes fadeInLeft {
            from {
                opacity: 0;
                transform: translateX(-20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        @keyframes fadeInRight {
            from {
                opacity: 0;
                transform: translateX(20px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }

        /* Scrollbar Styling */
        .chat-container::-webkit-scrollbar {
            width: 8px;
        }

        .chat-container::-webkit-scrollbar-track {
            background: #1e1e1e;
            border-radius: 10px;
        }

        .chat-container::-webkit-scrollbar-thumb {
            background: #555;
            border-radius: 10px;
        }

        .chat-container::-webkit-scrollbar-thumb:hover {
            background: #888;
        }

        /* Typography Improvements */
        .chat-container h2,
        .chat-message {
            font-family: 'Roboto', sans-serif;
            font-weight: 400;
        }

        /* Subtle Hover Effect */
        .chat-message:hover {
            transform: scale(1.02);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
        }


        """
