from textarena.wrappers.RenderWrappers.PrettyRenderWrapper.base import BaseRenderer
import shutil
from pathlib import Path

class ChessRenderer(BaseRenderer):
    """Chess-specific browser renderer"""
    
    def __init__(self, env, player_names=None, port=8000, host="127.0.0.1"):
        super().__init__(env, player_names, port, host)
        self._setup_piece_images()

    def _setup_piece_images(self):
        """Set up chess piece images"""
        pieces_dir = self.static_dir / "pieces"
        pieces_dir.mkdir(exist_ok=True)
        
        # Copy piece images from assets
        assets_dir = Path(__file__).parent / "assets" / "pieces"
        if assets_dir.exists():
            piece_mapping = {
                'P': 'pawn.png', 'N': 'knight.png', 'B': 'bishop.png',
                'R': 'rook.png', 'Q': 'queen.png', 'K': 'king.png',
                'p': 'pawn.png', 'n': 'knight.png', 'b': 'bishop.png',
                'r': 'rook.png', 'q': 'queen.png', 'k': 'king.png'
            }
            for piece, filename in piece_mapping.items():
                src = assets_dir / filename
                if src.exists():
                    shutil.copy(src, pieces_dir / f"{piece}.png")

    def get_state(self) -> dict:
        """Get chess-specific state"""
        try:
            board = self.env.board
            return {
                "fen": board.fen() if board else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "current_player": "White" if self.env.state.current_player_id == 0 else "Black",
                "is_check": board.is_check() if board else False,
                "is_checkmate": board.is_checkmate() if board else False,
                "is_stalemate": board.is_stalemate() if board else False,
                "move_stack": [move.uci() for move in board.move_stack] if board else [],
                "valid_moves": [move.uci() for move in board.legal_moves] if board else []
            }
        except Exception as e:
            print(f"Error getting state: {e}")
            return {
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "current_player": "White",
                "is_check": False,
                "is_checkmate": False,
                "is_stalemate": False,
                "move_stack": [],
                "valid_moves": []
            }

    def get_custom_js(self) -> str:
        """Get chess-specific JavaScript code"""
        return """
        console.log('Loading chess components...');

        // Chess components
        const ChessBoard = ({ fen }) => {
            const renderSquare = (i) => {
                const file = i % 8;
                const rank = Math.floor(i / 8);
                const isLight = (file + rank) % 2 === 0;
                
                return (
                    <div key={i} className={`square ${isLight ? 'light' : 'dark'}`}>
                        {getPiece(fen, i)}
                    </div>
                );
            };

            return (
                <div className="board-container">
                    <div className="chess-board">
                        {[...Array(64)].map((_, i) => renderSquare(i))}
                    </div>
                </div>
            );
        };

        const getPiece = (fen, squareIndex) => {
            const [position] = fen.split(' ');
            const rows = position.split('/');
            const rank = Math.floor(squareIndex / 8);
            const file = squareIndex % 8;
            
            let currentFile = 0;
            const row = rows[7 - rank];
            
            for (let i = 0; i < row.length; i++) {
                const char = row[i];
                if (isNaN(char)) {
                    if (currentFile === file) {
                        return (
                            <img 
                                src={`/static/pieces/${char}.png`}
                                alt={char}
                                className="piece-img"
                            />
                        );
                    }
                    currentFile++;
                } else {
                    currentFile += parseInt(char);
                }
                if (currentFile > file) break;
            }
            return null;
        };

        const renderChessGameInfo = (gameState) => {
            // Group moves into pairs
            const moveGroups = [];
            for (let i = 0; i < gameState.move_stack.length; i += 2) {
                moveGroups.push({
                    number: Math.floor(i / 2) + 1,
                    white: gameState.move_stack[i],
                    black: gameState.move_stack[i + 1],
                });
            }

            // Reference to the trade history container
            const moveHistoryRef = React.useRef(null);

            // Scroll to the bottom when trade history updates
            React.useEffect(() => {
                if (moveHistoryRef.current) {
                    moveHistoryRef.current.scrollTop = moveHistoryRef.current.scrollHeight;
                }
            }, [gameState.move_stack]); // Re-run whenever trade history updates

            return (
                <div>
                    <div>Current Turn: {gameState.current_player}</div>
                    {gameState.is_check && <div className="alert">Check!</div>}
                    {gameState.is_checkmate && <div className="alert">Checkmate!</div>}
                    {gameState.is_stalemate && <div className="alert">Stalemate</div>}

                    <h3>Players</h3>
                    <div className="players">
                        {Object.entries(gameState.player_names).map(([id, name]) => (
                            <div key={id} className={id === '0' ? 'white-player' : 'black-player'}>
                                {name} ({id === '0' ? 'White' : 'Black'})
                            </div>
                        ))}
                    </div>

                    <h3>Move History</h3>
                    <div 
                        className="move-history"
                        ref={moveHistoryRef}
                    >
                        {moveGroups.map((group, i) => (
                            <div key={i} className="move-pair">
                                <span className="move-number">{group.number}.</span>
                                <span className="move white">{group.white}</span>
                                {group.black && <span className="move black">{group.black}</span>}
                            </div>
                        ))}
                    </div>
                </div>
            );
        };


        // Main app
        const ChessGame = () => {
            console.log('Initializing ChessGame');
            const [gameState, setGameState] = React.useState(null);

            React.useEffect(() => {
                const ws = new WebSocket(`ws://${window.location.host}/ws`);
                ws.onopen = () => console.log('WebSocket connected');
                ws.onmessage = (event) => {
                    const state = JSON.parse(event.data);
                    console.log('Received state:', state);
                    setGameState(state);
                };
                return () => ws.close();
            }, []);

            if (!gameState) {
                console.log('Waiting for game state...');
                return <div>Loading game state...</div>;
            }

            return (
                <BaseGameContainer gameState={gameState} renderGameInfo={renderChessGameInfo}>
                    <div className="chess-layout">
                        <div className="main-content">
                            <ChessBoard fen={gameState.fen} />
                        </div>
                    </div>
                </BaseGameContainer>
            );
        };


        // Initialize the app
        console.log('Initializing React app');
        const root = ReactDOM.createRoot(document.getElementById('root'));
        root.render(<ChessGame />);
        console.log('Chess components loaded');
        """

    def get_custom_css(self) -> str:
        return """
        .game-header {
            text-align: center;
            margin-bottom: 20px;
        }

        .game-title {
            font-size: 32px;
            font-weight: bold;
            color: #ffffff;
            margin: 0;
            padding: 10px 0;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            background: linear-gradient(45deg, #4CAF50, #2196F3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        }

        .main-content {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }

        .board-container {
            flex: 0 0 600px;
            background: #363636;
            padding: 20px;
            border-radius: 8px;
        }

        .chess-board {
            display: grid;
            grid-template-columns: repeat(8, 1fr);
            width: 600px;
            height: 600px;
            border: 2px solid #404040;
        }

        .square {
            width: 75px;
            height: 75px;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .square.light { background-color: #EEEED2; }
        .square.dark { background-color: #769656; }

        .piece-img {
            width: 60px;
            height: 60px;
            user-select: none;
        }

        .info-container {
            flex: 1;
            background: #363636;
            padding: 20px;
            border-radius: 8px;
        }

        .status { margin-bottom: 20px; }
        .alert { color: #ff4444; font-weight: bold; }
        
        .players {
            margin-bottom: 20px;
        }

        .white-player { color: #ffffff; }
        .black-player { color: #000000; }

        .move-history {
            font-family: monospace;
            margin-top: 10px;
            background: #2B2B2B;
            padding: 12px;
            border-radius: 4px;
            max-height: 200px;
            overflow-y: auto;
        }

        .move {
            display: inline-block;
            margin: 2px 0;
            padding: 2px 6px;
            border-radius: 3px;
        }

        .move.white {
            color: #ffffff;
            background: #404040;
        }

        .move.black {
            color: #a0a0a0;
            background: #333333;
        }

        .move-number {
            color: #666666;
            margin-right: 4px;
        }

        .move-pair {
            display: block;
            margin-bottom: 4px;
        }

        .chat-message {
            margin-bottom: 10px;
            padding: 8px;
            border-radius: 4px;
            background: #404040;
        }

        .chat-message .player-name {
            font-weight: bold;
            margin-bottom: 5px;
        }

        .chat-message.white .player-name { color: #ffffff; }
        .chat-message.black .player-name { color: #a0a0a0; }
        """