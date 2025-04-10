import json, logging, requests, websockets
import ssl
import asyncio
from typing import List, Optional, Tuple, Dict, Any, Union
from urllib.parse import urlencode
import warnings
from urllib3.exceptions import InsecureRequestWarning
import traceback

# Suppress SSL warnings
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# Server URLs - Change these to match your server
MATCHMAKING_WS_URI = "wss://matchmaking.textarena.ai/ws"
MATCHMAKING_HTTP_URI = "https://matchmaking.textarena.ai"


# Environment ID mapping
NAME_TO_ID_DICT = {
    "Chess-v0": 0,
    "ConnectFour-v0": 1,
    "DontSayIt-v0": 3,
    "Battleship-v0": 5,
    "LiarsDice-v0": 6,
    "SimpleNegotiation-v0": 8,
    "Poker-v0": 9,
    "SpellingBee-v0": 10,
    "SpiteAndMalice-v0": 11,
    "Stratego-v0": 12,
    "Tak-v0": 13,
    "TruthAndDeception-v0": 14,
    "UltimateTicTacToe-v0": 15,
    "WordChains-v0": 16,
    "TicTacToe-v0": 35,
    "Breakthrough-v0": 37,
    "Checkers-v0": 38,
    "KuhnPoker-v0": 46,
    "LetterAuction-v0": 47,
    "MemoryGame-v0": 48,
    "Nim-v0": 50,
    "Othello-v0": 51,
    "PigDice-v0": 52,
    "SimpleBlindAuction-v0": 56,
    "Snake-v0": 69,
    "SecretMafia-v0": 75,
    "WildTicTacToe-v0": 77,
    "ReverseTicTacToe-v0": 78,
    "RandomizedTicTacToe-v0": 79,
    "QuantumTicTacToe-v0": 80,
    "IteratedRockPaperScissors-v0": 81
}

class OnlineEnvWrapper:
    def __init__(self, env_ids: List[int], model_name: str, model_token: str):
        self.env_ids = env_ids
        self.model_name = model_name
        self.model_token = model_token
        self.websocket = None
        
        # Matchmaking variables
        self.matchmaking_websocket = None
        self.game_url = None
        self.environment_id = None
        self.env_id = None
        
        # The full observations are stored as a dictionary mapping player id -> list of (sender_id, message) tuples
        self.full_observations = {}
        
        # Game state tracking
        self.current_player_id = None
        self.current_observation = None
        self.game_over = False
        self.server_shutdown = False  # New flag to track server shutdown
        self.game_over_timeout = 30.0  # Increased time to wait for additional messages after game_over

        self.rewards = {}
        self.info = {}
        
        # Timeouts
        # self.move_timeout = 240     # Move deadline is longer than server's to avoid timing out
        self.matchmaking_timeout = 1800  # Timeout for matchmaking (30 minutes)

        
        # Message and action queues
        self.message_queue = asyncio.Queue()
        self.action_queue = asyncio.Queue()
        self.matchmaking_queue = asyncio.Queue()  # For matchmaking messages
        
        # State tracking
        self.in_game = False
        self.pending_action = False
        self.update_task = None  # Reference to the main update loop task
        self.matchmaking_complete = False
        
        # For compatibility
        DummyState = type("DummyState", (), {})
        self.state = DummyState()
        self.state.role_mapping = {0: "Player 0", 1: "Player 1", -1: "GAME"}

    async def _message_receiver(self):
        """Task to receive and queue messages from websocket."""
        try:
            while True:
                try:
                    message = await self.websocket.recv()
                    print(f"Received: {message}")
                    await self.message_queue.put(message)
                    
                    # Quick check if this is a server_shutdown message
                    # This ensures we don't miss it if the queue processing is slow
                    try:
                        msg_data = json.loads(message)
                        if msg_data.get("command") == "server_shutdown":
                            print("Server shutdown message detected in receiver")
                            self.server_shutdown = True
                    except:
                        pass
                        
                except websockets.exceptions.ConnectionClosed:
                    print("WebSocket connection closed by server")
                    self.server_shutdown = True
                    break
                except Exception as e:
                    print(f"Error receiving message: {e}")
                    break
        except Exception as e:
            print(f"Message receiver error: {e}")
            self.server_shutdown = True  # Set server_shutdown to ensure all loops terminate

    async def _matchmaking_receiver(self):
        """Task to receive and queue messages from matchmaking websocket."""
        try:
            while not self.matchmaking_complete:
                try:
                    message = await self.matchmaking_websocket.recv()
                    print(f"Received from matchmaking: {message}")
                    await self.matchmaking_queue.put(message)
                except websockets.exceptions.ConnectionClosed:
                    print("Matchmaking WebSocket connection closed")
                    break
                except Exception as e:
                    print(f"Error receiving matchmaking message: {e}")
                    break
        except Exception as e:
            print(f"Matchmaking receiver error: {e}")

    async def _action_sender(self):
        """Task to send actions to the websocket."""
        try:
            while True:
                action = await self.action_queue.get()
                if action == "CLOSE":
                    break
                
                try:
                    action_msg = {"command": "action", "action": action}
                    await self.websocket.send(json.dumps(action_msg))
                    print(f"Sent action: {action[:100]}...")
                    self.pending_action = True
                except Exception as e:
                    print(f"Error sending action: {e}")
                
                self.action_queue.task_done()
        except Exception as e:
            print(f"Action sender error: {e}")
            self.server_shutdown = True  # Changed from self.game_over

    async def _ping_sender(self):
        """Task to send periodic pings to keep connection alive."""
        try:
            while not self.server_shutdown:  # Changed from self.game_over
                try:
                    await self.websocket.send(json.dumps({"command": "ping"}))
                    await asyncio.sleep(25)  # Send ping every 25 seconds
                except Exception as e:
                    print(f"Ping error: {e}")
                    break
        except Exception as e:
            print(f"Ping sender error: {e}")
            self.server_shutdown = True  # Changed from self.game_over


    async def _process_matchmaking_message(self, message_str: str):
        """Process incoming matchmaking WebSocket messages."""
        try:
            message = json.loads(message_str)
            command = message.get("command")
            
            if command == "queued":
                avg_queue_time = message.get("avg_queue_time", 0)
                num_players = message.get("num_players_in_queue", 0)
                print(f"In queue. Average wait time: {avg_queue_time:.1f}s. Players in queue: {num_players}")
                
            elif command == "match_found":
                # We found a match and need to connect to the game server
                self.game_url = message.get("game_url")
                self.env_id = message.get("env_id")
                self.environment_id = message.get("environment_id")
                print(f"Match found! Connecting to game server: {self.game_url}")
                self.matchmaking_complete = True
                
            elif command == "error":
                error_msg = message.get("message", "Unknown error")
                print(f"Matchmaking error: {error_msg}")
                
            elif command == "left":
                print("Left matchmaking queue")
                
            else:
                print(f"Unknown matchmaking command: {command}")
                
        except json.JSONDecodeError:
            print(f"Invalid JSON received from matchmaking: {message_str}")
        except Exception as e:
            print(f"Error processing matchmaking message: {e}")

    async def connect_to_matchmaking(self):
        """Connect to matchmaking server and queue for a game."""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        # Connect with model info for AI models
        query_params = {
            "model_name": self.model_name,
            "model_token": self.model_token,
        }
        query_string = urlencode(query_params)
        ws_uri = f"{MATCHMAKING_WS_URI}?{query_string}"
        
        print(f"Connecting to matchmaking server: {ws_uri}")
        
        try:
            # Create WebSocket connection
            self.matchmaking_websocket = await websockets.connect(
                ws_uri,
                # ssl=ssl_context,  # Uncomment for HTTPS
                ping_interval=20,
                ping_timeout=60
            )
            
            # Start background tasks for matchmaking
            asyncio.create_task(self._matchmaking_receiver())
            
            # Queue for a game
            queue_message = {
                "command": "queue",
                "environments": self.env_ids
            }
            await self.matchmaking_websocket.send(json.dumps(queue_message))
            print(f"Sent queue request for environments: {self.env_ids}")
            
            # Wait for match to be found or timeout
            start_time = asyncio.get_event_loop().time()
            while not self.matchmaking_complete:
                try:
                    message = await asyncio.wait_for(
                        self.matchmaking_queue.get(),
                        timeout=1.0  # Check every second
                    )
                    await self._process_matchmaking_message(message)
                except asyncio.TimeoutError:
                    # Check if we should timeout the matchmaking
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed > self.matchmaking_timeout:
                        print("Timeout waiting for match")
                        await self.matchmaking_websocket.close()
                        return False
                    continue
            
            # Close matchmaking connection - the server will do this anyway
            try:
                await self.matchmaking_websocket.close()
            except:
                pass
                
            return self.game_url is not None
            
        except Exception as e:
            print(f"Matchmaking connection error: {e}")
            return False

    async def connect_to_game_server(self):
        """Connect to the game server after matchmaking is complete."""
        if not self.game_url:
            print("No game server IP available")
            return False
                
        # Properly configure SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Short delay to allow server initialization
        await asyncio.sleep(1)

        # Connect to the game server with our token
        ws_uri = f"wss://{self.game_url}/ws?token={self.model_token}"
        # ws_uri = f"ws://localhost:8000/ws?token={self.model_token}"
        print(f"Connecting to game server: {ws_uri}")
        
        # Try multiple connection attempts
        max_attempts = 10
        for attempt in range(1, max_attempts + 1):
            try:
                # Create WebSocket connection with compatible parameters
                self.websocket = await websockets.connect(
                    ws_uri,
                    ssl=ssl_context,
                    ping_interval=30,
                    ping_timeout=90
                )
                
                # Start background tasks
                asyncio.create_task(self._message_receiver())
                asyncio.create_task(self._action_sender())
                asyncio.create_task(self._ping_sender())
                
                print("Connected to game server")
                return True
                
            except Exception as e:
                print(f"Connection error (attempt {attempt}/{max_attempts}): {e}")
                if attempt < max_attempts:
                    await asyncio.sleep(2)
                else:
                    print("All connection attempts failed")
                    return False

    async def connect(self):
        """Connect to matchmaking and then to the game server."""
        # First connect to matchmaking
        matchmaking_success = await self.connect_to_matchmaking()
        if not matchmaking_success:
            print("Failed to get a match")
            return False
            
        # Then connect to the game server
        return await self.connect_to_game_server()

    async def _process_message(self, message_str: str):
        """Process incoming WebSocket messages."""
        try:
            message = json.loads(message_str)
            command = message.get("command")
            
            if command == "observation":
                # Received game state - our turn to act
                observation = message.get("observation")
                player_id = message.get("player_id")
                
                print(f"Received observation for player {player_id}")
                self.current_player_id = player_id
                self.current_observation = observation
                self.full_observations[player_id] = observation
                self.pending_action = False
                self.in_game = True
                
            elif command == "game_over":
                # Game has completed
                print("Game over received")
                self.game_over = True  # Set game_over but not server_shutdown
                outcome = message.get("outcome", "unknown")
                reason = message.get("reason", "No reason provided")
                
                # Extract reward info if available
                trueskill_change = message.get("trueskill_change", 0)
                if self.current_player_id is not None:
                    self.rewards[self.current_player_id] = trueskill_change
                
                self.info = {"reason": reason, "outcome": outcome}
                print(f"Game over: {outcome}, reason: {reason}")
                
            elif command == "timed_out":
                # Someone timed out
                timeout_msg = message.get("message", "Unknown timeout")
                print(f"Game timeout: {timeout_msg}")
                self.game_over = True
                self.info = {"reason": "timeout", "message": timeout_msg}
                
            elif command == "error":
                # Server error
                error_msg = message.get("message", "Unknown error")
                print(f"Server error: {error_msg}")
                self.info = {"error": error_msg}
                
            elif command == "action_ack":
                # Action acknowledged
                print("Action acknowledged by server")
                
            elif command == "pong":
                # Response to our ping
                pass
                
            elif command == "ping":
                # Server ping - respond with pong
                try:
                    await self.websocket.send(json.dumps({"command": "pong"}))
                except Exception as e:
                    print(f"Error sending pong: {e}")
                    
            elif command == "server_shutdown":
                # Server is shutting down
                print("Server shutdown message received")
                self.server_shutdown = True  # This will cause the update loop to exit
                
            else:
                print(f"Unknown command received: {command}")
                
        except json.JSONDecodeError:
            print(f"Invalid JSON received: {message_str}")
        except Exception as e:
            print(f"Error processing message: {e}")
            
    async def update_loop(self):
        """Main loop that processes messages."""
        game_over_time = None  # Track when game_over was received
        
        while not self.server_shutdown:  # Changed from self.game_over to self.server_shutdown
            try:
                # If game is over, use a shorter timeout to not wait too long
                # Increased from 1.0 to 5.0 for game_over timeout to ensure we get any final messages

                timeout = 5.0 if self.game_over else None
                
                # Process incoming messages with timeout
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=timeout)
                    await self._process_message(message)
                    
                    # If we just set game_over, record the time
                    if self.game_over and game_over_time is None:
                        game_over_time = asyncio.get_event_loop().time()
                        print("Game over received, waiting for additional messages...")
                    
                except asyncio.TimeoutError:
                    if self.game_over:
                        elapsed = asyncio.get_event_loop().time() - game_over_time
                        print(f"Timeout after {elapsed:.1f}s while waiting for additional messages after game over")
                        
                        # Only shut down if we've waited long enough
                        if elapsed > self.game_over_timeout:
                            print(f"No more messages after {self.game_over_timeout}s of game over, exiting loop")
                            self.server_shutdown = True
                    else:
                        print(f"Timeout while waiting for messages")
                        self.game_over = True
                        self.server_shutdown = True
                    
                # Check if we've waited long enough after game_over
                if self.game_over and game_over_time is not None:
                    elapsed = asyncio.get_event_loop().time() - game_over_time
                    if elapsed > self.game_over_timeout:
                        print(f"No more messages after {self.game_over_timeout}s of game over, exiting loop")
                        self.server_shutdown = True
                    
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed by server")
                self.server_shutdown = True  # Set server_shutdown when connection is closed
                break
                
            except Exception as e:
                print(f"Error in update loop: {e}")
                await asyncio.sleep(0.1)
                
        print("Update loop exiting")

    async def async_get_observation(self) -> Tuple[Optional[int], List]:
        """Get the current observation."""
        # If we already have an observation, return it
        if self.current_player_id is not None and self.current_observation:
            observation = self.current_observation
            player_id = self.current_player_id
            return player_id, observation
        
        # Otherwise, wait for an observation
        if not self.server_shutdown:  # Changed from self.game_over
            # Make sure we're not starting the update loop multiple times
            if self.update_task is None or self.update_task.done():
                self.update_task = asyncio.create_task(self.update_loop())
            
            try:
                # Wait until we get an observation or server shuts down
                start_time = asyncio.get_event_loop().time()
                
                while not self.server_shutdown:  # Changed from self.game_over
                    # Check if we have an observation
                    if self.current_player_id is not None and self.current_observation:
                        return self.current_player_id, self.current_observation
                    
                    await asyncio.sleep(0.1)
                    
                    # # Check for timeout
                    # elapsed = asyncio.get_event_loop().time() - start_time
                    # if elapsed > self.move_timeout:
                    #     print("Timeout waiting for observation")
                    #     self.game_over = True
                    #     self.server_shutdown = True  # Also set server_shutdown
                    #     break
                        
            except Exception as e:
                print(f"Error waiting for observation: {e}")
                    
        # If server is shutting down or we timed out
        self.observation_valid = False  # <-- ADD THIS
        return None, []

    def get_observation(self) -> Tuple[Optional[int], List]:
        """Synchronous wrapper for async_get_observation."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                new_loop = True
            else:
                new_loop = False
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            new_loop = True

        try:
            player_id, obs = loop.run_until_complete(self.async_get_observation())

            # Raise if invalid observation
            if getattr(self, "observation_valid", True) is False:
                raise RuntimeError(f"No valid observation — reason: {self.info.get('reason', 'unknown')}")

            return player_id, obs
        finally:
            if new_loop:
                loop.close()



    async def async_step(self, action: str):
        """Take an action in the game."""
        if self.server_shutdown:  # Changed from self.game_over
            return True, self.info
        
        # Queue action to be sent
        await self.action_queue.put(action)
        
        # Clear current observation - the key fix!
        # We need to clear this AFTER we successfully send the action
        # but we don't want to clear it until we've actually sent it
        await self.action_queue.join()  # Wait for action to be sent
        self.current_observation = None
        
        # Wait for response (new observation or game over)
        start_time = asyncio.get_event_loop().time()
        while not self.server_shutdown and self.pending_action:  # Game might be over but server not shut down
            await asyncio.sleep(0.1)
            
            # # Check for timeout
            # elapsed = asyncio.get_event_loop().time() - start_time
            # if elapsed > self.move_timeout:
            #     print("Timeout waiting for server response")
            #     self.game_over = True
            #     self.server_shutdown = True  # Also set server_shutdown
            #     break
                
        return self.game_over, self.info  # Return game_over not server_shutdown


    def step(self, action: str):
        """Synchronous wrapper for async_step."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                new_loop = True
            else:
                new_loop = False
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            new_loop = True
            
        try:
            return loop.run_until_complete(self.async_step(action))
        finally:
            if new_loop:
                loop.close()

    async def async_reset(self, num_players=None, seed=None):
        """Connect to server and wait for game to start."""
        # Reset state
        self.current_player_id = None
        self.current_observation = None
        self.game_over = False
        self.server_shutdown = False  # Reset server_shutdown flag too
        self.rewards = {}
        self.info = {}
        self.full_observations = {}
        self.in_game = False
        self.update_task = None
        self.matchmaking_complete = False
        
        # Connect to server - this now includes matchmaking
        if not self.websocket:
            connected = await self.connect()
            if not connected:
                print("Failed to connect to server")
                # Cancel any existing tasks explicitly
                await self.async_close()  # This would ensure all tasks are cancelled
                return []
                
        # Wait for game to start and get initial observation
        self.update_task = asyncio.create_task(self.update_loop())
        
        try:
            # Wait until we either get an observation or the server shuts down
            start_time = asyncio.get_event_loop().time()
            while not self.server_shutdown and not self.in_game:  # Changed from self.game_over
                await asyncio.sleep(0.1)
                
                # Check if we have an observation
                if self.current_player_id is not None and self.current_observation:
                    self.in_game = True
                    return self.current_observation
                
                # # Check for timeout
                # elapsed = asyncio.get_event_loop().time() - start_time
                # if elapsed > self.move_timeout:
                #     print("Timeout waiting for game to start")
                #     self.server_shutdown = True  # Set server_shutdown on timeout
                #     break
                    
        except Exception as e:
            print(f"Error waiting for game start: {e}")
                
        # Return current observation or empty list
        return self.current_observation if self.current_observation else []


    def reset(self, num_players=None, seed=None):
        """Synchronous wrapper for async_reset."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                new_loop = True
            else:
                new_loop = False
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            new_loop = True
            
        try:
            return loop.run_until_complete(self.async_reset(num_players))
        finally:
            if new_loop:
                loop.close()

    async def async_close(self):
        """Clean up resources."""
        # Set server_shutdown flag to ensure all loops terminate
        self.server_shutdown = True
        
        # Signal action sender to stop
        try:
            await self.action_queue.put("CLOSE")
        except:
            pass
            
        # Close websocket
        if self.websocket and not getattr(self.websocket, 'closed', True):
            try:
                await self.websocket.close()
            except:
                pass
            
        # Close matchmaking websocket if still open
        if self.matchmaking_websocket and not getattr(self.matchmaking_websocket, 'closed', True):
            try:
                await self.matchmaking_websocket.close()
            except:
                pass
                
        # Cancel update task if running
        if self.update_task and not self.update_task.done():
            try:
                self.update_task.cancel()
            except:
                pass
                
        return self.rewards

    def close(self):
        """Synchronous wrapper for async_close."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                new_loop = True
            else:
                new_loop = False
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            new_loop = True
            
        try:
            return loop.run_until_complete(self.async_close())
        finally:
            if new_loop:
                loop.close()


def register_model(model_name: str, description: str, email: str) -> str:
    """Register a model with the matchmaking server and get a token."""
    try:
        response = requests.post(
            f"{MATCHMAKING_HTTP_URI}/register_model",
            json={
                "model_name": model_name,
                "description": description,
                "email": email
            }
        )
        response.raise_for_status()
        data = response.json()
        return data.get("model_token")
    except Exception as e:
        print(f"Error registering model: {e}")
        return None


def make_online(
    env_id: Union[str, List[str]],
    model_name: str,
    model_token: Optional[str] = None,
    model_description: Optional[str] = None,
    email: Optional[str] = None,
) -> OnlineEnvWrapper:
    """Create and return the online environment wrapper.
    
    Args:
        env_id: The environment ID (e.g., "SpellingBee-v0") or a list of environment IDs
        model_name: The name of the model
        model_token: Optional token for the model (if already registered)
        model_description: Description of the model (required if model_token is None)
        email: Email address (required if model_token is None)
    
    Returns:
        An OnlineEnvWrapper instance
    """
    # Convert env_id to a list if it's a single string
    env_ids = [env_id] if isinstance(env_id, str) else env_id
    
    # Convert environment names to IDs
    env_ids_int = []
    if env_ids[0] == "all":
        env_ids_int = list(NAME_TO_ID_DICT.values())
    else:
        for env_name in env_ids:
            if env_name in NAME_TO_ID_DICT:
                env_ids_int.append(NAME_TO_ID_DICT[env_name])
            else:
                raise ValueError(f"Environment {env_name} not recognized")
    
    # Generate or get a token
    if model_token is None:
        if model_description is None or email is None:
            raise ValueError("Provide model_description and email if model_token is not given")
        
        # Register model with server to get token
        model_token = register_model(model_name, model_description, email)
        if not model_token:
            raise ValueError("Failed to register model with server")
            
        print(f"Registered model and received token: {model_token}")
    
    return OnlineEnvWrapper(env_ids_int, model_name, model_token)