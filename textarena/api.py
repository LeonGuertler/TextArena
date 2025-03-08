import json, logging, requests, websockets
import ssl
import asyncio
from typing import List, Optional, Tuple, Dict, Any, Union
from urllib.parse import urlencode
import warnings
from urllib3.exceptions import InsecureRequestWarning

# Suppress SSL warnings
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# Constants
# WS_SERVER_URI = "ws://0.0.0.0:8000/ws"
# HTTP_SERVER_URI = "http://0.0.0.0:8000"
WS_SERVER_URI = "wss://api.textarena.ai/ws"
HTTP_SERVER_URI = "https://api.textarena.ai"



NAME_TO_ID_DICT = {
    "Chess-v0": 0,
    "ConnectFour-v0": 1,
    # "Debate-v0": 2,
    "DontSayIt-v0": 3,
    # "IteratedPrisonersDilemma-v0": 4,
    "Battleship-v0": 5,
    "LiarsDice-v0": 6,
    # "Mastermind-v0": 7,
    "SimpleNegotiation-v0": 8,
    "Poker-v0": 9,
    "SpellingBee-v0": 10,
    "SpiteAndMalice-v0": 11,
    "Stratego-v0": 12,
    "Tak-v0": 13,
    "TruthAndDeception-v0": 14,
    "UltimateTicTacToe-v0": 15,
    "TicTacToe-v0": 35,
    "Breakthrough-v0": 37,
    "Checkers-v0": 38,
    "KuhnPoker-v0": 46,
    "LetterAuction-v0": 47,
    "Nim-v0": 50,
    "Othello-v0": 51,
    "PigDice-v0": 52,
    # "SimpleBlindAuction-v0": 56,
    "Snake-v0": 69
}

class OnlineEnvWrapper:
    def __init__(self, env_ids: List[int], model_name: str, model_token: str):
        self.env_ids = env_ids
        self.model_name = model_name
        self.model_token = model_token
        self.websocket = None
        
        # The full observations are stored as a dictionary mapping player id -> list of (sender_id, message) tuples.
        self.full_observations: Dict[int, List[Tuple[int, str]]] = {}
        
        # For synchronization between websocket and game loop
        self.current_player_id = None
        self.current_observation = None
        self.game_over = False
        self.rewards = {}
        self.info = {}
        
        # Timeouts for waiting
        self.queue_timeout = 1800  # 30 minutes for matchmaking
        self.game_timeout = 300    # 5 minutes for game moves
        
        # Message queue for receiving websocket messages
        self.message_queue = asyncio.Queue()
        
        # Action queue for sending actions
        self.action_queue = asyncio.Queue()
        
        # State tracking
        self.in_game = False
        self.waiting_for_action_response = False
        self.connection_established = False  # Track if connection is established
        
        # Dummy state for compatibility if needed
        DummyState = type("DummyState", (), {})
        self.state = DummyState()
        self.state.role_mapping = {0: "Player 0", 1: "Player 1", -1: "GAME"}

    async def _message_receiver(self):
        """Background task to receive messages from the websocket."""
        try:
            while True:
                try:
                    message = await self.websocket.recv()
                    print(f"Received websocket message: {message}")
                    await self.message_queue.put(message)
                except websockets.exceptions.ConnectionClosedOK:
                    print("Websocket closed normally")
                    break
                except websockets.exceptions.ConnectionClosed:
                    print("Websocket connection closed unexpectedly")
                    break
                except Exception as e:
                    print(f"Error receiving message: {e}")
                    break
        except Exception as e:
            if not self.game_over:
                print(f"Message receiver error: {e}")

    async def _action_sender(self):
        """Background task to send actions to the websocket."""
        try:
            while True:
                action = await self.action_queue.get()
                if action == "CLOSE":
                    print("Action sender received close signal")
                    break
                
                try:
                    action_msg = {"command": "action", "action": action}
                    await self.websocket.send(json.dumps(action_msg))
                    print(f"Sent action: {action}")
                    self.waiting_for_action_response = True
                except Exception as e:
                    print(f"Error sending action: {e}")
                
                self.action_queue.task_done()
        except Exception as e:
            print(f"Action sender error: {e}")

    async def connect(self):
        """Connect to server and queue for game."""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        params = {"model_name": self.model_name, "model_token": self.model_token}
        query_params = "?" + urlencode(params)
        
        try:
            print(f"Connecting to {WS_SERVER_URI + query_params}")
            self.websocket = await websockets.connect(
                WS_SERVER_URI + query_params,
                # ssl=ssl_context,  # Uncomment for HTTPS
                ping_interval=20,
                ping_timeout=60
            )
            print("Connected to server")
            
            # Start background tasks
            asyncio.create_task(self._message_receiver())
            asyncio.create_task(self._action_sender())
            
            # Queue for a game
            queue_command = {"command": "queue", "environments": self.env_ids}
            await self.websocket.send(json.dumps(queue_command))
            print("Sent queue request")
            
            # Wait for queue confirmation
            message = await self.message_queue.get()
            print(f"Queue response: {message}")
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    async def update_loop(self):
        """Main loop that processes messages and updates game state."""
        while not self.game_over:
            try:
                # Process any new messages
                timeout = self.queue_timeout if not self.in_game else self.game_timeout
                
                try:
                    message = await asyncio.wait_for(self.message_queue.get(), timeout=timeout)
                    await self._process_message(message)
                except asyncio.TimeoutError:
                    state = "matchmaking" if not self.in_game else "game"
                    print(f"Timeout during {state}")
                    self.game_over = True
                    break
            
            except Exception as e:
                print(f"Error in update loop: {e}")
                if not self.game_over:
                    # Small delay before trying again
                    await asyncio.sleep(0.1)
        
        print("Update loop exiting")

    async def _process_message(self, message: str):
        """Process a received websocket message."""
        try:
            payload = json.loads(message)
            command = payload.get("command")
            
            if command == "queued":
                # Successfully queued, just log and continue waiting
                avg_queue_time = payload.get("avg_queue_time", 0)
                num_active = payload.get("num_active_players", 0)
                print(f"Queued for game. Avg wait: {avg_queue_time}s, Active players: {num_active}")
            
            elif command == "match_found":
                # Game is starting
                self.in_game = True
                player_id = payload.get("player_id")
                obs = payload.get("observation", [])
                
                print(f"Match found! Playing as player {player_id}")
                
                if obs:  # Starting player
                    self.current_player_id = player_id
                    self.full_observations[player_id] = obs
                    self.current_observation = obs
                    print(f"Starting player received observation of length {len(obs)}")
                else:
                    # Not our turn yet, just store player_id
                    self.current_player_id = player_id
                    print("Waiting for first observation (not starting player)")
            
            elif command == "observation":
                # Received a new observation (our turn)
                obs = payload.get("observation", [])
                player_id = payload.get("player_id")
                
                if obs:
                    self.current_player_id = player_id
                    self.full_observations[player_id] = obs
                    self.current_observation = obs
                    print(f"Received observation for player {player_id}, length: {len(obs)}")
                
                # If we were waiting for action response, clear the flag
                self.waiting_for_action_response = False
            
            elif command == "game_over":
                # Game has ended
                self.game_over = True
                game_id = payload.get("game_id")
                opponent = payload.get("opponent_name", "Unknown")
                outcome = payload.get("outcome", "unknown")
                reason = payload.get("reason", "No reason provided")
                
                print(f"Game over! ID: {game_id}, Opponent: {opponent}, Outcome: {outcome}, Reason: {reason}")
                
                # Store rewards information if available
                change_in_skill = payload.get("change_in_skill")
                if change_in_skill is not None:
                    self.rewards[self.current_player_id] = float(change_in_skill)
                
                self.info = {"reason": reason, "outcome": outcome}
            
            elif command == "error":
                error_msg = payload.get("message", "Unknown error")
                print(f"Received error from server: {error_msg}")
                self.game_over = True
                self.info = {"error": error_msg}
            
            else:
                print(f"Unknown command received: {command}")
        
        except json.JSONDecodeError:
            print(f"Invalid JSON received: {message}")
        except Exception as e:
            print(f"Error processing message: {e}, message: {message}")

    async def async_get_observation(self) -> Tuple[Optional[int], List[Tuple[int, str]]]:
        """Asynchronous method to get the next observation."""
        # If we're in a game and have an observation, return it immediately
        if self.current_player_id is not None and self.current_observation:
            return self.current_player_id, self.current_observation
        
        # Otherwise, we need to wait for an observation
        if not self.game_over:
            timeout = self.queue_timeout if not self.in_game else self.game_timeout
            
            # Start the update loop if not already running
            update_task = asyncio.create_task(self.update_loop())
            
            # Wait until we have a valid observation or the game ends
            start_time = asyncio.get_event_loop().time()
            while not self.game_over:
                # Return when we have a valid observation
                if self.current_player_id is not None and self.current_observation:
                    if not update_task.done():
                        update_task.cancel()
                    return self.current_player_id, self.current_observation
                
                await asyncio.sleep(0.1)
                
                # Check for timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout:
                    print(f"Timeout waiting for observation")
                    self.game_over = True
                    break
            
            # Cancel the update task if still running
            if not update_task.done():
                update_task.cancel()
        
        # If we get here, either game is over or we timed out
        return None, []

    def get_observation(self) -> Tuple[Optional[int], List[Tuple[int, str]]]:
        """
        Synchronous wrapper for async_get_observation.
        When the local code calls get_observation() (without awaiting), it will receive the result.
        
        Returns:
            A tuple of (player_id, observation)
        """
        # Check if there's an existing event loop we can use
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                new_loop = True
            else:
                new_loop = False
        except RuntimeError:
            # No event loop found, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            new_loop = True
            
        try:
            return loop.run_until_complete(self.async_get_observation())
        finally:
            if new_loop:
                loop.close()

    async def async_step(self, action: str):
        """Take an action in the game asynchronously."""
        if self.game_over:
            return True, self.info
        
        # Queue the action to be sent
        await self.action_queue.put(action)
        
        # Clear current observation
        self.current_observation = None
        self.waiting_for_action_response = True
        
        # Wait for response (new observation or game over)
        start_time = asyncio.get_event_loop().time()
        while not self.game_over and self.waiting_for_action_response:
            await asyncio.sleep(0.1)
            
            # Check for timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > self.game_timeout:
                print(f"Timeout waiting for action response")
                self.game_over = True
                break
        
        return self.game_over, self.info
        
    def step(self, action: str):
        """
        Synchronous wrapper for async_step.
        
        Args:
            action: The action to take
            
        Returns:
            Tuple of (done, info)
        """
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
        """Asynchronous reset - establish connection and wait for a match.
        
        Args:
            num_players: Ignored for online play (included for API compatibility)
            seed: Ignored for online play (included for API compatibility)
        
        Returns:
            The initial observation if available
        """
        # Connect to the server if not already connected
        if not self.websocket or self.websocket.closed:
            connected = await self.connect()
            if not connected:
                print("Failed to connect to server")
                return []
                
        # Start the update loop
        update_task = asyncio.create_task(self.update_loop())
        
        # Wait until we have an observation or the game ends
        # This means we've found a match and received our first observation
        start_time = asyncio.get_event_loop().time()
        while not self.game_over and not self.in_game:
            await asyncio.sleep(0.1)
            
            # Check for timeout
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > self.queue_timeout:
                print(f"Timeout waiting for match")
                self.game_over = True
                break
        
        # Once we find a match, wait for initial observation if we're not first to move
        if self.in_game and not self.current_observation:
            player_id, observation = await self.async_get_observation()
            if player_id is not None:
                return observation
        
        # Return current observation or empty list
        return self.current_observation if self.current_observation else []

    async def async_close(self):
        """Asynchronous close - clean up resources."""
        # Signal action sender to stop
        await self.action_queue.put("CLOSE")
        
        # Close websocket connection
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
        
        return self.rewards

    def reset(self, num_players=None, seed=None):
        """
        Synchronous wrapper for async_reset.
        
        Args:
            num_players: Ignored for online play (included for API compatibility)
        
        Returns:
            The initial observation if available
        """
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
    
    def close(self):
        """
        Synchronous wrapper for async_close.
        
        Returns:
            Final rewards dictionary mapping player IDs to reward values
        """
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
    try:
        # Convert env_id to a list if it's a single string
        env_ids = [env_id] if isinstance(env_id, str) else env_id
        env_ids_int = [NAME_TO_ID_DICT[env] for env in env_ids]
    except KeyError as e:
        logging.error(f"Environment {e} not recognized")
        raise

    if model_token is None:
        if model_description is None or email is None:
            raise ValueError("Provide model_description and email if model_token is not given")
        url = f"{HTTP_SERVER_URI}/register_model"
        data = {"model_name": model_name, "description": model_description, "email": email}
        response = requests.post(url, json=data, timeout=10, verify=False)
        response.raise_for_status()
        model_token = response.json()["model_token"]
        logging.info("Model registered successfully")

    return OnlineEnvWrapper(env_ids_int, model_name, model_token)