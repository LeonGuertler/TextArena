from typing import Dict, Optional, List, Tuple, Any
import random
import textarena as ta
import re

class SpiteAndMaliceEnv(ta.Env):
    """
    Environment for Spite and Malice.
    """
    def __init__(self):
        """ Initialize the Spite and Malice environment """
        # Initialize the deck and shuffle
        self.deck = [f"{rank}{suit}" for rank in "A23456789JQK" for suit in "♠♥♦♣"] * 2
        self.action_space = lambda player_id: re.compile(r"\[(play|discard|draw)(?: ([A23456789JQK][♠♥♦♣]) ([0-3]))?\]")
        
    @property
    def terminal_render_keys(self):
        return ["rendered_board","player_turn"]

    def reset(self, num_players: int = 2, seed: Optional[int] = None):
        """ Reset the environment to start a new game """
        # Initialize the game state
        self.state = ta.State(num_players=2, min_players=2, max_players=2, max_turns=None, seed=seed)
        
        ## Initialize the players' payoff piles, hand, discard piles, and center piles
        random.shuffle(self.deck)
        self.players = self._initialize_players()
        self.center_piles = [[] for _ in range(4)]
        
        ## Draw cards for each player
        self._draw_cards(0)
        self._draw_cards(1)

        ## Return the initial observations
        game_state={
            "players": self.players, "center_piles": self.center_piles,
            "player_turn": self.state.current_player_id, "rendered_board": self._render_board()
            }
        return self.state.reset(
            game_state=game_state,
            player_prompt_function=self._generate_player_prompt
        )
    
    def _initialize_players(self):
        """ Initialize the players' payoff piles, hand, and discard piles. """
        players = {0: {"payoff": [], "hand": [], "discard": [[] for _ in range(4)]},
                   1: {"payoff": [], "hand": [], "discard": [[] for _ in range(4)]}}
        
        ## Deal the payoff piles (20 cards each for a shorter game)
        for player in players:
            players[player]["payoff"] = [self.deck.pop() for _ in range(20)]
        
        return players
    
    def _draw_cards(self, player_id: int):
        """ Draw cards to maintain 5 cards in hand. """
        if self.deck:
            while len(self.players[player_id]["hand"]) < 5:
                self.players[player_id]["hand"].append(self.deck.pop())
        else:
            message=(
                "There are no more cards to draw from. Remember that you can play cards from these sources:\n"
                "  1. Your **hand**.\n"
                "  2. The **top card of your payoff pile**.\n"
                "  3. The **top card of any of your discard piles**.\n\n"
            )
            self.state.add_observation(from_id=ta.GAME_ID, to_id=player_id, message=message)

    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """
        Generate the player prompt.

        Args:
            player_id (int): ID of the player.

        Returns:
            str: Player prompt.
        """
        prompt = (
            f"You are Player {player_id} in a two-player game of Spite and Malice. Your goal is to be the first to empty your payoff pile.\n\n"
            
            "### Game Overview:\n"
            "- The objective is to clear your payoff pile by playing cards to the center piles.\n"
            "- You can play cards from three sources:\n"
            "  1. Your **hand** (you start each turn with up to 5 cards in hand).\n"
            "  2. The **top card of your payoff pile**.\n"
            "  3. The **top card of any of your discard piles**.\n\n"
            
            "### Playing Rules:\n"
            "- You may play a card to a center pile if it is **one rank higher** than the top card on that pile (center piles start with Ace and go up to Queen; Kings are wild - they can be played on any card but do not change the rank sequence. This means if a King is used after 4, then that King is ranked 5 and the next card must be a 6).\n"
            "- If you can't play any more cards, you must **discard a card** to one of your discard piles to end your turn.\n"
            "- If a center pile reaches Queen, it will be cleared automatically.\n"
            "- The rank order is: A=1, 2=2, ..., 9=9, J=10, Q=11, K as wild.\n\n"
            
            "### Actions:\n"
            "1. **Draw**: At the start of your turn, draw cards to fill your hand up to 5 cards. Enter **[draw]** to begin.\n"
            "2. **Play a Card**: To play a card, specify the card and the center pile like this: **[play A♠ 0]** (where 'A♠' is the card and '0' is the center pile index).\n"
            "3. **Discard**: If you can’t play any more cards, discard a card from your hand to a discard pile to end your turn. Enter **[discard A♠ 1]** (where 'A♠' is the card and '1' is the discard pile index). Note that you cannot discard any card from the payoff pile. You may only discard the cards from your hand.\n\n"
            
            "Here is the current game state:\n"
        )
        prompt += self._render_board(player_id=player_id)
        prompt += "\n\nPlayer 0, you will start first. Please enter your action in the format [action card center_index]."
        return prompt
    
    def _play_card(self, player_id: int, card: str, center_index: int):
        """ Play a card from hand, payoff pile, or discard pile to a center pile """
        # Check if the card can be played on the specified center pile
        if self._can_play_on_center(card, self.center_piles[center_index]):
            # Check if the card is the top card of the payoff pile first
            if self.players[player_id]["payoff"] and card == self.players[player_id]["payoff"][-1]:
                self.players[player_id]["payoff"].pop()
            # Check if the card is in the player's hand
            elif card in self.players[player_id]["hand"]:
                self.players[player_id]["hand"].remove(card)
            # Check if the card is the top card of any discard pile
            else:
                found_in_discard = False
                for discard_pile in self.players[player_id]["discard"]:
                    if discard_pile and discard_pile[-1] == card:
                        discard_pile.pop()
                        found_in_discard = True
                        break
                if not found_in_discard:
                    return False  # Exit if the card was not in any valid pile

            # Add the card to the center pile
            self.center_piles[center_index].append(card)
            # Check if the center pile has reached Queen and clear it if so
            if len(self.center_piles[center_index]) == 11:
                self.center_piles[center_index] = []
            return True
        # If the card could not be played, return False
        return False

    def _can_play_on_center(self, card: str, pile: List[str]):
        """
        Determine if a card can be played on a center pile.
        Note that king cards are wild and can be played on any card, e.g. [play K♠ 0].
        """
        # Allow King to be played as a wild card in any position
        if card[0] == "K":
            return True
        # If the pile is empty, allow an Ace or King to start it
        if not pile:
            return card[0] == "A" or card[0] == "K"
        # If the top card of the pile is a King, treat it as the next rank in sequence
        if pile[-1][0] == "K":
            # Get the rank the King is substituting by assuming it’s the next rank in sequence
            top_card_rank = len(pile) -1 if len(pile) >= 1 else 0  # Treat as '1' if K is the only card
        else:
            # Otherwise, use the actual rank of the top card
            top_card_rank = self._card_rank(pile[-1][0])
        # Check if the played card is one rank higher than the top card or King-replaced rank
        return self._card_rank(card[0]) == top_card_rank + 1
    
    def _card_rank(self, card: str):
        """ Define the rank order (A=1, 2=2, ..., Q=12, K as wild) """
        ranks = "A23456789JQK"
        return ranks.index(card[0])
    
    def _discard_card(self, player_id: int, card: str, discard_index: int):
        """ Discard a card to one of the player's discard piles """
        self.players[player_id]["hand"].remove(card)
        self.players[player_id]["discard"][discard_index].append(card)

    def _check_win(self, player_id: int):
        """ Check if the player's payoff pile is empty, which means they won """
        return len(self.players[player_id]["payoff"]) == 0
        
    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """
        Process the player's action.
        
        Args:
            action (str): The action taken by the player.
            
        Returns:
            bool: done.
            Info: Additional information about the game state
        """

        player_id = self.state.current_player_id

        ## update the observation
        self.state.add_observation(from_id=player_id, to_id=player_id, message=action)

        ## action search pattern
        matches = self.action_space(player_id).findall(action)
        ## Let's allow for the player to parse multiple actions 

        rotate_player  = False

        if not matches:
            reason=f"Invalid move format. Player {player_id} did not respond with a valid move in square brackets."
            self.state.set_invalid_move(player_id=player_id, reason=reason)
            rotate_player  = True
        else:
            ## at least one action is matched. Let's process them.
            for match in matches:
                action_type, card, index = match
                if action_type == "draw":
                    self._draw_cards(player_id)
                    message=f"You drew cards. Your updated view:\n{self._render_board(player_id=player_id)}"
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=player_id, message=message)
                    message=f"Player {player_id} drew cards."
                    self.state.add_observation(from_id=ta.GAME_ID, to_id=1-player_id, message=message)

                elif action_type == "play":
                    ## check if the player has the card in hand or payoff pile or discard pile
                    if self._play_card(player_id, card, int(index)):
                        message=f"You played {card} on center pile {index}. Your updated view:\n{self._render_board(player_id=player_id)}"
                        self.state.add_observation(from_id=ta.GAME_ID, to_id=player_id, message=message)
                        message=f"Player {player_id} played {card} on center pile {index}."
                        self.state.add_observation(from_id=ta.GAME_ID, to_id=1-player_id, message=message)
                    else:
                        reason=f"Invalid play. Player {player_id} tried to play {card} on center pile {index}."
                        self.state.set_invalid_move(player_id=player_id, reason=reason)
                        break
                elif action_type == "discard":
                    ## player is discarding a card, which also ends the players turn
                    if card == self.players[player_id]["payoff"][-1] and card not in self.players[player_id]["hand"]:
                        reason=f"Invalid discard. Player {player_id} tried to discard a card from the payoff pile."
                        self.state.set_invalid_move(player_id=player_id, reason=reason)
                        break
                    elif card not in self.players[player_id]["hand"]:
                        reason=f"Invalid discard. Player {player_id} tried to discard a card that is not in hand."
                        self.state.set_invalid_move(player_id=player_id, reason=reason)
                        break
                    else:
                        self._discard_card(player_id, card, int(index))
                        message=f"You have discarded {card} to discard pile {index}, which also means you have finished their turn. Your updated view:\n{self._render_board(player_id=player_id)}\n\nNo further actions of yours is considered for this turn. Player {1 - player_id} will go next.", # TODO - can probably improve this message.
                        self.state.add_observation(from_id=ta.GAME_ID, to_id=player_id, message=message)
                        message=f"Player {player_id} discarded {card} to discard pile {index}, which also means they finished their turn. No further actions of Player {player_id} is considered for this turn. Player {1 - player_id}, you will go next. Please enter your action in the format [action card center_index].", # TODO - can probably improve this message.
                        self.state.add_observation(from_id=ta.GAME_ID, to_id=-1, message=message)
                        rotate_player  = True
                        self.state.game_state["player_turn"] = 1 - player_id
                        break
                else:
                    reason=f"Invalid move type. Player {player_id} did not respond with a valid move type."
                    self.state.set_invalid_move(player_id=player_id, reason=reason)
                    break
        
        ## udpate the rendered board game state
        self.state.game_state["rendered_board"] = self._render_board()

        ## check if the game is over
        if self._check_win(player_id):
            reason=f"Player {player_id} has finished its payoff pile! Player {player_id} wins!"
            self.state.set_winners(player_ids=[player_id], reason=reason)  
        return self.state.step(rotate_player)        
    
    def _render_board(self, player_id: Optional[int] = None) -> str:
        """ Render the game board """
        board = "--- Center Piles ---\n"
        for i, pile in enumerate(self.center_piles):
            board += f"Pile {i}: {pile}\n"
        
        if player_id is not None:
            board += f"\n--- Player {player_id}'s View ---\n"
            board += f"Payoff Pile (Top Card): {self.players[player_id]['payoff'][-1] if self.players[player_id]['payoff'] else 'Empty'}, Payoff Pile Length: {len(self.players[player_id]['payoff'])}\n"
            board += f"Hand: {self.players[player_id]['hand']}\n"
            board += f"Discard Piles: {self.players[player_id]['discard']}\n"

        else: 
            for player in self.players:
                board += f"\n--- Player {player}'s View ---\n"
                board += f"Payoff Pile (Top Card): {self.players[player]['payoff'][-1] if self.players[player]['payoff'] else 'Empty'}, Payoff Pile Length: {len(self.players[player]['payoff'])}\n"
                board += f"Hand: {self.players[player]['hand']}\n"
                board += f"Discard Piles: {self.players[player]['discard']}\n"
        return board
