""" Register all environments """

from textarena.utils.jury import OpenRouterJury
from textarena.envs.registration import register


# Mastermind (single-player)
register(id="Mastermind-v0", entry_point="textarena.envs.Mastermind.env:MastermindEnv", code_length=4, num_numbers=6, max_turns=20, duplicate_numbers=False)
register(id="Mastermind-v0-hard", entry_point="textarena.envs.Mastermind.env:MastermindEnv", code_length=4, num_numbers=8, max_turns=30, duplicate_numbers=False)
register(id="Mastermind-v0-extreme", entry_point="textarena.envs.Mastermind.env:MastermindEnv", code_length=6, num_numbers=12, max_turns=50, duplicate_numbers=True)

# Crosswords (single-player)
register(id="Crosswords-v0", entry_point="textarena.envs.Crosswords.env:CrosswordsEnv", hardcore=False, max_turns=30, num_words=3)
register(id="Crosswords-v0-hardcore", entry_point="textarena.envs.Crosswords.env:CrosswordsEnv", hardcore=True, max_turns=30, num_words=3)

# FifteenPuzzle (single-player)
register(id="FifteenPuzzle-v0", entry_point="textarena.envs.FifteenPuzzle.env:FifteenPuzzleEnv", max_turns=50)

# GuessTheNumber (single-player)
register(id="GuessTheNumber-v0", entry_point="textarena.envs.GuessTheNumber.env:GuessTheNumberEnv", min_number=1, max_number=20, max_turns=10)
register(id="GuessTheNumber-v0-hardcore", entry_point="textarena.envs.GuessTheNumber.env:GuessTheNumberEnv", min_number=1, max_number=50, max_turns=7)

# GuessWho (single-player)
register(id="GuessWho-v0", entry_point="textarena.envs.GuessWho.env:GuessWhoEnv", max_turns=20)

# Hangman (single-player)
register(id="Hangman-v0", entry_point="textarena.envs.Hangman.env:HangmanEnv", hardcore=False)
register(id="Hangman-v0-hardcore", entry_point="textarena.envs.Hangman.env:HangmanEnv", hardcore=True)

# LogicPuzzle (single-player)
register(id="LogicPuzzle-v0", entry_point="textarena.envs.LogicPuzzle.env:LogicPuzzleEnv", difficulty="easy")
register(id="LogicPuzzle-v0-hard", entry_point="textarena.envs.LogicPuzzle.env:LogicPuzzleEnv", difficulty="hard")

# Minesweeper (single-player)
register(id="Minesweeper-v0", entry_point="textarena.envs.Minesweeper.env:MinesweeperEnv", rows=8, cols=8, num_mines=10, max_turns=100)
register(id="Minesweeper-v0-medium", entry_point="textarena.envs.Minesweeper.env:MinesweeperEnv", rows=10, cols=10, num_mines=20, max_turns=100)
register(id="Minesweeper-v0-hard", entry_point="textarena.envs.Minesweeper.env:MinesweeperEnv", rows=12, cols=12, num_mines=30, max_turns=100)

# Sudoku (single-player)
register(id="Sudoku-v0", entry_point="textarena.envs.Sudoku.env:SudokuEnv", clues=30, max_turns=100)
register(id="Sudoku-v0-medium", entry_point="textarena.envs.Sudoku.env:SudokuEnv", clues=40, max_turns=100)
register(id="Sudoku-v0-hard", entry_point="textarena.envs.Sudoku.env:SudokuEnv", clues=50, max_turns=100)

# TowerOfHanoi (single-player)
register(id="TowerOfHanoi-v0", entry_point="textarena.envs.TowerOfHanoi.env:TowerOfHanoiEnv", num_disks=3, max_turns=100)
register(id="TowerOfHanoi-v0-medium", entry_point="textarena.envs.TowerOfHanoi.env:TowerOfHanoiEnv", num_disks=4, max_turns=100)
register(id="TowerOfHanoi-v0-hard", entry_point="textarena.envs.TowerOfHanoi.env:TowerOfHanoiEnv", num_disks=5, max_turns=100)
register(id="TowerOfHanoi-v0-extreme", entry_point="textarena.envs.TowerOfHanoi.env:TowerOfHanoiEnv", num_disks=7, max_turns=100)

# TwentyQuestions (single-player)
register(id="TwentyQuestions-v0", entry_point="textarena.envs.TwentyQuestions.env:TwentyQuestionsEnv", hardcore=False)
register(id="TwentyQuestions-v0-hardcore", entry_point="textarena.envs.TwentyQuestions.env:TwentyQuestionsEnv", hardcore=True)

# WordLadder (single-player)
register(id="WordLadder-v0", entry_point="textarena.envs.WordLadder.env:WordLadderEnv", min_distance=5, max_distance=7, max_turns=100)
register(id="WordLadder-v0-medium", entry_point="textarena.envs.WordLadder.env:WordLadderEnv", min_distance=8, max_distance=12, max_turns=100)
register(id="WordLadder-v0-hard", entry_point="textarena.envs.WordLadder.env:WordLadderEnv", min_distance=13, max_distance=15, max_turns=100)

# Wordle (single-player)
register(id="Wordle-v0", entry_point="textarena.envs.Wordle.env:WordleEnv", hardcore=False, word_length=5, num_guesses=6)
register(id="Wordle-v0-hardcore", entry_point="textarena.envs.Wordle.env:WordleEnv", hardcore=True, word_length=5, num_guesses=6)
register(id="Wordle-v0-long", entry_point="textarena.envs.Wordle.env:WordleEnv", hardcore=False, word_length=7, num_guesses=9)
register(id="Wordle-v0-long-hardcore", entry_point="textarena.envs.Wordle.env:WordleEnv", hardcore=True, word_length=7, num_guesses=9)

# WordSearch (single-player)
register(id="WordSearch-v0", entry_point="textarena.envs.WordSearch.env:WordSearchEnv", hardcore=False)
register(id="WordSearch-v0-hardcore", entry_point="textarena.envs.WordSearch.env:WordSearchEnv", hardcore=True)




# Battleship (two-player)
register(id="Battleship-v0", entry_point="textarena.envs.Battleship.env:BattleshipEnv", grid_size=5)
register(id="Battleship-v0-standard", entry_point="textarena.envs.Battleship.env:BattleshipEnv", grid_size=10)
register(id="Battleship-v0-large", entry_point="textarena.envs.Battleship.env:BattleshipEnv", grid_size=14)
register(id="Battleship-v0-extreme", entry_point="textarena.envs.Battleship.env:BattleshipEnv", grid_size=20)

# Coup (2-6 player)
register(id="Coup-v0", entry_point="textarena.envs.Coup.env:CoupEnv")

# Breakthrough (two-player)
register(id="Breakthrough-v0", entry_point="textarena.envs.Breakthrough.env:BreakthroughEnv", board_size=8, max_turns=100, is_open=True)
register(id="Breakthrough-v0-small", entry_point="textarena.envs.Breakthrough.env:BreakthroughEnv", board_size=6, max_turns=80, is_open=True)
register(id="Breakthrough-v0-large", entry_point="textarena.envs.Breakthrough.env:BreakthroughEnv", board_size=10, max_turns=120, is_open=True)
register(id="Breakthrough-v0-blind", entry_point="textarena.envs.Breakthrough.env:BreakthroughEnv", board_size=8, max_turns=100, is_open=False)
register(id="Breakthrough-v0-long", entry_point="textarena.envs.Breakthrough.env:BreakthroughEnv", board_size=8, max_turns=200, is_open=True)


# Chess (two-player)
register(id="Chess-v0", entry_point="textarena.envs.Chess.env:ChessEnv", is_open=True, max_turns=100, show_valid=True)
register(id="Chess-v0-long", entry_point="textarena.envs.Chess.env:ChessEnv", is_open=True, max_turns=250, show_valid=True)
register(id="Chess-v0-blind", entry_point="textarena.envs.Chess.env:ChessEnv", is_open=False, max_turns=150, show_valid=False)

# Checkers (two-player)
register(id="Checkers-v0", entry_point="textarena.envs.Checkers.env:CheckersEnv", max_turns=100)
register(id="Checkers-v0-long", entry_point="textarena.envs.Checkers.env:CheckersEnv", max_turns=300)

# ConnectFour (two-player)
register(id="ConnectFour-v0", entry_point="textarena.envs.ConnectFour.env:ConnectFourEnv", is_open=True, num_rows=6, num_cols=7)
register(id="ConnectFour-v0-blind", entry_point="textarena.envs.ConnectFour.env:ConnectFourEnv", is_open=False, num_rows=6, num_cols=7)
register(id="ConnectFour-v0-large", entry_point="textarena.envs.ConnectFour.env:ConnectFourEnv", is_open=True, num_rows=12, num_cols=15)

# DontSayIt (two-player)
register(id="DontSayIt-v0", entry_point="textarena.envs.DontSayIt.env:DontSayItEnv", hardcore=False, max_turns=20)
register(id="DontSayIt-v0-hardcore", entry_point="textarena.envs.DontSayIt.env:DontSayItEnv", hardcore=True, max_turns=30)
register(id="DontSayIt-v0-unlimited", entry_point="textarena.envs.DontSayIt.env:DontSayItEnv", hardcore=False, max_turns=None)

# KuhnPoker (two-player)
register(id="KuhnPoker-v0", entry_point="textarena.envs.KuhnPoker.env:KuhnPokerEnv", ante=1, max_rounds=10)
register(id="KuhnPoker-v0-long", entry_point="textarena.envs.KuhnPoker.env:KuhnPokerEnv", ante=1, max_rounds=15)
register(id="KuhnPoker-v0-blind", entry_point="textarena.envs.KuhnPoker.env:KuhnPokerEnv", ante=1, max_rounds=10)
register(id="KuhnPoker-v0-highstakes", entry_point="textarena.envs.KuhnPoker.env:KuhnPokerEnv", ante=5, max_rounds=10)
register(id="KuhnPoker-v0-extended", entry_point="textarena.envs.KuhnPoker.env:KuhnPokerEnv", ante=1, max_rounds=30)

# LetterAuction (two-player)
register(id="LetterAuction-v0", entry_point="textarena.envs.LetterAuction.env:LetterAuctionEnv", starting_coins=100)
register(id="LetterAuction-v0-medium", entry_point="textarena.envs.LetterAuction.env:LetterAuctionEnv", starting_coins=50)
register(id="LetterAuction-v0-hard", entry_point="textarena.envs.LetterAuction.env:LetterAuctionEnv", starting_coins=25)

# MemoryGame (two-player)
register(id="MemoryGame-v0", entry_point="textarena.envs.MemoryGame.env:MemoryGameEnv", grid_size=4)
register(id="MemoryGame-v0-medium", entry_point="textarena.envs.MemoryGame.env:MemoryGameEnv", grid_size=6)
register(id="MemoryGame-v0-hard", entry_point="textarena.envs.MemoryGame.env:MemoryGameEnv", grid_size=8)

# Nim (two-player)
register(id="Nim-v0", entry_point="textarena.envs.Nim.env:NimEnv", piles=[3, 4, 5])
register(id="Nim-v0-small", entry_point="textarena.envs.Nim.env:NimEnv", piles=[1, 2, 3])
register(id="Nim-v0-large", entry_point="textarena.envs.Nim.env:NimEnv", piles=[5, 7, 9])

# Othello/Reversi (two-player)
register(id="Othello-v0", entry_point="textarena.envs.Othello.env:OthelloEnv", max_turns=60, show_valid=True)
register(id="Othello-v0-hard", entry_point="textarena.envs.Othello.env:OthelloEnv", max_turns=60, show_valid=False)

# Pig (two-player)
register(id="PigDice-v0", entry_point="textarena.envs.PigDice.env:PigDiceEnv", winning_score=100, max_turns=100)
register(id="PigDice-v0-short", entry_point="textarena.envs.PigDice.env:PigDiceEnv", winning_score=50, max_turns=50)
register(id="PigDice-v0-long", entry_point="textarena.envs.PigDice.env:PigDiceEnv", winning_score=500, max_turns=500)


# ScenarioPlanning (two-player)
register(id="ScenarioPlanning-v0", entry_point="textarena.envs.ScenarioPlanning.env:ScenarioPlanningEnv", jury_class=OpenRouterJury, jury_size=11)

# SpellingBee (two-player)
register(id="SpellingBee-v0", entry_point="textarena.envs.SpellingBee.env:SpellingBeeEnv", num_letters=7)
register(id="SpellingBee-v0-small", entry_point="textarena.envs.SpellingBee.env:SpellingBeeEnv", num_letters=4)
register(id="SpellingBee-v0-large", entry_point="textarena.envs.SpellingBee.env:SpellingBeeEnv", num_letters=10)

# Taboo (two-player)
register(id="Taboo-v0", entry_point="textarena.envs.Taboo.env:TabooEnv", max_turns=6, categories=["things"])
register(id="Taboo-v0-animals", entry_point="textarena.envs.Taboo.env:TabooEnv", max_turns=6, categories=["animals"])
register(id="Taboo-v0-cars", entry_point="textarena.envs.Taboo.env:TabooEnv", max_turns=6, categories=["cars"])
register(id="Taboo-v0-city/country", entry_point="textarena.envs.Taboo.env:TabooEnv", max_turns=6, categories=["city/country"])
register(id="Taboo-v0-food", entry_point="textarena.envs.Taboo.env:TabooEnv", max_turns=6, categories=["food"])
register(id="Taboo-v0-literature", entry_point="textarena.envs.Taboo.env:TabooEnv", max_turns=6, categories=["literature"])
register(id="Taboo-v0-people", entry_point="textarena.envs.Taboo.env:TabooEnv", max_turns=6, categories=["people"])
register(id="Taboo-v0-tv", entry_point="textarena.envs.Taboo.env:TabooEnv", max_turns=6, categories=["tv"])
register(id="Taboo-v0-long", entry_point="textarena.envs.Taboo.env:TabooEnv", max_turns=24, categories=["things"])
register(id="Taboo-v0-full", entry_point="textarena.envs.Taboo.env:TabooEnv", max_turns=6, categories=["animals", "cars", "city/country", "food", "literature", "people", "things", "tv"])

# TicTacToe (two-player)
register(id="TicTacToe-v0", entry_point="textarena.envs.TicTacToe.env:TicTacToeEnv")

# WildTicTacToe (two-player)
register(id="WildTicTacToe-v0", entry_point="textarena.envs.WildTicTacToe.env:WildTicTacToeEnv")

# ReverseTicTacToe (two-player)
register(id="ReverseTicTacToe-v0", entry_point="textarena.envs.ReverseTicTacToe.env:ReverseTicTacToeEnv")

# RandomizedTicTacToe (two-player)
register(id="RandomizedTicTacToe-v0", entry_point="textarena.envs.RandomizedTicTacToe.env:RandomizedTicTacToeEnv")

# QuantumTicTacToe (two-player)
register(id="QuantumTicTacToe-v0", entry_point="textarena.envs.QuantumTicTacToe.env:QuantumTicTacToeEnv")

# IteratedPrisonersDilemma (two-player)
register(id="IteratedPrisonersDilemma-v0", entry_point="textarena.envs.IteratedPrisonersDilemma.env:IteratedPrisonersDilemmaEnv", num_rounds=10, communication_turns=3, cooperate_reward=3, defect_reward=5, sucker_reward=0, mutual_defect_reward=1)

# IteratedRockPaperScissors (two-player)
register(id="IteratedRockPaperScissors-v0", entry_point="textarena.envs.IteratedRockPaperScissors.env:IteratedRockPaperScissorsEnv", num_rounds=10)

# Stratego (two-player)
register(id="Stratego-v0", entry_point="textarena.envs.Stratego.env:StrategoEnv")

# SpiteAndMalice (two-player)
register(id="SpiteAndMalice-v0", entry_point="textarena.envs.SpiteAndMalice.env:SpiteAndMaliceEnv")

# Tak (two-player)
register(id="Tak-v0", entry_point="textarena.envs.Tak.env:TakEnv", board_size=4, stones=15, capstones=1)
register(id="Tak-v0-medium", entry_point="textarena.envs.Tak.env:TakEnv", board_size=5, stones=21, capstones=1)
register(id="Tak-v0-hard", entry_point="textarena.envs.Tak.env:TakEnv", board_size=6, stones=30, capstones=1)

# SimpleTak (two-player)
register(id="SimpleTak-v0", entry_point="textarena.envs.SimpleTak.env:SimpleTakEnv", board_size=4)
register(id="SimpleTak-v0-medium", entry_point="textarena.envs.SimpleTak.env:SimpleTakEnv", board_size=5)
register(id="SimpleTak-v0-large", entry_point="textarena.envs.SimpleTak.env:SimpleTakEnv", board_size=6)
register(id="SimpleTak-v0-extra-large", entry_point="textarena.envs.SimpleTak.env:SimpleTakEnv", board_size=8)

# TruthAndDeception (two-player) [TODO can extend to more players]
register(id="TruthAndDeception-v0", entry_point="textarena.envs.TruthAndDeception.env:TruthAndDeceptionEnv", max_turns=6)
register(id="TruthAndDeception-v0-long", entry_point="textarena.envs.TruthAndDeception.env:TruthAndDeceptionEnv", max_turns=12)
register(id="TruthAndDeception-v0-extreme", entry_point="textarena.envs.TruthAndDeception.env:TruthAndDeceptionEnv", max_turns=50)

# UltimateTicTacToe (two-player)
register(id="UltimateTicTacToe-v0", entry_point="textarena.envs.UltimateTicTacToe.env:UltimateTicTacToeEnv")

# WordChains (two-player)
register(id="WordChains-v0", entry_point="textarena.envs.WordChains.env:WordChainsEnv")

# Debate (two-player)
register(id="Debate-v0", entry_point="textarena.envs.Debate.env:DebateEnv", max_turns=6, jury_class=OpenRouterJury, jury_size=7)
register(id="Debate-v0-medium", entry_point="textarena.envs.Debate.env:DebateEnv", max_turns=12, jury_class=OpenRouterJury, jury_size=9)
register(id="Debate-v0-long", entry_point="textarena.envs.Debate.env:DebateEnv", max_turns=30, jury_class=OpenRouterJury, jury_size=13)

# SimpleNegotiation (two-player)
register(id="SimpleNegotiation-v0", entry_point="textarena.envs.SimpleNegotiation.env:SimpleNegotiationEnv", max_turns=10)
register(id="SimpleNegotiation-v0-short", entry_point="textarena.envs.SimpleNegotiation.env:SimpleNegotiationEnv", max_turns=6)
register(id="SimpleNegotiation-v0-long", entry_point="textarena.envs.SimpleNegotiation.env:SimpleNegotiationEnv", max_turns=30)

# SimpleBlindAunction (two-player)
register(id="SimpleBlindAuction-v0", entry_point="textarena.envs.SimpleBlindAuction.env:SimpleBlindAuctionEnv", starting_capital=1000, num_items=5, conversation_rounds=3)
register(id="SimpleBlindAuction-v0-quick", entry_point="textarena.envs.SimpleBlindAuction.env:SimpleBlindAuctionEnv", starting_capital=750, num_items=3, conversation_rounds=1)
register(id="SimpleBlindAuction-v0-rich", entry_point="textarena.envs.SimpleBlindAuction.env:SimpleBlindAuctionEnv", starting_capital=2000,  num_items=5, conversation_rounds=5)


# Negotiation (2-15 players)
register(id="Negotiation-v0", entry_point="textarena.envs.Negotiation.env:NegotiationEnv", turn_multiple=8)
register(id="Negotiation-v0-long", entry_point="textarena.envs.Negotiation.env:NegotiationEnv", turn_multiple=15)

# BlindAuction (3-15 players)
register(id="BlindAuction-v0", entry_point="textarena.envs.BlindAuction.env:BlindAuctionEnv", starting_capital=1000, num_items=5, conversation_rounds=3)
register(id="BlindAuction-v0-high", entry_point="textarena.envs.BlindAuction.env:BlindAuctionEnv", starting_capital=2500, num_items=8, conversation_rounds=5)
register(id="BlindAuction-v0-fast", entry_point="textarena.envs.BlindAuction.env:BlindAuctionEnv", starting_capital=750,  num_items=3, conversation_rounds=1)
register(id="BlindAuction-v0-complex", entry_point="textarena.envs.BlindAuction.env:BlindAuctionEnv", starting_capital=1500, num_items=12, conversation_rounds=8)

# Snake (2-15 players)
register(id="Snake-v0", entry_point="textarena.envs.Snake.env:SnakeEnv", width=5, height=5, num_apples=2, max_turns=40)
register(id="Snake-v0-standard", entry_point="textarena.envs.Snake.env:SnakeEnv", width=10, height=10, num_apples=3, max_turns=100)
register(id="Snake-v0-large", entry_point="textarena.envs.Snake.env:SnakeEnv", width=15, height=15, num_apples=5, max_turns=250)

# Surround (2-15 players)
register(id="Surround-v0", entry_point="textarena.envs.Surround.env:SurroundEnv", width=5, height=5, max_turns=40)
register(id="Surround-v0-large", entry_point="textarena.envs.Surround.env:SurroundEnv", width=10, height=10, max_turns=100)
register(id="Surround-v0-huge", entry_point="textarena.envs.Surround.env:SurroundEnv", width=15, height=15, max_turns=250)


# LiarsDice (2-15 players)
register(id="LiarsDice-v0", entry_point="textarena.envs.LiarsDice.env:LiarsDiceEnv", num_dice=5)
register(id="LiarsDice-v0-large", entry_point="textarena.envs.LiarsDice.env:LiarsDiceEnv", num_dice=12)

# Poker (2-15 players)
register(id="Poker-v0", entry_point="textarena.envs.Poker.env:PokerEnv", num_rounds=10, starting_chips=1_000, small_blind=10, big_blind=20)
register(id="Poker-v0-long", entry_point="textarena.envs.Poker.env:PokerEnv", num_rounds=15, starting_chips=1_000, small_blind=10, big_blind=20)
register(id="Poker-v0-extreme", entry_point="textarena.envs.Poker.env:PokerEnv", num_rounds=50, starting_chips=1_000, small_blind=10, big_blind=20)

# Character Conclave (3-15 players)
register(id="CharacterConclave-v0", entry_point="textarena.envs.CharacterConclave.env:CharacterConclaveEnv", character_budget=1_000)
register(id="CharacterConclave-v0-long", entry_point="textarena.envs.CharacterConclave.env:CharacterConclaveEnv", character_budget=5_000)
register(id="CharacterConclave-v0-extreme", entry_point="textarena.envs.CharacterConclave.env:CharacterConclaveEnv", character_budget=10_000)

# Diplomacy (3-7 players)
register(id="Diplomacy-v0", entry_point="textarena.envs.Diplomacy.env:DiplomacyEnv", max_turns=1_000)

# SecretMafia (5-15 players)
register(id="SecretMafia-v0", entry_point="textarena.envs.SecretMafia.env:SecretMafiaEnv", mafia_ratio=0.25, discussion_rounds=3)

# TwoRoomsAndABoom (6-20 players)
register(id="TwoRoomsAndABoom-v0", entry_point="textarena.envs.TwoRoomsAndABoom.env:TwoRoomsAndABoomEnv", num_rounds=3, cards_per_room=3, discussion_rounds=2)

# Codenames (4 players)
register(id="Codenames-v0", entry_point="textarena.envs.Codenames.env:CodenamesEnv", hardcore=False) 
register(id="Codenames-v0-hardcore", entry_point="textarena.envs.Codenames.env:CodenamesEnv", hardcore=True) 

# EmojiCharade (4 players)
register(id="EmojiCharade-v0", entry_point="textarena.envs.EmojiCharade.env:EmojiCharadeEnv") 

# ThreePlayerTicTacToe (3 players)
register(id="ThreePlayerTicTacToe-v0", entry_point="textarena.envs.ThreePlayerTicTacToe.env:ThreePlayerTicTacToeEnv")

# classical evals as single-player envs

# GSM8K - Grade School Math Word Problems
register(id="GSM8K-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="gsm8k/test.jsonl", n_samples=None)
register(id="GSM8K-v0-subsampled", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="gsm8k/test.jsonl", n_samples=100)
register(id="GSM8K-v0-pass@16", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="gsm8k/test.jsonl", n_samples=None, eval_method="x@k", k=16)

# AIME - American Invitational Mathematics Examination
register(id="AIME-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="aime24/test.jsonl", n_samples=None)
register(id="AIME-v0-pass@16", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="aime24/test.jsonl", n_samples=None, eval_method="x@k", k=16)
register(id="AIME23-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="aime23/test.jsonl", n_samples=None)

# AQUA - Algebra Question Answering
register(id="AQUA-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="aqua/test.jsonl", n_samples=None)
register(id="AQUA-v0-pass@16", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="aqua/test.jsonl", n_samples=None, eval_method="x@k", k=16)

# ASDIV - Arithmetic with diverse operations
register(id="ASDIV-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="asdiv/test.jsonl", n_samples=None)

# CARP - Complex Arithmetic Problems
register(id="CARP_EN-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="carp_en/test.jsonl", n_samples=None)

# CMATH - College Mathematics
register(id="CMATH-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="cmath/test.jsonl", n_samples=None)
register(id="CMATH-v0-pass@16", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="cmath/test.jsonl", n_samples=None, eval_method="x@k", k=16)

# Chinese Middle School Math
register(id="CN_MIDDLE_SCHOOL-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="cn_middle_school/test.jsonl", n_samples=None)

# College Math
register(id="COLLEGE_MATH-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="college_math/test.jsonl", n_samples=None)
register(id="COLLEGE_MATH-v0-pass@16", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="college_math/test.jsonl", n_samples=None, eval_method="x@k", k=16)

# GAOKAO Math
register(id="GAOKAO_MATH-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="gaokao_math/test.jsonl", n_samples=None)
register(id="GAOKAO_MATH_QA-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="gaokao_math_qa/test.jsonl", n_samples=None)
register(id="GAOKAO2023EN-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="gaokao2023en/test.jsonl", n_samples=None)
register(id="GAOKAO2024_I-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="gaokao2024_I/test.jsonl", n_samples=None)
register(id="GAOKAO2024_II-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="gaokao2024_II/test.jsonl", n_samples=None)
register(id="GAOKAO2024_MCQ-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="gaokao2024_mcq/test.jsonl", n_samples=None)

# MATH
register(id="MATH-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="math/test.jsonl", n_samples=None)
register(id="MATH-v0-pass@16", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="math/test.jsonl", n_samples=None, eval_method="x@k", k=16)
register(id="MATH-v0-subsampled", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="math/test.jsonl", n_samples=100)

# MATH500
register(id="MATH500-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="math500/test.jsonl", n_samples=None)

# MAWPS - Math Word Problems
register(id="MAWPS-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="mawps/test.jsonl", n_samples=None)

# Minerva Math
register(id="MINERVA_MATH-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="minerva_math/test.jsonl", n_samples=None)

# MMLU STEM
register(id="MMLU_STEM-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="mmlu_stem/test.jsonl", n_samples=None)

# Math Olympiad Benchmark
register(id="OLYMPIADBENCH-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="olympiadbench/test.jsonl", n_samples=None)
register(id="OLYMPIADBENCH-v0-pass@16", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="olympiadbench/test.jsonl", n_samples=None, eval_method="x@k", k=16)

# SAT Math
register(id="SAT_MATH-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="sat_math/test.jsonl", n_samples=None)

# SVAMP
register(id="SVAMP-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="svamp/test.jsonl", n_samples=None)

# TabMWP - Tabular Math Word Problems
register(id="TABMWP-v0", entry_point="textarena.envs.ClassicalReasoningEvals.env:ClassicalReasoningEvalsEnv", file_name="tabmwp/test.jsonl", n_samples=None)
