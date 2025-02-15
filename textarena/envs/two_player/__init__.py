""" Register all environments """

from textarena.envs.registration import register
from textarena.envs.utils.jury import OpenRouterJury

# Register Game sets
register(
    id="BalancedSubset-v0",
    entry_point="textarena.envs.two_player.Subsets.env:SubsetEnv",
    env_ids=[
        "DontSayIt-v0",
        "Negotiation-v0",
        "LiarsDice-v0",
        "Chess-v0",
        "TruthAndDeception-v0",
        "SpellingBee-v0",
        "Poker-v0",
        "Stratego-v0",
        "Tak-v0",
        "UltimateTicTacToe-v0",
    ],
    max_num_characters=1_000
)

# Register Individual Games
register(
    id="DontSayIt-v0",
    entry_point="textarena.envs.two_player.DontSayIt.env:DontSayItEnv",
    hardcore=False,
    max_turns=20,
)
register(
    id="DontSayIt-v0-hardcore",
    entry_point="textarena.envs.two_player.DontSayIt.env:DontSayItEnv",
    hardcore=True,
    max_turns=30,
)
register(
    id="DontSayIt-v0-unlimited",
    entry_point="textarena.envs.two_player.DontSayIt.env:DontSayItEnv",
    hardcore=False,
    max_turns=None,
)

register(
    id="Negotiation-v0",
    entry_point="textarena.envs.two_player.Negotiation.env:NegotiationEnv",
    max_turns=10,
)
register(
    id="Negotiation-v0-short",
    entry_point="textarena.envs.two_player.Negotiation.env:NegotiationEnv",
    max_turns=6,
)
register(
    id="Negotiation-v0-long",
    entry_point="textarena.envs.two_player.Negotiation.env:NegotiationEnv",
    max_turns=30,
)

register(
    id="IteratedPrisonersDilemma-v0",
    entry_point="textarena.envs.two_player.IteratedPrisonersDilemma.env:IteratedPrisonersDilemmaEnv",
    num_rounds=10,
    communication_turns=3,
    cooperate_reward=3,
    defect_reward=5,
    sucker_reward=0,
    mutual_defect_reward=1,
)

register(
    id="Chess-v0",
    entry_point="textarena.envs.two_player.Chess.env:ChessEnv",
    is_open=True,
    max_turns=100,
    show_valid=True,
)
# register(
#     id="Chess-v0-open",
#     entry_point="textarena.envs.two_player.Chess.env:ChessEnv",
#     is_open=True,
#     max_turns=100,
#     show_valid=False,
# )
register(
    id="Chess-v0-long",
    entry_point="textarena.envs.two_player.Chess.env:ChessEnv",
    is_open=False,
    max_turns=250,
    show_valid=True,
)
register(
    id="Chess-v0-blind",
    entry_point="textarena.envs.two_player.Chess.env:ChessEnv",
    is_open=False,
    max_turns=150,
    show_valid=False,
)

register(
    id="TruthAndDeception-v0",
    entry_point="textarena.envs.two_player.TruthAndDeception.env:TruthAndDeceptionEnv",
    max_turns=6,
)
register(
    id="TruthAndDeception-v0-long",
    entry_point="textarena.envs.two_player.TruthAndDeception.env:TruthAndDeceptionEnv",
    max_turns=12,
)
register(
    id="TruthAndDeception-v0-super-long",
    entry_point="textarena.envs.two_player.TruthAndDeception.env:TruthAndDeceptionEnv",
    max_turns=50,
)

register(
    id="SpellingBee-v0",
    entry_point="textarena.envs.two_player.SpellingBee.env:SpellingBeeEnv",
    num_letters=7,
)
register(
    id="SpellingBee-v0-small",
    entry_point="textarena.envs.two_player.SpellingBee.env:SpellingBeeEnv",
    num_letters=4,
)
register(
    id="SpellingBee-v0-large",
    entry_point="textarena.envs.two_player.SpellingBee.env:SpellingBeeEnv",
    num_letters=10,
)

register(
    id="Poker-v0",
    entry_point="textarena.envs.two_player.Poker.env:PokerEnv",
    num_rounds=10,
    starting_chips=1_000,
    small_blind=10,
    big_blind=20
)

register(
    id="Poker-v0-long",
    entry_point="textarena.envs.two_player.Poker.env:PokerEnv",
    num_rounds=15,
    starting_chips=1_000,
    small_blind=10,
    big_blind=20
)


register(
    id="Poker-v0-super-long",
    entry_point="textarena.envs.two_player.Poker.env:PokerEnv",
    num_rounds=50,
    starting_chips=1_000,
    small_blind=10,
    big_blind=20
)


register(
    id="Stratego-v0",
    entry_point="textarena.envs.two_player.Stratego.env:StrategoEnv"
)

register(
    id="Tak-v0-easy",
    entry_point="textarena.envs.two_player.Tak.env:TakEnv"
)

register(
    id="Tak-v0",
    entry_point="textarena.envs.two_player.Tak.env:TakEnv"
)

register(
    id="Tak-v0-hard",
    entry_point="textarena.envs.two_player.Tak.env:TakEnv"
)

register(
    id="UltimateTicTacToe-v0",
    entry_point="textarena.envs.two_player.UltimateTicTacToe.env:UltimateTicTacToeEnv"
)

register(
    id="LiarsDice-v0",
    entry_point="textarena.envs.two_player.LiarsDice.env:LiarsDiceEnv",
    num_dice=5
)

register(
    id="LiarsDice-v0-large",
    entry_point="textarena.envs.two_player.LiarsDice.env:LiarsDiceEnv",
    num_dice=12,
)

register(
    id="ConnectFour-v0",
    entry_point="textarena.envs.two_player.ConnectFour.env:ConnectFourEnv",
    is_open=True,
    num_rows=6,
    num_cols=7
)

register(
    id="ConnectFour-v0-blind",
    entry_point="textarena.envs.two_player.ConnectFour.env:ConnectFourEnv",
    is_open=False,
    num_rows=6,
    num_cols=7
)

register(
    id="ConnectFour-v0-large",
    entry_point="textarena.envs.two_player.ConnectFour.env:ConnectFourEnv",
    is_open=True,
    num_rows=12,
    num_cols=15
)

register(
    id="WordChains-v0",
    entry_point="textarena.envs.two_player.WordChains.env:WordChainsEnv",
    max_turns=100,
)
register(
    id="WordChains-v0-infinite",
    entry_point="textarena.envs.two_player.WordChains.env:WordChainsEnv",
    max_turns=None,
)



register(
    id="Debate-v0",
    entry_point="textarena.envs.two_player.Debate.env:DebateEnv",
    max_turns=6,
    jury_class=OpenRouterJury,
    jury_size=7,
)
register(
    id="Debate-v0-medium",
    entry_point="textarena.envs.two_player.Debate.env:DebateEnv",
    max_turns=12,
    jury_class=OpenRouterJury,
    jury_size=9,
)
register(
    id="Debate-v0-long",
    entry_point="textarena.envs.two_player.Debate.env:DebateEnv",
    max_turns=30,
    jury_class=OpenRouterJury,
    jury_size=13,
)

register(
    id="Battleship-v0",
    entry_point="textarena.envs.two_player.Battleship.env:BattleshipEnv",
    grid_size=10,
)

register(
    id="Battleship-v0-large",
    entry_point="textarena.envs.two_player.Battleship.env:BattleshipEnv",
    grid_size=14,
)

register(
    id="Battleship-v0-giant",
    entry_point="textarena.envs.two_player.Battleship.env:BattleshipEnv",
    grid_size=20,
)

register(
    id="SpiteAndMalice-v0",
    entry_point="textarena.envs.two_player.SpiteAndMalice.env:SpiteAndMaliceEnv",
)

register(
    id="Mastermind-v0",
    entry_point="textarena.envs.two_player.Mastermind.env:MastermindEnv",
    code_length=4,
    num_numbers=6,
    max_turns=20,
    duplicate_numbers=False,
)

register(
    id="Mastermind-v0-hard",
    entry_point="textarena.envs.two_player.Mastermind.env:MastermindEnv",
    code_length=4,
    num_numbers=8,
    max_turns=30,
    duplicate_numbers=False,
)

register(
    id="Mastermind-v0-extreme",
    entry_point="textarena.envs.two_player.Mastermind.env:MastermindEnv",
    code_length=6,
    num_numbers=12,
    max_turns=50,
    duplicate_numbers=True,
)


register(
    "LetterAuction-v0-easy",
    entry_point="textarena.envs.two_player.LetterAuction.env:LetterAuctionEnv",
    difficulty="easy"
)

register(
    "LetterAuction-v0-medium",
    entry_point="textarena.envs.two_player.LetterAuction.env:LetterAuctionEnv",
    difficulty="medium"
)

register(
    "LetterAuction-v0-hard",
    entry_point="textarena.envs.two_player.LetterAuction.env:LetterAuctionEnv",
    difficulty="hard"
)

register(
    id="ScenarioPlanning-v0",
    entry_point="textarena.envs.two_player.ScenarioPlanning.env:ScenarioPlanningEnv",
    jury_class=OpenRouterJury,
    jury_size=11,
)

register(
    "MemoryGame-v0-easy",
    entry_point="textarena.envs.two_player.MemoryGame.env:MemoryGameEnv",
    difficulty="easy",
)

register(
    "MemoryGame-v0-medium",
    entry_point="textarena.envs.two_player.MemoryGame.env:MemoryGameEnv",
    difficulty="medium",
)

register(
    "MemoryGame-v0-hard",
    entry_point="textarena.envs.two_player.MemoryGame.env:MemoryGameEnv",
    difficulty="hard",
)

register(
    id="Taboo-v0",
    entry_point="textarena.envs.two_player.Taboo.env:TabooEnv",
    max_turns=6,
    categories=["things"],
)
register(
    id="Taboo-v0-animals",
    entry_point="textarena.envs.two_player.Taboo.env:TabooEnv",
    max_turns=6,
    categories=["animals"],
)
register(
    id="Taboo-v0-cars",
    entry_point="textarena.envs.two_player.Taboo.env:TabooEnv",
    max_turns=6,
    categories=["cars"],
)
register(
    id="Taboo-v0-city/country",
    entry_point="textarena.envs.two_player.Taboo.env:TabooEnv",
    max_turns=6,
    categories=["city/country"],
)
register(
    id="Taboo-v0-food",
    entry_point="textarena.envs.two_player.Taboo.env:TabooEnv",
    max_turns=6,
    categories=["food"],
)
register(
    id="Taboo-v0-literature",
    entry_point="textarena.envs.two_player.Taboo.env:TabooEnv",
    max_turns=6,
    categories=["literature"],
)
register(
    id="Taboo-v0-people",
    entry_point="textarena.envs.two_player.Taboo.env:TabooEnv",
    max_turns=6,
    categories=["people"],
)
register(
    id="Taboo-v0-tv",
    entry_point="textarena.envs.two_player.Taboo.env:TabooEnv",
    max_turns=6,
    categories=["tv"],
)
register(
    id="Taboo-v0-long",
    entry_point="textarena.envs.two_player.Taboo.env:TabooEnv",
    max_turns=24,
    categories=["things"],
)
register(
    id="Taboo-v0-full",
    entry_point="textarena.envs.two_player.Taboo.env:TabooEnv",
    max_turns=6,
    categories=[
        "animals",
        "cars",
        "city/country",
        "food",
        "literature",
        "people",
        "things",
        "tv",
    ],
)

register(
    id="Snake-v0",
    entry_point="textarena.envs.two_player.Snake.env:TwoPlayerSnakeEnv",
    width=5,
    height=5,
    num_apples=2,
    max_turns=40
)

register(
    id="Snake-v0-standard",
    entry_point="textarena.envs.two_player.Snake.env:TwoPlayerSnakeEnv",
    width=10,
    height=10,
    num_apples=3,
    max_turns=100
)

register(
    id="Snake-v0-large",
    entry_point="textarena.envs.two_player.Snake.env:TwoPlayerSnakeEnv",
    width=15,
    height=15,
    num_apples=5,
    max_turns=200
)