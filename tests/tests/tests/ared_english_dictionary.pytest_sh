"""Regression tests for the shared EnglishDictionary factory (issue #211)."""

import contextlib
import io

import textarena as ta
from textarena.envs.utils.word_lists import (
    EnglishDictionary,
    shared_english_dictionary,
)


def _quiet(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def test_word_game_envs_share_one_dictionary_instance():
    wordle = _quiet(lambda: ta.make("Wordle-v0"))
    spelling_bee = _quiet(lambda: ta.make("SpellingBee-v0"))
    word_chains = _quiet(lambda: ta.make("WordChains-v0"))
    word_ladder = _quiet(lambda: ta.make("WordLadder-v0"))
    assert wordle.dictionary is spelling_bee.dictionary is word_chains.dictionary
    # WordLadder consumes the words rather than the object; contents must match.
    assert word_ladder.universal_word_list == wordle.dictionary.get_all_words()


def test_shared_instance_matches_fresh_build():
    fresh = _quiet(lambda: EnglishDictionary(keep_proper_nouns=False, include_nltk=True))
    shared = shared_english_dictionary(keep_proper_nouns=False, include_nltk=True)
    assert shared.get_all_words() == fresh.get_all_words()
    assert shared.get_basic_words() == fresh.get_basic_words()
    assert shared.is_english_word("apple")
    assert not shared.is_english_word("zzqqxv")


def test_cache_is_keyed_by_parameters():
    default = shared_english_dictionary(keep_proper_nouns=False, include_nltk=True)
    proper = shared_english_dictionary(keep_proper_nouns=True, include_nltk=True)
    assert default is shared_english_dictionary(keep_proper_nouns=False, include_nltk=True)
    assert proper is not default


def test_shared_dictionary_does_not_leak_mutable_state():
    shared = shared_english_dictionary(keep_proper_nouns=False, include_nltk=True)
    leaked = shared.get_all_words()
    leaked.add("zzznotaword")  # mutate the returned set
    assert "zzznotaword" not in shared.get_all_words()


def test_wordle_guess_semantics_unchanged():
    env = _quiet(lambda: ta.make("Wordle-v0"))
    _quiet(lambda: env.reset(num_players=1, seed=1))
    done, _ = _quiet(lambda: env.step(action="[apple]"))
    assert not done and env.state.turn == 1  # valid word consumes a turn
    done, _ = _quiet(lambda: env.step(action="[zzqqx]"))
    assert not done and env.state.turn == 1  # invalid word does not
    done, _ = _quiet(lambda: env.step(action="nonsense"))
    assert done  # malformed action triggers invalid-move handling
