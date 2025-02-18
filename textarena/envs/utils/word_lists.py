import importlib.resources
import nltk
from nltk.corpus import words


class EnglishDictionary:
    """Dictionary Utils for English words."""

    def __init__(self, keep_proper_nouns=False, include_nltk=False):
        self.include_nltk = include_nltk
        self.uk_words = self._load_dic("en_GB.dic")
        self.us_words = self._load_dic("en_US.dic")
        self.nltk_words = self._load_nltk() if include_nltk else set()
        self.keep_proper_nouns = keep_proper_nouns

    def _load_dic(self, filename: str) -> set[str]:
        """Load words from a .dic file inside the package's data folder."""
        with importlib.resources.open_text("mypackage.data", filename) as f:
            lines = f.readlines()[1:]  # Skip first line (word count)
        words = set(line.split("/")[0].strip() for line in lines)
        return set(
            word
            for word in words
            if word.isalpha() and (word.islower() or self.keep_proper_nouns)
        )  # remove non-alphabetic words and words with capital letters (nouns)

    def _load_nltk(self) -> set[str]:
        # Load NLTK word list
        nltk.download("words")
        return set(words.words("en"))

    def is_english_word(self, word: str) -> bool:
        """Check if a word is in the UK and/or US and/or NLTK English dictionary."""
        word = word.lower()
        return word in self.uk_words or word in self.us_words or word in self.nltk_words

    def get_all_words(self) -> set[str]:
        """Get all words in the dictionary as a set"""
        return self.uk_words.union(self.us_words).union
