import nltk
import emoji
from typing import List


class TextNormalizer:
    def __init__(self):
        nltk.download("punkt_tab")

    def _normalize_text(self, text: str) -> str:
        text = text.replace(":", "–").replace(";", "–")
        text = text.replace("“", "'").replace("”", "'")
        text = text.replace("‘", "'").replace("’", "'")
        # convert double quotes to single quotes
        text = text.replace('"', "'")
        # de-markdownify
        text = text.replace("_", "").replace("*", "")
        # remove emojis
        text = emoji.replace_emoji(text, "")
        # remove extra whitespace
        text = text.strip()

        return text

    def sentenceize(self, text: str) -> List[str]:
        sentences = nltk.sent_tokenize(text)
        sentences = [sentence for sentence in sentences if sentence.strip()]
        return [self._normalize_text(sentence) for sentence in sentences]
