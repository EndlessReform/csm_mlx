import nltk
import emoji
from typing import List
import re


def nuke_parens(text):
    # Replace opening or closing parens with a comma
    text = re.sub(r"[()]", ",", text)
    # Normalize: remove duplicate commas and excess space
    text = re.sub(r",\s*,+", ", ", text)  # multiple commas to one
    text = re.sub(r"\s+,", ", ", text)  # space before comma
    text = re.sub(r",\s+", ", ", text)  # extra spaces after comma
    text = re.sub(r"\s{2,}", " ", text)  # collapse extra spaces
    return text.strip()


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
        # Convert ` (`, `) ` to `, `
        text = nuke_parens(text)

        return text

    def sentenceize(self, text: str) -> List[str]:
        sentences = nltk.sent_tokenize(text)
        sentences = [sentence for sentence in sentences if sentence.strip()]
        return [self._normalize_text(sentence) for sentence in sentences]
