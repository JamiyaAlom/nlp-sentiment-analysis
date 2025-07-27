import re
import string

class TextPreprocessor:
    @staticmethod
    def clean(text):
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"\b(can|does|do|did|is|are|was|were|would|should|could|have|has|had)\s+not\b", r"\1_not", text)
        text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()