"""
TransCorpus Languages Module.

This module defines the `M2M100_Languages` type, which is a Literal type representing
the supported languages for the M2M100 multilingual translation model. The list of
languages includes ISO 639-1 codes for various languages.

Type Aliases:
    - M2M100_Languages: A Literal type containing all supported language codes.

Typical usage example:
    from transcorpus.languages import M2M100_Languages

    def validate_language(language: M2M100_Languages) -> bool:
        if language in ["en", "fr", "es"]:
            return True
        return False

    print(validate_language("en"))  # True
    print(validate_language("xx"))  # Error: Argument "xx" is not valid for type "M2M100_Languages"
"""

from typing import Literal

M2M100_Languages = Literal[
    "af",
    "am",
    "ar",
    "ast",
    "az",
    "ba",
    "be",
    "bg",
    "bn",
    "br",
    "bs",
    "ca",
    "ceb",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fa",
    "ff",
    "fi",
    "fr",
    "fy",
    "ga",
    "gd",
    "gl",
    "gu",
    "ha",
    "he",
    "hi",
    "hr",
    "ht",
    "hu",
    "hy",
    "id",
    "ig",
    "ilo",
    "is",
    "it",
    "ja",
    "jv",
    "ka",
    "kk",
    "km",
    "kn",
    "ko",
    "lb",
    "lg",
    "ln",
    "lo",
    "lt",
    "lv",
    "mg",
    "mk",
    "ml",
    "mn",
    "mr",
    "ms",
    "my",
    "ne",
    "nl",
    "no",
    "ns",
    "oc",
    "or",
    "pa",
    "pl",
    "ps",
    "pt",
    "ro",
    "ru",
    "sd",
    "si",
    "sk",
    "sl",
    "so",
    "sq",
    "sr",
    "ss",
    "su",
    "sv",
    "sw",
    "ta",
    "th",
    "tl",
    "tn",
    "tr",
    "uk",
    "ur",
    "uz",
    "vi",
    "wo",
    "xh",
    "yi",
    "yo",
    "zh",
    "zu",
]
