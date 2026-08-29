from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
 
DetectorFactory.seed = 0
 
SUPPORTED_LANGUAGES = {
    "ta": "Tamil",
    "hi": "Hindi",
    "en": "English"
}
 
def detect_language(text):
    if not any(
        '\u0B80' <= ch <= '\u0BFF' or '\u0900' <= ch <= '\u097F'
        for ch in text
    ):
        return "en"
 
    try:
        detected = detect(text)
    except Exception:
        return "en"
 
    if detected not in SUPPORTED_LANGUAGES:
        return "en"
 
    return detected
 
def translate_to_english(text, source_language):
    if source_language == "en":
        return text
 
    translated = GoogleTranslator(
        source=source_language,
        target="en"
    ).translate(text)
 
    return translated
 
def translate_from_english(text, target_language):
    if target_language == "en":
        return text
 
    translated = GoogleTranslator(
        source="en",
        target=target_language
    ).translate(text)
 
    return translated
 
def get_language_name(code):
    return SUPPORTED_LANGUAGES.get(code, "English")
