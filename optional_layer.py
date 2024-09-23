# OPTIONAL LAYER
## IT IS NOT USED IN THE CURRENT PROJECT

from googletrans import Translator

translator = Translator()

# Translate review to English before sentiment analysis
def translate_to_english(review):
    try:
        return translator.translate(review, dest='en').text
    except Exception as e:
        print(f"Translation failed: {e}")
        return review  # Return original review if translation fails

data['translated_review'] = data['review'].apply(translate_to_english)
