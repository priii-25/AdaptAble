import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import spacy
from transformers import pipeline
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
nltk.download('wordnet')

nlp = spacy.load("en_core_web_sm")

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0)

def simplify_sentence(sentence):
    """
    Simplifies a sentence by breaking down long sentences,
    replacing complex words with simpler synonyms, and
    rewriting in active voice where possible.
    """
    if len(sentence.split()) > 20:
        sentences = sent_tokenize(sentence)
    else:
        sentences = [sentence]

    simplified_sentences = []

    for sent in sentences:
        doc = nlp(sent)
        simplified_sentence = []

        for token in doc:
            if len(token.text) > 6:
                synonyms = wordnet.synsets(token.text)
                if synonyms:
                    simple_word = synonyms[0].lemmas()[0].name()
                    simplified_sentence.append(simple_word)
                else:
                    simplified_sentence.append(token.text)
            else:
                simplified_sentence.append(token.text)

        simplified_sentences.append(" ".join(simplified_sentence))

    return " ".join(simplified_sentences)

def summarize_text(text, max_length=50):
    """
    Summarizes the input text using a pre-trained model.
    """
    return summarizer(text, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']

def improve_text_accessibility(input_text):
    """
    Main function to process the input text for accessibility improvement.
    """
    summarized_text = summarize_text(input_text)
    print("Summarized Text:\n", summarized_text)

    simplified_text = ""
    for sentence in sent_tokenize(summarized_text):
        simplified_text += simplify_sentence(sentence) + " "

    return simplified_text.strip()

input_text = """
In the realm of contemporary educational paradigms, the integration of technology into pedagogical methodologies has emerged as a pivotal factor in enhancing the learning experience. This phenomenon is characterized by the utilization of digital tools and resources, which facilitate a more interactive and engaging environment for students. The incorporation of multimedia elements, such as videos, simulations, and interactive software, not only caters to diverse learning styles but also fosters critical thinking and problem-solving skills among learners.
"""
output_text = improve_text_accessibility(input_text)
print("Simplified and Summarized Text:\n", output_text)