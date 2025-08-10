import re
import spacy
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('FAQ-Chatbot')

class EnhancedFAQChatbot:
    def __init__(self, faqs: List[Dict[str, str]], similarity_threshold: float = 0.25):
        """
        Initialize FAQ chatbot
        
        Args:
            faqs: List of dictionaries with 'question' and 'answer' keys
            similarity_threshold: Minimum cosine similarity to consider a match
        """
        try:
            # Validate input data
            if not faqs:
                raise ValueError("FAQ list cannot be empty")
            if any('question' not in q or 'answer' not in q for q in faqs):
                raise ValueError("Each FAQ must contain 'question' and 'answer' keys")
                
            self.faqs = faqs
            self.threshold = similarity_threshold
            self.nlp = self._load_spacy_model()
            self.vectorizer = TfidfVectorizer()
            
            # Preprocess and vectorize FAQs
            self.processed_faqs = [self._preprocess_text(q["question"]) for q in faqs]
            self.faq_vectors = self.vectorizer.fit_transform(self.processed_faqs)
            
            logger.info(f"Chatbot initialized with {len(faqs)} FAQs")
            
        except Exception as e:
            logger.exception(f"Initialization failed: {str(e)}")
            raise

    def _load_spacy_model(self) -> spacy.language.Language:
        """Load spaCy model with error handling"""
        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            logger.info("spaCy model loaded successfully")
            return nlp
        except OSError:
            logger.warning("spaCy model not found. Downloading...")
            try:
                import subprocess
                subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                return spacy.load("en_core_web_sm", disable=["parser", "ner"])
            except Exception as e:
                logger.critical(f"Model download failed: {str(e)}")
                raise RuntimeError("spaCy model unavailable") from e

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text using NLP techniques"""
        if not text or not isinstance(text, str):
            logger.warning(f"Invalid input for preprocessing: {text}")
            return ""
            
        try:
            # Basic cleaning
            text = text.lower().strip()
            text = re.sub(r'[^a-z0-9\s]', '', text)
            
            # Advanced NLP processing
            doc = self.nlp(text)
            tokens = [
                token.lemma_.strip() 
                for token in doc 
                if not token.is_stop and 
                   not token.is_punct and 
                   token.lemma_.strip() and 
                   len(token.lemma_) > 1
            ]
            
            return " ".join(tokens)
            
        except Exception as e:
            logger.error(f"Preprocessing failed for '{text}': {str(e)}")
            return ""

    def _validate_query(self, query: str) -> bool:
        """Check if query is valid"""
        if not query or not isinstance(query, str) or len(query.strip()) < 2:
            logger.warning(f"Invalid query: '{query}'")
            return False
        return True

    def get_response(self, user_query: str) -> str:
        """Get best matching FAQ response"""
        try:
            # Input validation
            if not self._validate_query(user_query):
                return "Please ask a clear question with at least 2 characters"
                
            # Preprocess query
            processed_query = self._preprocess_text(user_query)
            if not processed_query:
                return "I couldn't process your question. Please try rephrasing."
                
            # Vectorize query
            query_vector = self.vectorizer.transform([processed_query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.faq_vectors)
            best_match_idx = np.argmax(similarities)
            highest_similarity = similarities[0, best_match_idx]
            
            logger.debug(f"Query: '{user_query}' | Processed: '{processed_query}'")
            logger.debug(f"Best match: '{self.faqs[best_match_idx]['question']}' | Similarity: {highest_similarity:.2f}")
            
            # Return response based on threshold
            if highest_similarity > self.threshold:
                return self.faqs[best_match_idx]["answer"]
            else:
                return self._get_fallback_response(user_query)
                
        except ValueError as ve:
            logger.error(f"Vectorization error: {str(ve)}")
            return "I'm having trouble understanding that. Could you try different words?"
        except Exception as e:
            logger.exception(f"Unexpected error: {str(e)}")
            return "Something went wrong on my end. Please try again later."

    def _get_fallback_response(self, query: str) -> str:
        """Handle low-confidence matches"""
        # Add custom logic for specific query patterns
        if "contact" in query.lower():
            return "For direct assistance, please email support@company.com"
            
        # Generic fallback
        suggestions = "\n".join([f"- {q['question']}" for q in self.faqs[:3]])
        return (
            "I'm not sure I understand. Here are some topics I can help with:\n" 
            f"{suggestions}\n"
            "Try asking about one of these or rephrase your question."
        )

# Sample FAQ dataset
faqs = [
    {"question": "What is your return policy?", "answer": "Items can be returned within 30 days with original receipt."},
    {"question": "How do I reset my password?", "answer": "Visit our login page and click 'Forgot Password'."},
    {"question": "Do you ship internationally?", "answer": "Yes, we ship to over 50 countries worldwide."},
    {"question": "What payment methods do you accept?", "answer": "We accept Visa, Mastercard, PayPal, and Apple Pay."},
    {"question": "How can I track my order?", "answer": "Use the tracking number in your confirmation email on our Orders page."}
]

if __name__ == "__main__":
    # Initialize chatbot with enhanced error handling
    try:
        chatbot = EnhancedFAQChatbot(faqs, similarity_threshold=0.3)
    except Exception as e:
        logger.critical(f"Chatbot initialization failed: {str(e)}")
        exit(1)
    
    # Test cases
    test_queries = [
        "how to return something I bought?",
        "password reset help",
        "do you deliver to France?",
        "payment options",
        "where's my package?",
        "",  # Empty query
        "   ",  # Whitespace
        123,  # Invalid type
        "this is a completely unrelated question"
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        print(f"Bot: {chatbot.get_response(str(query))}")