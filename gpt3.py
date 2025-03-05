import re
import stanza
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# -------------------------------
# 1. Set Up Stanza for Serbian
# -------------------------------
stanza.download('sr')  # Download the Serbian model (run this once)
nlp = stanza.Pipeline('sr', processors='tokenize,lemma')

# -------------------------------
# 2. Load the Synonyms Dictionary from a File
# -------------------------------
def load_synonyms(file_path):
    """
    Loads a synonyms dictionary from a text file.
    Each line should be in the format: 
       canonical_term: synonym1, synonym2, synonym3
    Returns a dictionary mapping canonical terms to a list of synonyms.
    """
    synonyms = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            canonical, syn_str = line.split(":", 1)
            canonical = canonical.strip()
            synonyms_list = [s.strip() for s in syn_str.split(",") if s.strip()]
            synonyms[canonical] = synonyms_list
    return synonyms

# Update the file path to point to your synonyms text file.
synonym_file_path = "synonyms.txt"
symptom_synonyms = load_synonyms(synonym_file_path)

# -------------------------------
# 3. Normalize Symptoms Using Stanza and Synonym Mapping
# -------------------------------
def normalize_symptoms(user_input):
    """
    Normalize user input by lowercasing, lemmatizing using Stanza,
    and mapping tokens to canonical symptom terms via the loaded dictionary.
    """
    # Lowercase and remove punctuation
    processed_input = re.sub(r'[^\w\s]', '', user_input.lower())
    doc = nlp(processed_input)
    tokens = []
    for sentence in doc.sentences:
        for word in sentence.words:
            tokens.append(word.lemma)
    
    canonical_tokens = []
    for token in tokens:
        replaced = False
        # Check each canonical term and its synonyms for a match
        for canonical, synonyms in symptom_synonyms.items():
            if token in synonyms:
                canonical_tokens.append(canonical)
                replaced = True
                break
        if not replaced:
            canonical_tokens.append(token)
    normalized = " ".join(canonical_tokens)
    return normalized

# -------------------------------
# 4. Helper Functions (Assumed from Your Previous Code)
# -------------------------------
def get_context_from_database(query, retriever):
    """
    Retrieve relevant context from the disease symptom database.
    Assumes that 'retriever' is set up (e.g., via a Chroma vector store)
    and that format_docs() and truncate_context() are available.
    """
    docs = retriever.get_relevant_documents(query)
    return truncate_context(format_docs(docs), max_tokens=1000, tokenizer=tokenizer)

def call_llm_chain(prompt_text):
    """
    Call your LLM chain with the provided prompt text.
    This function uses your previously defined llm_chain and extract_answer functions.
    """
    llm_output = llm_chain(prompt_text)
    answer = extract_answer(llm_output)
    return answer

# -------------------------------
# 5. Define Prompt Templates for Diagnosis
# -------------------------------
# Initial prompt: generate candidate diseases based on initial user symptoms.
initial_prompt_template = """
You are a medical diagnostic assistant with access to a disease symptom database.
Each entry in the database is formatted as "Disease: Symptoms:".
Based on the user's reported symptoms, list the candidate diseases along with their symptoms.
If too many diseases match, reply with: "Too many diseases found, please provide more specific symptoms."
User Symptoms: {symptoms}
Database:
{context}
Your Answer (list diseases and their symptoms):
"""

# Refinement prompt: narrow the candidate list using additional symptoms.
refinement_prompt_template = """
You have the following current list of candidate diseases:
{current_candidates}
The user has now provided additional symptoms: {new_symptoms}
Refine the list to only include diseases that match all the reported symptoms.
If the revised list is too broad, reply with: "Too many diseases found, please provide more specific symptoms."
If no disease matches, reply with: "No matching disease found."
Provide the updated candidate list:
"""

initial_prompt = PromptTemplate.from_template(initial_prompt_template)
refinement_prompt = PromptTemplate.from_template(refinement_prompt_template)

# -------------------------------
# 6. Define the Diagnostic Agent Class
# -------------------------------
class DiagnosticAgent:
    def __init__(self, retriever, initial_prompt, refinement_prompt):
        self.retriever = retriever
        self.initial_prompt = initial_prompt
        self.refinement_prompt = refinement_prompt
        self.candidate_list = None  # Stores the current candidate disease list

    def get_initial_candidates(self, symptoms):
        # Normalize the user input using our synonym mapping
        normalized_symptoms = normalize_symptoms(symptoms)
        context = get_context_from_database(normalized_symptoms, self.retriever)
        prompt_text = self.initial_prompt.format(symptoms=normalized_symptoms, context=context)
        self.candidate_list = call_llm_chain(prompt_text)
        return self.candidate_list

    def refine_candidates(self, new_symptoms):
        normalized_new = normalize_symptoms(new_symptoms)
        prompt_text = self.refinement_prompt.format(
            current_candidates=self.candidate_list,
            new_symptoms=normalized_new
        )
        self.candidate_list = call_llm_chain(prompt_text)
        return self.candidate_list

# -------------------------------
# 7. Interactive Loop with the Diagnostic Agent
# -------------------------------
# Assume that 'disease_retriever', 'tokenizer', 'llm_chain', 'extract_answer',
# 'format_docs', and 'truncate_context' are defined from your previous project.
agent = DiagnosticAgent(disease_retriever, initial_prompt, refinement_prompt)

print("Disease Diagnostic Agent (Serbian)")
print("Enter your description of symptoms (or type 'exit' to quit):")

while True:
    user_input = input("Symptoms: ")
    if user_input.strip().lower() == 'exit':
        break

    # Get the initial candidate list
    candidates = agent.get_initial_candidates(user_input)
    print("\nInitial Candidate List:")
    print(candidates)
    
    # Ask if the user wishes to refine further
    while True:
        more = input("Would you like to add more symptoms to refine the candidate list? (yes/no): ")
        if more.strip().lower() not in ['yes', 'y']:
            break
        additional = input("Additional Symptoms: ")
        candidates = agent.refine_candidates(additional)
        print("\nUpdated Candidate List:")
        print(candidates)
    
    print("\n--- New Diagnosis Session ---\n")
