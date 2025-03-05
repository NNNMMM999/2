from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# ------------------------------------------------------------------------------
# Define two prompt templates:
# ------------------------------------------------------------------------------
# Initial prompt template: generate candidate diseases from user symptoms.
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

# Refinement prompt template: refine the candidate list using additional symptoms.
refinement_prompt_template = """
You have the following current list of candidate diseases:
{current_candidates}
The user has now provided additional symptoms: {new_symptoms}
Refine the list to only include diseases that match all the reported symptoms.
If the revised list is too broad, reply with: "Too many diseases found, please provide more specific symptoms."
If no disease matches, reply with: "No matching disease found."
Provide the updated candidate list:
"""

# Create PromptTemplate objects.
initial_prompt = PromptTemplate.from_template(initial_prompt_template)
refinement_prompt = PromptTemplate.from_template(refinement_prompt_template)

# ------------------------------------------------------------------------------
# Helper functions.
# ------------------------------------------------------------------------------

def get_context_from_database(query, retriever):
    """
    Retrieve relevant context from the disease symptom database.
    Assumes your 'retriever' is set up (e.g., via a Chroma vector store) and 
    that format_docs() and truncate_context() are available.
    """
    docs = retriever.get_relevant_documents(query)
    return truncate_context(format_docs(docs), max_tokens=1000, tokenizer=tokenizer)

def call_llm_chain(prompt_text):
    """
    Calls your LLM chain with the provided prompt text.
    This function uses your previously defined `llm_chain` and `extract_answer`.
    """
    llm_output = llm_chain(prompt_text)
    answer = extract_answer(llm_output)
    return answer

# ------------------------------------------------------------------------------
# Define a Diagnostic Agent.
# ------------------------------------------------------------------------------
class DiagnosticAgent:
    def __init__(self, retriever, initial_prompt, refinement_prompt):
        self.retriever = retriever
        self.initial_prompt = initial_prompt
        self.refinement_prompt = refinement_prompt
        self.candidate_list = None  # Stores current candidate diseases.

    def get_initial_candidates(self, symptoms):
        """
        Uses the initial prompt to generate a candidate list based on the user’s symptoms.
        """
        context = get_context_from_database(symptoms, self.retriever)
        prompt_text = self.initial_prompt.format(symptoms=symptoms, context=context)
        self.candidate_list = call_llm_chain(prompt_text)
        return self.candidate_list

    def refine_candidates(self, new_symptoms):
        """
        Uses the refinement prompt to narrow down the candidate list based on additional symptoms.
        """
        if self.candidate_list is None:
            raise ValueError("No candidate list to refine. Please run get_initial_candidates first.")
        prompt_text = self.refinement_prompt.format(
            current_candidates=self.candidate_list, 
            new_symptoms=new_symptoms
        )
        self.candidate_list = call_llm_chain(prompt_text)
        return self.candidate_list

# ------------------------------------------------------------------------------
# Example Usage: Interactive loop for disease diagnosis.
# ------------------------------------------------------------------------------
# Create the agent with your disease symptom retriever.
agent = DiagnosticAgent(disease_retriever, initial_prompt, refinement_prompt)

print("Disease Diagnostic Agent")
print("Enter your initial symptoms (or type 'exit' to quit):")

# Initial candidate generation.
while True:
    initial_symptoms = input("Initial Symptoms: ")
    if initial_symptoms.strip().lower() == 'exit':
        break

    candidates = agent.get_initial_candidates(initial_symptoms)
    print("\nInitial Candidate List:")
    print(candidates)

    # Ask if user wants to refine the list.
    while True:
        more = input("Would you like to add more symptoms to refine the candidate list? (yes/no): ")
        if more.strip().lower() not in ['yes', 'y']:
            break
        new_symptoms = input("Additional Symptoms: ")
        candidates = agent.refine_candidates(new_symptoms)
        print("\nUpdated Candidate List:")
        print(candidates)
    
    print("\n--- New Diagnosis Session ---\n")
