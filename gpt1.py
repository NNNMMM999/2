def get_initial_candidates(symptoms):
    # Invoke your LLM chain with initial symptoms to retrieve candidate diseases.
    response = disease_rag_chain.invoke(symptoms)
    return response  # Assume response is a string listing candidate diseases and their symptoms.

def refine_candidates(current_candidates, new_symptoms):
    # Create a prompt that includes the current candidate list and new symptoms.
    # The prompt instructs the LLM to narrow down the list based on the new symptoms.
    prompt = f"The current list of diseases and their symptoms is:\n{current_candidates}\n\n"
    prompt += f"New symptoms reported: {new_symptoms}\n"
    prompt += "Refine the list to only include diseases that exhibit the new symptoms."
    
    refined_response = llm_chain(prompt)
    return refined_response

# Interactive loop
print("Enter your symptoms for a diagnosis (type 'exit' to quit):")
candidates = None
while True:
    user_input = input("Symptoms: ")
    if user_input.strip().lower() == 'exit':
        break

    if candidates is None:
        # First run: get initial candidate list from the database.
        candidates = get_initial_candidates(user_input)
    else:
        # Refinement: update candidate list based on additional symptoms.
        candidates = refine_candidates(candidates, user_input)
    
    print("\nCurrent Candidate List:")
    print(candidates)
    
    # Optionally, ask the user if the list is acceptable or if they want to add more symptoms.
    more = input("Do you want to add more symptoms to refine further? (yes/no): ")
    if more.strip().lower() not in ['yes', 'y']:
        break
