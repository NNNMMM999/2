def get_matching_diseases(symptoms, candidate_entries=None):
    # Get initial candidates if first run
    if not candidate_entries:
        # Retrieve full disease entries with symptoms
        docs = retriever.get_relevant_documents(" ".join(symptoms))
        disease_entries = [d.page_content for d in docs]
    else:
        # Use existing full entries for refinement
        disease_entries = candidate_entries
    
    # Create LLM prompt with FULL ENTRIES
    prompt_text = diagnosis_prompt.format(
        symptoms=", ".join(symptoms),
        disease_list="\n".join(disease_entries)
    
    # Get and parse response (returns disease NAMES)
    response = query_llm(prompt_text)
    matched_names = [d.strip() for d in response.split(",") if d.strip() != "None"]
    
    # Keep full entries for matched diseases
    return [entry for entry in disease_entries 
           if any(name in entry for name in matched_names)]

# Updated diagnosis flow
def diagnose():
    symptoms = []
    current_entries = None  # Now stores FULL disease entries
    
    while True:
        # Get initial symptoms
        if not symptoms:
            print("Describe your symptoms (one at a time):")
            while len(symptoms) < 1:
                symptom = input("First symptom: ").strip().lower()
                if symptom: symptoms.append(symptom)
        
        # Initial matching (returns full entries)
        current_entries = get_matching_diseases(symptoms, current_entries)
        current_names = [parse_disease_name(entry) for entry in current_entries]
        
        print(f"\nPossible matches ({len(current_names)}): {', '.join(current_names)}")
        
        # Handle results
        if len(current_names) == 0:
            print("No matching diseases found.")
            return []
        elif len(current_names) <= 3:
            print("\nFinal diagnosis candidates:")
            return current_names
        
        # Refinement phase
        print("\nAdd more symptoms to narrow down:")
        while True:
            symptom = input("Next symptom (or 'exit'): ").strip().lower()
            if symptom == 'exit':
                return current_names
            if symptom:
                symptoms.append(symptom)
                break
        
        # Refined matching with previous FULL ENTRIES
        current_entries = get_matching_diseases(symptoms, current_entries)

# Helper function
def parse_disease_name(entry):
    """Extract disease name from full entry"""
    return entry.split("Disease:", 1)[1].split(";", 1)[0].strip()
