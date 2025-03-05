from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage

# 1. Define specialized prompts
INITIAL_PROMPT = """You are a medical diagnosis assistant. Analyze symptoms and list potential diseases from this context:
{context}

Current symptoms: {symptoms}
Return ONLY comma-separated disease names matching ALL symptoms. Say 'None' if no matches."""

REFINEMENT_PROMPT = """Filter these candidate diseases based on NEW symptoms. Keep only diseases that have ALL symptoms (existing + new):
Current symptoms: {all_symptoms}
Candidates: {candidates}

New symptoms: {new_symptoms}
Return ONLY comma-separated surviving disease names. Say 'None' if no matches."""

# 2. Create agent tools
def setup_tools(retriever, llm):
    return [
        Tool(
            name="initial_diagnosis",
            func=lambda symptoms: handle_llm_call(INITIAL_PROMPT, {"symptoms": symptoms}, retriever),
            description="Use for first diagnosis with symptoms. Returns list of possible diseases."
        ),
        Tool(
            name="refine_diagnosis",
            func=lambda inputs: handle_llm_call(REFINEMENT_PROMPT, inputs, None),
            description="Use to refine diagnosis with additional symptoms. Requires: candidates, all_symptoms, new_symptoms."
        )
    ]

def handle_llm_call(prompt_template, inputs, retriever):
    if retriever:
        docs = retriever.get_relevant_documents(inputs["symptoms"])
        inputs["context"] = "\n".join([d.page_content for d in docs])
    
    prompt = prompt_template.format(**inputs)
    return query_llm(prompt)

# 3. Configure the agent
def create_diagnosis_agent(tools, llm):
    system_message = SystemMessage(content="""You are a medical diagnosis assistant. Follow these steps:
1. Use initial_diagnosis for first symptoms
2. If >3 results, ask for more symptoms
3. Use refine_diagnosis with new symptoms
4. Repeat until <=3 results or user stops""")
    
    prompt = ChatPromptTemplate.from_messages([
        system_message,
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    
    return create_react_agent(llm, tools, prompt)

# 4. Execution flow
class DiagnosisAgent:
    def __init__(self, retriever, llm):
        self.tools = setup_tools(retriever, llm)
        self.agent = create_diagnosis_agent(self.tools, llm)
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        
        self.state = {
            "candidates": [],
            "all_symptoms": [],
            "iteration": 0
        }

    def run(self):
        while True:
            if self.state["iteration"] == 0:
                symptom = input("Enter first symptom: ").strip().lower()
                self.state["all_symptoms"].append(symptom)
                result = self.executor.invoke({
                    "input": f"Perform initial diagnosis with: {symptom}"
                })
            else:
                symptom = input("Enter additional symptom (or 'exit'): ").strip().lower()
                if symptom == 'exit':
                    break
                    
                self.state["all_symptoms"].append(symptom)
                result = self.executor.invoke({
                    "input": f"Refine diagnosis with: {symptom}",
                    "candidates": self.state["candidates"],
                    "all_symptoms": self.state["all_symptoms"]
                })

            if "None" in result["output"]:
                print("No matching diseases found.")
                return
            
            self.state["candidates"] = result["output"].split(", ")
            self.state["iteration"] += 1
            
            print(f"\nCurrent candidates ({len(self.state['candidates'])}):")
            print("\n".join(f"- {d}" for d in self.state["candidates"]))
            
            if len(self.state["candidates"]) <= 3:
                break

        return self.state["candidates"]

# 5. Helper functions
def parse_disease_name(entry):
    return entry.split("Disease:", 1)[1].split(";", 1)[0].strip()

def query_llm(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(llm.device)
    outputs = llm.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Usage
if __name__ == "__main__":
    # Initialize components
    disease_docs = load_diseases("diseases.txt")
    vectorstore = Chroma.from_documents(disease_docs, embedding=your_embedding_model)
    retriever = vectorstore.as_retriever()
    
    agent = DiagnosisAgent(retriever, llm)
    results = agent.run()
    
    if results:
        print("\nMost likely diagnoses:")
        for i, disease in enumerate(results, 1):
            print(f"{i}. {disease}")
