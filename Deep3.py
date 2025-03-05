import torch
import re
import stanza
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Initialize Serbian NLP pipeline
stanza.download('sr')
nlp = stanza.Pipeline('sr')

# Initialize Serbian LLM
tokenizer = AutoTokenizer.from_pretrained(
    "sambanovasystems/SambaLingo-Serbian-Chat",
    use_fast=False
)
llm = AutoModelForCausalLM.from_pretrained(
    "sambanovasystems/SambaLingo-Serbian-Chat",
    device_map="auto",
    torch_dtype=torch.float16
)

# Custom disease data loader
class SerbianDiseaseLoader:
    def __init__(self, file_paths):
        self.file_paths = file_paths
    
    def load_documents(self):
        documents = []
        for path in self.file_paths:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Disease:") and "Symptom:" in line:
                        documents.append(Document(
                            page_content=line,
                            metadata={"source": path}
                        ))
        return documents

# Serbian embeddings wrapper
class SerbianEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer('djovak/embedic-large')
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode(text).tolist()

# Vector store initialization
class SerbianMedicalVectorStore:
    def __init__(self, documents):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.splits = self.text_splitter.split_documents(documents)
        self.embeddings = SerbianEmbeddings()
        self.vectorstore = Chroma.from_documents(
            documents=self.splits,
            embedding=self.embeddings
        )
        self.retriever = self.vectorstore.as_retriever()

# Medical diagnosis agent
class SerbianMedicalAgent:
    def __init__(self, retriever, llm, tokenizer):
        self.retriever = retriever
        self.llm = llm
        self.tokenizer = tokenizer
        self.symptom_cache = {}
        self.current_candidates = []
        self.collected_symptoms = []
        
        self.tools = [
            Tool(
                name="proširi_simptome",
                func=self.proširi_simptome,
                description="Pretvara opis simptoma u medicinske termine"
            ),
            Tool(
                name="pronalaženje_bolesti",
                func=self.pronalaženje_bolesti,
                description="Pronalazi potencijalne bolesti na osnovu simptoma"
            ),
            Tool(
                name="precizna_dijagnoza",
                func=self.precizna_dijagnoza,
                description="Poboljšava dijagnozu sa dodatnim simptomima"
            )
        ]
        
        self.agent = self._napravi_agenta()
        self.executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5
        )
    
    def _napravi_agenta(self):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""Vi ste medicinski asistent za dijagnostiku. Koristite sledeći tok:
1. Proširite opis simptoma u medicinske termine
2. Pronađite potencijalne bolesti koristeći proširene termine
3. Ako ima više od 3 rezultata, zatražite dodatne simptome
4. Ponovite proces sa novim simptomima"""),
            MessagesPlaceholder(variable_name="istorija_razgovora"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_notes")
        ])
        return create_react_agent(self.llm, self.tools, prompt)
    
    def proširi_simptome(self, korisnički_unos):
        """Proširuje simptome koristeći LLM"""
        cache_key = korisnički_unos.lower()
        if cache_key in self.symptom_cache:
            return self.symptom_cache[cache_key]
            
        prompt = f"""Pretvori opis simptoma u standardne medicinske termine:
        Opis: "{korisnički_unos}"
        Vrati listu termina odvojenih zarezima.
        Primer: glavobolja, mučnina, vrtoglavica
        Odgovor: """
        
        prošireno = self._upit_llm(prompt)
        termini = [t.strip().lower() for t in prošireno.split(",")]
        self.symptom_cache[cache_key] = termini
        return termini
    
    def pronalaženje_bolesti(self, simptomi):
        """Inicijalno pronalaženje bolesti"""
        prošireni_termini = []
        for simptom in simptomi:
            prošireni_termini.extend(self.proširi_simptome(simptom))
            
        dokumenti = []
        viđeni = set()
        for termin in prošireni_termini:
            for doc in self.retriever.get_relevant_documents(termin):
                if doc.page_content not in viđeni:
                    viđeni.add(doc.page_content)
                    dokumenti.append(doc)
        
        return self._procesuiraj_rezultate(
            [d.page_content for d in dokumenti],
            simptomi
        )
    
    def precizna_dijagnoza(self, unos):
        """Poboljšanje dijagnoze sa novim simptomima"""
        novi_simptomi = self._izdvoji_simptome(unos["novi_simptomi"])
        self.collected_symptoms.extend(novi_simptomi)
        
        prompt = f"""Filtrirajte bolesti koristeći nove simptome:
        Postojeći simptomi: {', '.join(self.collected_symptoms)}
        Novi simptomi: {', '.join(novi_simptomi)}
        Kandidati: {'\n'.join(self.current_candidates)}

        Vrati SAMO preostale bolesti odvojene zarezima.
        Odgovor: """
        
        odgovor = self._upit_llm(prompt)
        return self._procesuiraj_odgovor(odgovor)
    
    def _procesuiraj_rezultate(self, bolesti, simptomi):
        prompt = f"""Analiziraj bolesti i izlistaj one koje odgovaraju svim simptomima:
        Simptomi: {', '.join(simptomi)}
        Bolesti: {'\n'.join(bolesti)}

        Vrati SAMO nazive bolesti odvojene zarezima.
        Odgovor: """
        
        odgovor = self._upit_llm(prompt)
        return self._procesuiraj_odgovor(odgovor)
    
    def _procesuiraj_odgovor(self, odgovor):
        bolesti = [b.strip() for b in odgovor.split(",") if b.strip()]
        self.current_candidates = bolesti
        return bolesti
    
    def _upit_llm(self, prompt):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        ).to(self.llm.device)
        
        outputs = self.llm.generate(
            **inputs,
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def _izdvoji_simptome(self, tekst):
        prompt = f"""Izvuci sve simptome iz sledećeg opisa:
        Opis: "{tekst}"
        Vrati listu simptoma odvojenih zarezima.
        Simptomi: """
        
        odgovor = self._upit_llm(prompt)
        return [s.strip().lower() for s in odgovor.split(",")]
    
    def _parsiraj_naziv_bolesti(self, unos):
        podudaranje = re.search(r"Disease:\s*(.*?)\s*;", unos)
        return podudaranje.group(1) if podudaranje else unos
    
    def pokreni_dijagnostiku(self):
        print("Dobrodošli u sistem za medicinsku dijagnostiku. Počnite sa opisom simptoma.")
        
        while True:
            if not self.collected_symptoms:
                unos = input("\nOpišite svoje simptome: ").strip()
                simptomi = self._izdvoji_simptome(unos)
                self.collected_symptoms = simptomi
                
                rezultat = self.pronalaženje_bolesti(simptomi)
            else:
                print("\nTrenutne moguće dijagnoze:")
                for i, bolest in enumerate(self.current_candidates, 1):
                    print(f"{i}. {self._parsiraj_naziv_bolesti(bolest)}")
                
                unos = input("\nUnesite dodatne simptome ili 'završi': ").strip()
                if unos.lower() in ['završi', 'exit']:
                    break
                
                rezultat = self.precizna_dijagnoza({"novi_simptomi": unos})
            
            if not rezultat:
                print("Nema pronađenih bolesti koje odgovaraju unetim simptomima.")
                return
            
            if len(rezultat) <= 3:
                print("\nKonačne mogućnosti:")
                for i, bolest in enumerate(rezultat, 1):
                    print(f"{i}. {self._parsiraj_naziv_bolesti(bolest)}")
                return
            
            print(f"Pronađeno {len(rezultat)} potencijalnih dijagnoza. Molimo unesite dodatne simptome.")

# Kompletna izvršna petlja
if __name__ == "__main__":
    # Konfiguracija
    FAJLOVI_BOLESTI = ['bolesti.txt']  # Postavite putanje do vaših datoteka
    
    # Učitavanje podataka
    loader = SerbianDiseaseLoader(FAJLOVI_BOLESTI)
    dokumenti = loader.load_documents()
    
    # Inicijalizacija vektorskog skladišta
    vektorsko_skladište = SerbianMedicalVectorStore(dokumenti)
    
    # Kreiranje i pokretanje agenta
    agent = SerbianMedicalAgent(
        retriever=vektorsko_skladište.retriever,
        llm=llm,
        tokenizer=tokenizer
    )
    
    agent.pokreni_dijagnostiku()
