import requests

class OllamaClient:
    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model = model
        self.base_url = host

    def chat(self, prompt: str, stream: bool = False) -> str:
        """
        Sendet eine Chat-Anfrage an das lokale Ollama-Modell.

        Args:
            prompt (str): Die Eingabeaufforderung an das Modell.
            stream (bool): Optional: Streaming aktivieren (nicht implementiert).

        Returns:
            str: Die Antwort des Modells.
        """
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream
        }

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()
            return data.get("response", "")
        else:
            print(f"Fehler beim Chat: {response.status_code} - {response.text}")
            return None

    def get_embedding(self, text: str) -> list:
        """
        Generiert ein Embedding für einen gegebenen Text mit dem aktuellen Modell.

        Args:
            text (str): Der Eingabetext.

        Returns:
            list[float]: Das erzeugte Vektor-Embedding.
        """
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": self.model,
            "prompt": text
        }

        response = requests.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()
            if "embedding" in data:
                return data["embedding"]
            else:
                print("⚠️ Dieses Modell unterstützt keine Embeddings.")
                return None
        else:
            print(f"Fehler beim Abrufen des Embeddings: {response.status_code} - {response.text}")
            return None


def start_chat(client: OllamaClient):

    print("🧠 Willkommen beim lokalen LLM-Chat mit Embedding-Funktion!")
    
    while True:
        user_input = input("Du: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Tschüss!")
            break
        elif user_input.lower().startswith("embed:"):
            text_to_embed = user_input[7:].strip()
            embedding = client.get_embedding(text_to_embed)
            if embedding:
                print(f"📈 Embedding (erste 5 Werte): {embedding[:5]} ... [Länge: {len(embedding)}]")
        else:
            reply = client.chat(user_input)
            if reply:
                print(f"🤖 Ollama: {reply}")


"""
if __name__ == "__main__":

    client = OllamaClient(model="llama3.2")

    #start_chat(client)


    output = client.get_embedding("Hello, how are you?")

    print(output)
    
"""


    
    
