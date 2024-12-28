import json
from dataclasses import dataclass
import os
import anthropic
import time


def load_and_process_sentences(file_path):
    # Load grouped sentences from the JSON file
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            grouped_sentences = json.load(file)
    except IOError as e:
        print(f"An error occurred while reading the file: {e}")
        return []

    return grouped_sentences


def run_aistudio(sentence_group):
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
    )

    # Replace placeholders like {{hi}} with real values,
    # because the SDK does not support variables.
    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=8192,
        temperature=0,
        system="Du bist ein author und kinderarzt der ein Kinderheilkunde-Buch für orthonormalverbraucher schreibt und mit einer KI kommuniziert um hilfreiche aussagen zu erhalten. Du hast nach einem harten arbeits sprint vergessen was du die KI gefragt hast und erhälst nur die ausgaben der KI. Gebe nun zurück was du am wahrscheinlichsten gefragt hast und nichts anderes. Schreibe nur das prompt welches der KI gestellt wurde auf mit keiner sonstigen erklärung. Z.B.:\\n\\n\\n <examples>\n<example>\n<hi>\n'»Meine Tochter isst viel zu wenig!« »Mein Sohn schreit die ganze Nacht!« »Schon wieder eine Ohrenentzündung!« Wenn Eltern in die Praxis der Kinderärztin Dr' \n</hi>\n<ideal_output>\nFasse die häufigsten Probleme zusammen, die Eltern in der Kinderarztpraxis ansprechen\\n\n</ideal_output>\n</example>\n</examples>\n\n",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": sentence_group,
                    }
                ],
            }
        ],
    )
    print(message.content[0].text)
    return message.content[0].text


@dataclass
class Result:
    info: list[dict]

    def append(self, prompt, response):
        self.info.append({"prompt": prompt, "response": response})


if __name__ == "__main__":
    # Example usage
    file_path = "/Users/sanderhergarten/datasources/bookhelper/grouped_sentences.json"
    sentences = load_and_process_sentences(file_path)

    results = Result([])

    for sentence in sentences:
        prompt = run_aistudio(sentence)
        results.append(prompt, sentence)
        time.sleep(0.1)

    with open("results.json", "w") as f:
        json.dump(results.info, f)
