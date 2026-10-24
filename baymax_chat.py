"""
baymax_chat.py - try the whole safety stack in your terminal.

    your message -> safety_gate -> (if not triggered) baymax model -> output_guard -> reply

Run:   python baymax_chat.py
Quit:  Ctrl+C or type "quit"
Needs: the Ollama model named "baymax" (ollama create baymax -f Modelfile)
"""

import ollama

from output_guard import clean_reply
from safety_gate import check_message

MODEL = "baymax"
SHOW_DEBUG = True    # print what the gate and guard did, so you can learn from it


def answer(user_text: str) -> str:
    gate = check_message(user_text)
    if gate.triggered:
        if SHOW_DEBUG:
            print(f"   [gate: {gate.category}, matched '{gate.matched}'. The model was NOT used.]")
        return gate.message

    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": user_text}],   # the Modelfile adds the system prompt
        options={"temperature": 0.4, "num_predict": 150},
    )
    raw = resp["message"]["content"].strip()
    cleaned, report = clean_reply(raw)
    if SHOW_DEBUG:
        for reason, s in report["removed"]:
            print(f"   [guard removed ({reason}): {s}]")
        if report["used_fallback"]:
            print("   [guard: nothing usable left, used the fallback reply]")
    return cleaned


def main():
    print("Baymax is ready. Type a message (or 'quit').\n")
    while True:
        try:
            text = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
        print(f"Baymax: {answer(text)}\n")


if __name__ == "__main__":
    main()
