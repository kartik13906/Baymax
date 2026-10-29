"""
remedies.py - a small, reviewable list of home remedies and first-aid steps. NO MEDICINES.

Why a list in code instead of teaching the model the facts:
  A 1.5B model trained on a few hundred rows will happily INVENT a remedy for something it never saw.
  Here the facts live in this file. When the person mentions a pain or injury, baymax_live.py looks the
  entry up and hands it to the model inside the message ("Remedies you may share (use only these): ...").
  The model only has to phrase it warmly, which is easy to learn and works for every entry, including
  ones that were never in the training data. To change what Baymax recommends, edit THIS file.

Each entry:
  key      short id
  q        how a person names it in a question ("a headache")
  causes   example things a person might say after "ouch" (used to generate training rows and tests)
  aliases  regex (on lowercase text without punctuation) that finds the entry
  remedy   one or two sentences: comfort and first-aid steps only
  see      one sentence: when to see a doctor (red flags)
  urgent   True when even a mild case deserves a medical check (head bump, eye injury)

Rules for every sentence (python remedies.py checks them with the output guard):
  no digits or number words, no medicine names, no exact times or schedules, no promises.
  Say "for a short while" or "until it feels comfortable", never "fifteen minutes".

THESE ARE GENERAL, WIDELY-TAUGHT FIRST-AID AND SELF-CARE STEPS, WRITTEN FROM GENERAL KNOWLEDGE, NOT BY A
CLINICIAN. Have a doctor or nurse read this file, or compare each entry with a trusted source (for example
your national health service or the Red Cross first-aid guides), before anyone else relies on it.

Order matters: the first entry whose alias matches wins, so specific entries come before general ones.
"""

import re
import sys

from safety_gate import normalize

ENTRIES = [
    dict(key="head_bump", q="a bump on the head", causes=["i bumped my head on a cupboard", "i hit my head on a shelf"],
         aliases=r"\b(bump(ed)?|hit|banged|knocked)\s+(my|the)\s+head\b",
         remedy="Please sit down, rest, and hold a cold cloth on the bump. Stay with someone and watch for vomiting, confusion or growing sleepiness.",
         see="If any of that happens, or you pass out even briefly, please get medical help right away.", urgent=True),
    dict(key="finger_door", q="a finger caught in a door", causes=["i slammed my finger in the car door", "i crushed my finger in a drawer"],
         aliases=r"\b(slammed|crushed|jammed|trapped|caught)\b.*\b(finger|thumb|hand|toe)\b|\b(finger|thumb)\b.*\b(door|drawer)\b",
         remedy="Please hold a cold pack wrapped in a cloth against it and keep it raised. Take off any rings before it swells.",
         see="If it is very swollen, looks bent, or the nail is bleeding or loose, please get it checked by a doctor soon.", urgent=False),
    dict(key="stubbed_toe", q="a stubbed toe", causes=["i stubbed my toe on the door", "i stubbed my toe on the bed"],
         aliases=r"\bstubbed\b|\b(toe|toes)\b",
         remedy="Please sit and rest it, then raise it and hold a cold pack wrapped in a cloth against it for a short while.",
         see="If it looks crooked, or is very swollen or bruised, please see a doctor.", urgent=False),
    dict(key="splinter", q="a splinter", causes=["i have a splinter in my finger"],
         aliases=r"\bsplinter\w*\b",
         remedy="Please wash your hands and the spot, then pull the splinter out gently in the direction it went in, using clean tweezers. Wash the spot again and cover it if needed.",
         see="If it is deep, will not come out, or the skin looks infected, please see a doctor.", urgent=False),
    dict(key="sting", q="an insect sting", causes=["a bee stung me on my arm", "a wasp stung my leg"],
         aliases=r"\b(sting|stings|stung|bee|wasp|insect\s+bite|mosquito\s+bite)\b",
         remedy="Please scrape out the stinger if you can see it, wash the spot, and cool it with a damp cloth. Raising the area can ease the swelling.",
         see="If your face or lips swell, or you have trouble breathing or feel faint, please get emergency help right away.", urgent=False),
    dict(key="nosebleed", q="a nosebleed", causes=["i have a nosebleed", "my nose is bleeding"],
         aliases=r"\bnose\s?bleed\w*\b|\bnose\s+is\s+bleeding\b|\bbleeding\s+nose\b",
         remedy="Please sit up, lean forward, and pinch the soft part of your nose firmly while you breathe through your mouth. A cool cloth on the bridge of your nose can help.",
         see="If it will not stop after a good while of pinching, or it followed a hard knock, please get medical help.", urgent=False),
    dict(key="sunburn", q="sunburn", causes=["i got sunburned at the beach", "my shoulders are sunburned"],
         aliases=r"\bsun\s?burn\w*\b",
         remedy="Please cool your skin with a damp cloth or a cool shower, drink plenty of water, and stay out of the sun. Loose clothes and a gentle moisturiser or aloe gel can soothe it.",
         see="If you have large blisters or a fever, or you feel dizzy, please see a doctor.", urgent=False),
    dict(key="burn", q="a minor burn", causes=["i burned my hand on the stove", "i scalded my arm with hot water"],
         aliases=r"\b(burn|burned|burnt|burns|burning|scald|scalded)\b",
         remedy="Please hold the burn under cool running water for several minutes, and take off rings or tight items nearby before it swells. Cover it loosely with a clean cloth, and do not use ice, butter or toothpaste.",
         see="If the burn is large, blistered widely, on your face, hands or private parts, or looks white or charred, please get medical help right away.", urgent=False),
    dict(key="blister", q="a blister", causes=["i have a blister on my foot", "i got a blister from my new shoes"],
         aliases=r"\bblister\w*\b",
         remedy="Please keep it clean and covered, and try not to pop it. If it bursts, wash it gently and cover it with a clean bandage.",
         see="If it looks red, swollen or leaks pus, please see a doctor.", urgent=False),
    dict(key="scrape", q="a scrape", causes=["i fell off my bike and scraped my knee", "i scraped my elbow on the road"],
         aliases=r"\b(scrape|scraped|scrapes|graze|grazed)\b",
         remedy="Please rinse the scrape gently with clean running water, pat it dry, and cover it with a clean bandage. Change the covering when it gets wet or dirty.",
         see="If it becomes red, swollen, warm or leaks pus, please see a doctor.", urgent=False),
    dict(key="cut", q="a cut", causes=["i cut my finger while cooking", "i got a paper cut on my thumb"],
         aliases=r"\b(cut|cuts|paper\s+cut)\b",
         remedy="Please wash your hands, then press on the cut with a clean cloth to slow the bleeding. Rinse it gently with clean running water and cover it with a clean bandage.",
         see="If it is deep, will not stop bleeding, or has dirt that will not wash out, please get medical help.", urgent=False),
    dict(key="sprain", q="a twisted ankle", causes=["i twisted my ankle while running", "i rolled my ankle on the stairs"],
         aliases=r"\b(twisted|sprain|sprained|rolled)\b",
         remedy="Please rest it, raise it up, and avoid putting weight on it. A cold pack wrapped in a cloth for a short while and a snug, not tight, bandage can help.",
         see="If you cannot put weight on it, it looks out of shape, or the swelling is severe, please get it checked by a doctor.", urgent=False),
    dict(key="bruise", q="a bruise", causes=["i have a big bruise on my leg", "i bruised my arm on a table"],
         aliases=r"\bbruise\w*\b",
         remedy="Please rest the area and raise it if you can. A cold pack wrapped in a cloth helps early on, and a warm cloth later can help it fade.",
         see="If it is very swollen, very painful, or appeared without any knock, please see a doctor.", urgent=False),
    dict(key="pulled_muscle", q="a pulled muscle", causes=["i pulled a muscle in my calf", "i strained my hamstring"],
         aliases=r"\b(pulled|strain|strained)\b|\b(calf|hamstring|groin)\b",
         remedy="Please rest the muscle and do not stretch it hard. A cold pack wrapped in a cloth helps early on, then a warm cloth and gentle movement as it eases.",
         see="If you heard a pop, cannot use the muscle, or the swelling is severe, please see a doctor.", urgent=False),
    dict(key="knee", q="knee pain", causes=["my knee hurts after football", "my knee is sore from running"],
         aliases=r"\bknees?\b",
         remedy="Please rest the knee, raise it, and avoid deep bends and stairs for now. A cold pack wrapped in a cloth can ease swelling after activity.",
         see="If it gives way, locks, is very swollen, or you cannot put weight on it, please see a doctor.", urgent=False),
    dict(key="wrist", q="wrist pain", causes=["my wrist hurts from typing", "my wrist is sore after the game"],
         aliases=r"\bwrists?\b",
         remedy="Please rest your wrist, take breaks from typing or phone use, and do gentle wrist stretches. Keeping it in a natural, supported position helps.",
         see="If you have numbness, tingling, or swelling that stays, please see a doctor.", urgent=False),
    dict(key="shoulder", q="shoulder pain", causes=["my shoulder hurts from sleeping in a strange position", "my shoulder is sore from lifting"],
         aliases=r"\bshoulders?\b",
         remedy="Please move it gently and rest it, and avoid heavy lifting for now. A warm cloth can ease the stiffness.",
         see="If the pain is sudden, spreads to your arm or chest, or you cannot lift your arm, please get medical help.", urgent=False),
    dict(key="neck", q="a stiff neck", causes=["my neck is stiff after sleeping badly", "my neck hurts from looking at my phone"],
         aliases=r"\bneck\b",
         remedy="Please move your neck slowly and gently, and put a warm cloth on it. Check that your pillow and screen height are not straining it.",
         see="If it comes with a fever, a bad headache, or numbness in your arms, please get medical help.", urgent=False),
    dict(key="jaw", q="jaw pain", causes=["my jaw hurts when i chew", "my jaw is sore after the dentist"],
         aliases=r"\bjaw\b",
         remedy="Please eat soft food, avoid wide yawning and chewing gum, and place a warm cloth on your jaw. Gentle jaw stretches can help.",
         see="If you cannot open your mouth, or you have a fever or swelling, please see a dentist or doctor.", urgent=False),
    dict(key="toothache", q="a toothache", causes=["i have a toothache", "my tooth is killing me"],
         aliases=r"\btooth\s?ache\b|\b(tooth|teeth)\b",
         remedy="Please rinse your mouth with warm salt water and avoid very hot, very cold or sugary food. A cold cloth against your cheek can ease it, and keeping your head raised helps at night.",
         see="If your face is swollen or you have a fever, please see a dentist or doctor soon.", urgent=False),
    dict(key="back", q="back pain", causes=["my lower back hurts after lifting boxes", "my back hurts from sitting all day"],
         aliases=r"\b(back\s?ache|back\s+(pain|hurts|is\s+sore)|lower\s+back|my\s+back)\b",
         remedy="Please keep moving gently with short walks instead of long bed rest. A warm cloth, slow stretching, and a pillow between your knees when you lie on your side can help.",
         see="If you feel numbness or weakness, have trouble passing urine, or the pain followed a fall, please get medical help.", urgent=False),
    dict(key="period", q="period cramps", causes=["my period cramps are really bad today", "i have bad period pain"],
         aliases=r"\b(period|periods|menstrual)\b",
         remedy="Please place a warm cloth on your belly, rest, and sip warm drinks. Gentle walking or light stretching can also ease the cramps.",
         see="If the pain is very severe, the bleeding is very heavy, or you feel faint, please see a doctor.", urgent=False),
    dict(key="cramp", q="a muscle cramp", causes=["i got a cramp in my leg", "i have a cramp in my foot"],
         aliases=r"\bcramps?\b",
         remedy="Please stretch the muscle slowly and hold it gently, then massage it softly. A warm cloth and some water can help it relax.",
         see="If cramps keep coming back or come with swelling or weakness, please see a doctor.", urgent=False),
    dict(key="sore_muscles", q="sore muscles", causes=["my muscles are sore after the gym", "my legs are sore after a long walk"],
         aliases=r"\bmuscles?\b|\b(gym|workout)\b|\bsore\s+(legs|arms)\b|\b(legs|arms)\s+(are\s+)?sore\b",
         remedy="Please rest, drink water, and do some gentle stretching. After the first day, a warm bath or warm cloth can ease the stiffness.",
         see="If the pain is sharp, you cannot move the muscle properly, or it is very swollen, please see a doctor.", urgent=False),
    dict(key="joint", q="joint pain", causes=["my joints are stiff this morning", "my hip hurts when i walk"],
         aliases=r"\b(joint|joints|arthritis|hip|hips)\b",
         remedy="Please move the joint gently and rest it from heavy use. A warm cloth can ease stiffness, and a wrapped cold pack can ease swelling.",
         see="If the joint is red, hot, very swollen, or you cannot move it, please see a doctor.", urgent=False),
    dict(key="crush", q="a heavy object landing on the foot", causes=["i dropped a heavy book on my foot", "a box fell on my foot"],
         aliases=r"\b(dropped|fell\s+on|landed\s+on|crushed)\b.*\b(foot|feet|toe|toes|leg|hand)\b",
         remedy="Please stop what you are doing, sit down, and keep the foot still and raised. A cold pack wrapped in a cloth for a short while can ease the swelling, and please do not try to walk on it.",
         see="A heavy knock can break a bone, so please see a doctor today, and get help right away if it looks bent, is badly swollen, or you cannot move it.", urgent=True),
    dict(key="foot", q="foot pain", causes=["my feet hurt from my new shoes", "my heel hurts when i walk"],
         aliases=r"\b(foot|feet|heel|heels)\b",
         remedy="Please sit down, raise your feet, and soak them in lukewarm water. Wear comfortable, roomy shoes for a while.",
         see="If there is swelling or redness, or you cannot walk properly, please see a doctor.", urgent=False),
    dict(key="eye_object", q="something in the eye", causes=["something got into my eye", "i got dust in my eye"],
         aliases=r"\b(in|into)\s+my\s+eyes?\b|\b(dust|sand|dirt)\b.*\beye\b|\beye\b.*\b(irritated|watering|red)\b",
         remedy="Please do not rub it. Blink a few times, rinse your eye gently with clean water, and take out your contact lenses if you wear them.",
         see="If it was a chemical or a sharp object, or your vision changes or the pain stays, please get medical help right away.", urgent=True),
    dict(key="eye_strain", q="tired eyes", causes=["my eyes are tired from the screen"],
         aliases=r"\beyes?\s+(are\s+)?(tired|strained|dry|sore)\b|\beye\s?strain\b",
         remedy="Please look away from the screen at something far away for a while, blink often, and rest your eyes. A warm cloth over closed eyes can feel soothing.",
         see="If your vision changes or the pain stays, please see a doctor.", urgent=False),
    dict(key="ear", q="an earache", causes=["my ear hurts", "i have an earache"],
         aliases=r"\bear\s?ache\b|\bear\s+(hurts|pain|is\s+sore)\b|\bmy\s+ears?\b",
         remedy="Please rest and place a warm cloth over the ear. Do not put anything inside your ear.",
         see="If you have fluid coming out, a fever, or your hearing changes, please see a doctor.", urgent=False),
    dict(key="sore_throat", q="a sore throat", causes=["my throat is sore", "i have a sore throat"],
         aliases=r"\bsore\s+throat\b|\bthroat\b",
         remedy="Please sip warm drinks, gargle gently with a pinch of salt in warm water, and rest your voice. Warm water with honey can soothe it, but never give honey to a baby.",
         see="If it is hard to swallow or breathe, or it lasts more than several days, please see a doctor.", urgent=False),
    dict(key="cough", q="a cough", causes=["i have a bad cough"],
         aliases=r"\bcough\w*\b",
         remedy="Please sip warm drinks, rest, and breathe the steam from a warm shower. Warm water with honey can soothe it, but never give honey to a baby.",
         see="If you feel breathless, cough up blood, or the cough lingers, please see a doctor.", urgent=False),
    dict(key="cold", q="a cold", causes=["i have a blocked nose", "i have a runny nose"],
         aliases=r"\b(blocked|stuffy|runny)\s+nose\b|\b(have|got|caught)\s+(a\s+)?cold\b|\bcommon\s+cold\b",
         remedy="Please rest, drink warm fluids, and breathe the steam from a warm shower. Sleeping with your head raised can help you breathe more easily.",
         see="If you have trouble breathing, a high fever, or it does not improve, please see a doctor.", urgent=False),
    dict(key="fever", q="a fever", causes=["i have a fever"],
         aliases=r"\bfever\w*\b|\bhigh\s+temperature\b",
         remedy="Please rest, drink plenty of fluids, and wear light clothing. A lukewarm, not cold, damp cloth can feel comforting.",
         see="If the fever is high, lasts long, or comes with a stiff neck, confusion or trouble breathing, please get medical help, and a baby with a fever needs a doctor straight away.", urgent=False),
    dict(key="headache", q="a headache", causes=["my head has been hurting since this morning", "i have a pounding headache"],
         aliases=r"\bhead\s?ache\w*\b|\bmigraine\w*\b|\bhead\s+(hurts|is\s+pounding|is\s+killing)\b|\bmy\s+head\b",
         remedy="Please rest in a quiet, dim room and sip some water. A cool, damp cloth on your forehead or gentle shoulder stretches can also ease it.",
         see="If it is sudden and severe, follows a bump to the head, or comes with a fever, a stiff neck, confusion or vision changes, please get medical help right away.", urgent=False),
    dict(key="dizzy", q="dizziness", causes=["i feel dizzy"],
         aliases=r"\b(dizzy|dizziness|lightheaded|light\s+headed)\b",
         remedy="Please sit or lie down right away, and sip water slowly. When you get up again, rise slowly.",
         see="If you faint, have chest pain, or the dizziness does not pass, please get medical help.", urgent=False),
    dict(key="nausea", q="nausea", causes=["i feel nauseous"],
         aliases=r"\b(nausea|nauseous|nauseated|queasy|feel\s+sick|throwing\s+up|vomit\w*)\b",
         remedy="Please sit upright, take small sips of water, and get some fresh air. Ginger tea and plain, dry food can settle your stomach, and strong smells are best avoided.",
         see="If you cannot keep fluids down or feel very weak, please see a doctor.", urgent=False),
    dict(key="diarrhea", q="diarrhea", causes=["i have diarrhea"],
         aliases=r"\bdiarrh\w+\b|\bloose\s+(stools|motions)\b",
         remedy="Please drink plenty of fluids, ideally an oral rehydration drink, and keep to plain food such as rice or banana. Rest as much as you can.",
         see="If there is blood, you feel faint or very dry, or it does not settle, please see a doctor, and children and older people need help sooner.", urgent=False),
    dict(key="constipation", q="constipation", causes=["i am constipated"],
         aliases=r"\bconstipat\w+\b",
         remedy="Please drink more water, eat fibre-rich food like fruit, vegetables and whole grains, and take a gentle walk. A warm drink in the morning can help things move.",
         see="If it lasts for a long time, or comes with strong pain or blood, please see a doctor.", urgent=False),
    dict(key="heartburn", q="heartburn", causes=["i have heartburn after dinner"],
         aliases=r"\bheart\s?burn\b|\bacid\s+reflux\b|\bacidity\b",
         remedy="Please sit upright, eat smaller meals, and avoid lying down right after eating. Raising your head while you sleep and avoiding spicy or fatty food can help.",
         see="If you have chest pain, trouble swallowing, or it keeps happening, please see a doctor.", urgent=False),
    dict(key="stomach", q="a stomach ache", causes=["my stomach hurts after lunch", "i have a stomach ache"],
         aliases=r"\b(stomach|tummy|belly|abdominal)\b|\bindigestion\b",
         remedy="Please sit quietly, sip water slowly, and keep to light, plain food such as rice or toast. A warm cloth on your belly or some ginger tea can also help.",
         see="If the pain is severe, keeps getting worse, or comes with repeated vomiting, blood or a fever, please get medical help.", urgent=False),
    dict(key="skin_itch", q="itchy skin", causes=["my skin is itchy", "i have an itchy rash"],
         aliases=r"\b(itch|itchy|itching|rash)\b",
         remedy="Please keep your skin moisturised, use cool, damp cloths, and avoid scratching. A lukewarm bath and loose cotton clothes can help.",
         see="If the rash spreads fast, blisters, or comes with a fever or swelling, please see a doctor.", urgent=False),
    dict(key="insomnia", q="trouble sleeping", causes=[],
         aliases=r"\binsomnia\b|\btrouble\s+sleeping\b|\bdifficulty\s+sleeping\b",
         remedy="Please keep a regular bedtime, dim the lights, and put screens away before bed. A warm, caffeine-free drink and slow breathing can help you relax.",
         see="If sleep trouble goes on for a long time, please talk to a doctor.", urgent=False),
    dict(key="stress", q="stress or anxiety", causes=[],
         aliases=r"\b(anxiety|panic|stress|stressed)\b",
         remedy="Please take slow, deep breaths, go for a short walk, and write down what is on your mind. Talking to someone you trust also helps.",
         see="If it feels too heavy to handle, please talk to a doctor or a counsellor.", urgent=False),
]

_COMPILED = [(e, re.compile(e["aliases"])) for e in ENTRIES]

CONTEXT_TEMPLATE = ("\n\nRemedies you may share (use only these): {remedy}\n"
                    "When to see a doctor: {see}")


def find_remedy(text):
    """The first entry whose alias matches the text, or None."""
    norm = normalize(text)
    for entry, pattern in _COMPILED:
        if pattern.search(norm):
            return entry
    return None


# A person asking for a remedy we do not have. The 1.5B model cannot be trusted to refuse (v7 still invented
# remedies for hiccups and bad breath), so the app answers this case itself with a fixed line, like the safety gate.
ASKS_REMEDY_RE = re.compile(r"\b(remed(y|ies)|home\s+cure|tips?\s+(for|on|to)|get\s+rid\s+of|how\s+(do|can|should)\s+i\s+(ease|stop|treat|soothe|relieve|cure|fix)|"
                            r"what\s+(helps|can\s+i\s+do|should\s+i\s+do)\s+(with|for|about|at\s+home)|what\s+(helps|works)\b|natural\s+way)", re.I)
NO_REMEDY_LINE = ("I do not have a reliable remedy for that, and I would rather not guess. "
                  "A pharmacist or doctor can advise you properly. Is anything else bothering you?")


def asks_for_remedy(text):
    return bool(ASKS_REMEDY_RE.search(normalize(text)))


def remedy_context(entry):
    return CONTEXT_TEMPLATE.format(remedy=entry["remedy"], see=entry["see"])


def with_remedy(message, lookup_text=None):
    """Add the matching remedy to the message that goes to the LOCAL model. The same function builds the
    training prompts (extra_data.py), so training and real use look identical. No match: unchanged."""
    entry = find_remedy(lookup_text if lookup_text is not None else message)
    return message + remedy_context(entry) if entry else message


def check_entries():
    """Return a list of problems (empty = fine). Used by `python remedies.py` and the tests."""
    from output_guard import check_sentence
    problems, seen = [], set()
    for e in ENTRIES:
        if e["key"] in seen:
            problems.append(f"duplicate key {e['key']}")
        seen.add(e["key"])
        for field in ("remedy", "see"):
            sentences = [s for s in re.split(r"(?<=[.!?])\s+", e[field].strip()) if s]
            if field == "remedy" and not 1 <= len(sentences) <= 2:
                problems.append(f"{e['key']}: remedy has {len(sentences)} sentences (use one or two)")
            if field == "see" and len(sentences) != 1:
                problems.append(f"{e['key']}: 'see' must be one sentence")
            for s in sentences:
                reason = check_sentence(s)
                if reason:
                    problems.append(f"{e['key']}: guard would remove ({reason}): {s}")
        for cause in e["causes"]:
            found = find_remedy(cause)
            if not found or found["key"] != e["key"]:
                problems.append(f"{e['key']}: example '{cause}' finds '{found['key'] if found else None}' instead")
        if not re.search(r"doctor|medical|dentist|emergency", e["see"]):
            problems.append(f"{e['key']}: 'see' does not point to a doctor or medical help")
    return problems


if __name__ == "__main__":
    problems = check_entries()
    if problems:
        print("Problems:")
        for p in problems:
            print("  -", p)
        sys.exit(1)
    print(f"{len(ENTRIES)} remedy entries, all clean.")
