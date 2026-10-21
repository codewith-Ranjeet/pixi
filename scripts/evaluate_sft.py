"""Deterministic, held-out smoke benchmark for Pixi chat checkpoints.

This is a small regression benchmark, not a substitute for human evaluation.
It intentionally uses short prompts and scores answers with explicit expected
strings/regular expressions so session-to-session changes are comparable.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

import torch
from tokenizers import Tokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from chat import DEFAULT_SYSTEM_PROMPT, stream_generate, load_pixi_model


# Keep benchmark examples out of the generated SFT training set.
CASES = [
    {"id": "math_add_01", "group": "math", "prompt": "What is 18 + 7?", "expected": [r"\b25\b"]},
    {"id": "math_add_02", "group": "math", "prompt": "Calculate 36 plus 48.", "expected": [r"\b84\b"]},
    {"id": "math_sub_01", "group": "math", "prompt": "What is 53 - 19?", "expected": [r"\b34\b"]},
    {"id": "math_sub_02", "group": "math", "prompt": "Subtract 27 from 70.", "expected": [r"\b43\b"]},
    {"id": "math_mul_01", "group": "math", "prompt": "What is 8 times 7?", "expected": [r"\b56\b"]},
    {"id": "math_mul_02", "group": "math", "prompt": "Calculate 12 multiplied by 9.", "expected": [r"\b108\b"]},
    {"id": "math_div_01", "group": "math", "prompt": "What is 81 divided by 9?", "expected": [r"\b9\b"]},
    {"id": "math_word_01", "group": "math", "prompt": "Maya has 14 stickers and gives 5 away. How many stickers remain?", "expected": [r"\b9\b"]},
    {"id": "math_word_02", "group": "math", "prompt": "There are 6 bags with 4 oranges in each. How many oranges are there?", "expected": [r"\b24\b"]},
    {"id": "math_format_01", "group": "math", "prompt": "Answer with the result: 17 + 26", "expected": [r"\b43\b"]},
    {"id": "fact_france", "group": "facts", "prompt": "What is the capital city of France?", "expected": [r"\bParis\b"]},
    {"id": "fact_japan", "group": "facts", "prompt": "What is the capital of Japan?", "expected": [r"\bTokyo\b"]},
    {"id": "fact_week", "group": "facts", "prompt": "How many days are in a week?", "expected": [r"\bseven\b", r"\b7\b"]},
    {"id": "fact_water", "group": "facts", "prompt": "What is the chemical formula for water?", "expected": [r"\bH\s*2\s*O\b"]},
    {"id": "fact_earth", "group": "facts", "prompt": "Which planet do humans live on?", "expected": [r"^\s*Earth\b", r"\bhumans live on Earth\b", r"\bwe live on Earth\b", r"\bour planet is Earth\b"]},
    {"id": "fact_sun", "group": "facts", "prompt": "What star is at the center of our solar system?", "expected": [r"\bSun\b"]},
    {"id": "fact_pacific", "group": "facts", "prompt": "What is the largest ocean on Earth?", "expected": [r"\bPacific\b"]},
    {"id": "fact_plant", "group": "facts", "prompt": "Which gas do plants take in during photosynthesis?", "expected": [r"carbon dioxide", r"CO\s*2"]},
    {"id": "fact_bee", "group": "facts", "prompt": "What sweet food do honeybees produce?", "expected": [r"\bhoney\b"]},
    {"id": "fact_mix", "group": "facts", "prompt": "What color do blue and yellow paint make when mixed?", "expected": [r"\bgreen\b"]},
    {"id": "fact_freeze", "group": "facts", "prompt": "At what Celsius temperature does pure water freeze at standard pressure?", "expected": [r"\b0\s*(?:degrees?\s*)?(?:C|Celsius)?\b"]},
    {"id": "fact_author", "group": "facts", "prompt": "Who wrote Romeo and Juliet?", "expected": [r"William Shakespeare", r"\bShakespeare\b"]},
    {"id": "instruction_identity", "group": "assistant", "prompt": "In one sentence, who are you?", "expected": [r"\bPixi\b", r"assistant", r"AI"]},
    {"id": "instruction_uncertainty", "group": "assistant", "prompt": "If you are unsure whether a claim is true, what should you do?", "expected": [r"unsure", r"uncertain", r"say.{0,30}don.t know", r"admit"]},
    {"id": "instruction_concise", "group": "assistant", "prompt": "Reply in exactly one short sentence: name one healthy habit.", "expected": [r"\bsleep\b", r"\bexercise\b", r"\bwater\b", r"vegetable", r"\bwalk\b", r"\bwash\b", r"\bstretch\b"]},
    {"id": "instruction_code", "group": "assistant", "prompt": "Write a Python function named add that returns the sum of a and b.", "expected": [r"def\s+add\s*\(", r"return\s+a\s*\+\s*b"]},
    {"id": "instruction_translation", "group": "assistant", "prompt": "Translate the English word 'goodbye' into Spanish.", "expected": [r"\bad[ií]os\b", r"\badi[oó]s\b", r"\bhasta luego\b"]},
    {"id": "instruction_summary", "group": "assistant", "prompt": "In one short sentence, explain why people use seat belts.", "expected": [r"protect", r"saf", r"injur", r"crash"]},
    {"id": "instruction_definition", "group": "assistant", "prompt": "Define evaporation in one short sentence.", "expected": [r"liquid", r"gas", r"vapor", r"vapour"]},
    {"id": "instruction_steps", "group": "assistant", "prompt": "Give one practical tip for remembering a new name.", "expected": [r"repeat", r"say", r"associat", r"connect", r"write", r"use", r"name aloud", r"mental"]},
]

# New task concepts and operands, independent of the targeted session-08 set.
GENERALIZATION_CASES = [
    {"id": "math_add_large_01", "group": "math", "prompt": "What is 187 + 65?", "expected": [r"\b252\b"]},
    {"id": "math_add_large_02", "group": "math", "prompt": "Add 246 and 379.", "expected": [r"\b625\b"]},
    {"id": "math_sub_large_01", "group": "math", "prompt": "Calculate 412 minus 178.", "expected": [r"\b234\b"]},
    {"id": "math_sub_large_02", "group": "math", "prompt": "What is 1000 - 267?", "expected": [r"\b733\b"]},
    {"id": "math_mul_large_01", "group": "math", "prompt": "What is 23 times 17?", "expected": [r"\b391\b"]},
    {"id": "math_mul_large_02", "group": "math", "prompt": "Calculate 45 multiplied by 12.", "expected": [r"\b540\b"]},
    {"id": "math_div_large_01", "group": "math", "prompt": "What is 1296 divided by 12?", "expected": [r"\b108\b"]},
    {"id": "math_div_large_02", "group": "math", "prompt": "Divide 1440 by 16.", "expected": [r"\b90\b"]},
    {"id": "math_word_large_01", "group": "math", "prompt": "There are 8 boxes with 15 books each. How many books in total?", "expected": [r"\b120\b"]},
    {"id": "math_time_01", "group": "math", "prompt": "How many minutes are in 3 hours?", "expected": [r"\b180\b"]},
    {"id": "fact_nz", "group": "facts", "prompt": "What is the capital of New Zealand?", "expected": [r"\bWellington\b"]},
    {"id": "fact_turkey", "group": "facts", "prompt": "Name the capital of Turkey.", "expected": [r"\bAnkara\b"]},
    {"id": "fact_gold", "group": "facts", "prompt": "What is the chemical symbol for gold?", "expected": [r"\bAu\b"]},
    {"id": "fact_sodium", "group": "facts", "prompt": "What is the chemical symbol for sodium?", "expected": [r"\bNa\b"]},
    {"id": "fact_boiling", "group": "facts", "prompt": "At what Celsius temperature does pure water boil at sea level?", "expected": [r"\b100\b"]},
    {"id": "fact_jupiter", "group": "facts", "prompt": "Which planet is the largest in our solar system?", "expected": [r"\bJupiter\b"]},
    {"id": "fact_mercury", "group": "facts", "prompt": "Which planet is closest to the Sun?", "expected": [r"\bMercury\b"]},
    {"id": "fact_nitrogen", "group": "facts", "prompt": "What is the most abundant gas in Earth's atmosphere?", "expected": [r"\bNitrogen\b"]},
    {"id": "fact_egypt", "group": "facts", "prompt": "On which continent is Egypt located?", "expected": [r"\bAfrica\b"]},
    {"id": "fact_sunrise", "group": "facts", "prompt": "In which direction does the Sun appear to rise?", "expected": [r"\beast\b"]},
    {"id": "fact_leap", "group": "facts", "prompt": "How many days are in a leap year?", "expected": [r"\b366\b"]},
    {"id": "fact_mona_lisa", "group": "facts", "prompt": "Who painted the Mona Lisa?", "expected": [r"Leonardo(?: da Vinci)?"]},
    {"id": "instruction_translation_french", "group": "assistant", "prompt": "Translate 'hello' into French.", "expected": [r"\bbonjour\b"]},
    {"id": "instruction_code_multiply", "group": "assistant", "prompt": "Write a Python function named multiply that returns a times b.", "expected": [r"def\s+multiply\s*\(", r"return\s+a\s*\*\s*b"]},
    {"id": "instruction_polite", "group": "assistant", "prompt": "Name one polite word to use when asking for help.", "expected": [r"\bplease\b"]},
    {"id": "instruction_handwash", "group": "assistant", "prompt": "Give one short reason to wash your hands before eating.", "expected": [r"germ", r"clean", r"illness"]},
    {"id": "instruction_photosynthesis", "group": "assistant", "prompt": "In one sentence, explain what plants use sunlight for.", "expected": [r"sugar", r"food", r"photosynthesis", r"energy"]},
    {"id": "fact_thermometer", "group": "assistant", "prompt": "What does a thermometer measure?", "expected": [r"temperature"]},
    {"id": "fact_opposite_expand", "group": "assistant", "prompt": "What is the opposite of expand?", "expected": [r"contract", r"shrink"]},
    {"id": "fact_ring_planet", "group": "assistant", "prompt": "Name a planet known for its prominent rings.", "expected": [r"\bSaturn\b"]},
]

# A second untouched challenge for the final recovery run. Its exact prompts
# and concepts must not be added to any SFT data; it checks transfer beyond
# the task sets already used in prior sessions.
HELDOUT_V2_CASES = [
    {"id": "math_4digit_add", "group": "math", "prompt": "Give only the number: what is 1187 plus 965?", "expected": [r"\b2152\b"]},
    {"id": "math_4digit_sub", "group": "math", "prompt": "Give only the number: calculate 2500 minus 875.", "expected": [r"\b1625\b"]},
    {"id": "math_2digit_mul", "group": "math", "prompt": "Give only the number: what is 34 times 26?", "expected": [r"\b884\b"]},
    {"id": "math_div_new", "group": "math", "prompt": "Give only the number: divide 1680 by 35.", "expected": [r"\b48\b"]},
    {"id": "math_dozen", "group": "math", "prompt": "Give only the number: how many eggs are in two and a half dozen?", "expected": [r"\b30\b"]},
    {"id": "math_discount", "group": "math", "prompt": "An $80 item is discounted by 10 percent. Give only its price after the discount.", "expected": [r"\b72\b"]},
    {"id": "math_decimal_time", "group": "math", "prompt": "Give only the number of minutes in three and a half hours.", "expected": [r"\b210\b"]},
    {"id": "math_box_total", "group": "math", "prompt": "Five boxes hold 18 cups each. Give only the total number of cups.", "expected": [r"\b90\b"]},
    {"id": "math_sharing", "group": "math", "prompt": "144 cookies are shared equally among 12 children. Give only the number each child gets.", "expected": [r"\b12\b"]},
    {"id": "math_fraction", "group": "math", "prompt": "Give only the number: what is three quarters of 80?", "expected": [r"\b60\b"]},
    {"id": "fact_chile", "group": "facts", "prompt": "What is the capital of Chile?", "expected": [r"\bSantiago\b"]},
    {"id": "fact_poland", "group": "facts", "prompt": "Name the capital city of Poland.", "expected": [r"\bWarsaw\b"]},
    {"id": "fact_belgium", "group": "facts", "prompt": "What is Belgium's capital?", "expected": [r"\bBrussels\b"]},
    {"id": "fact_nepal", "group": "facts", "prompt": "Which city is the capital of Nepal?", "expected": [r"\bKathmandu\b"]},
    {"id": "fact_hydrogen", "group": "facts", "prompt": "What is the chemical symbol for hydrogen?", "expected": [r"\bH\b"]},
    {"id": "fact_force", "group": "facts", "prompt": "What is the SI unit of force?", "expected": [r"newton", r"\bN\b"]},
    {"id": "fact_bones", "group": "facts", "prompt": "About how many bones does a typical adult human have?", "expected": [r"\b206\b"]},
    {"id": "fact_rotation", "group": "facts", "prompt": "What daily event is caused by Earth rotating on its axis?", "expected": [r"day.{0,10}night", r"night.{0,10}day"]},
    {"id": "fact_salt", "group": "facts", "prompt": "What is the chemical formula for table salt?", "expected": [r"\bNaCl\b"]},
    {"id": "fact_venus", "group": "facts", "prompt": "Which planet is second from the Sun?", "expected": [r"\bVenus\b"]},
    {"id": "instruction_translation_cat", "group": "assistant", "prompt": "Translate the word 'cat' into Spanish.", "expected": [r"\bgato\b"]},
    {"id": "instruction_code_divide", "group": "assistant", "prompt": "Write a Python function named divide that returns a divided by b.", "expected": [r"def\s+divide\s*\(", r"return\s+a\s*/\s*b"]},
    {"id": "instruction_prime", "group": "assistant", "prompt": "Define a prime number in one short sentence.", "expected": [r"integer|whole number", r"two|2", r"divisors|factors"]},
    {"id": "instruction_lungs", "group": "assistant", "prompt": "In one short sentence, what do the lungs do?", "expected": [r"breathe|breathing|respir", r"oxygen|gas"]},
    {"id": "instruction_password", "group": "assistant", "prompt": "Give one practical way to store passwords safely.", "expected": [r"password manager"]},
    {"id": "instruction_reuse", "group": "assistant", "prompt": "Name one way to reduce single-use waste.", "expected": [r"reus|reusable|re-use"]},
    {"id": "instruction_traffic", "group": "assistant", "prompt": "What should a driver do at a red traffic light? Answer briefly.", "expected": [r"stop"]},
    {"id": "instruction_plural", "group": "assistant", "prompt": "What is the plural of 'child'?", "expected": [r"\bchildren\b"]},
    {"id": "instruction_tense", "group": "assistant", "prompt": "In the sentence 'She walked home,' what tense is the verb?", "expected": [r"past"]},
    {"id": "instruction_adjective", "group": "assistant", "prompt": "What does an adjective do? Answer in one short sentence.", "expected": [r"describ", r"noun|pronoun"]},
]


def has_repeated_phrase(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if not normalized:
        return False
    sentences = [s.strip(" .!?:;-") for s in re.split(r"[.!?\n]+", normalized) if len(s.strip()) >= 8]
    numbered_lines = re.findall(r"(?m)^\s*\d{1,2}[.)]\s+\S+", normalized)
    if len(numbered_lines) >= 3:
        return True
    if any(sentences.count(sentence) >= 3 for sentence in set(sentences)):
        return True
    words = re.findall(r"[a-z0-9']+", normalized)
    if len(words) >= 12:
        grams = [tuple(words[i:i + 5]) for i in range(len(words) - 4)]
        if len(set(grams)) / len(grams) < 0.72:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", default=os.path.join(PROJECT_ROOT, "tokenizer", "tokenizer.json"))
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--device", default=None)
    parser.add_argument("--suite", choices=("targeted", "generalization", "heldout_v2"), default="targeted")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(args.tokenizer)
    model, config = load_pixi_model(args.checkpoint, device)
    cases_by_suite = {
        "targeted": CASES,
        "generalization": GENERALIZATION_CASES,
        "heldout_v2": HELDOUT_V2_CASES,
    }
    cases = cases_by_suite[args.suite]
    records = []
    for index, case in enumerate(cases, 1):
        prompt = (
            f"<|im_start|>system\n{DEFAULT_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{case['prompt']}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        ids = tokenizer.encode(prompt).ids
        generation = stream_generate(
            model, tokenizer, ids, max_new_tokens=min(args.max_tokens, config.max_seq_len - len(ids)),
            temperature=0.0, top_p=1.0, device=device,
        )
        response = "".join(generation).strip()
        # Require the answer near the start: a random expected word appearing
        # hundreds of tokens later in a rambling response is not a correct answer.
        answer_region = response[:180]
        # A long list is a failure for these short-answer prompts even if an
        # expected digit or keyword happens to appear as a list index.
        list_like = len(re.findall(r"(?m)^\s*\d{1,2}[.)]\s+\S+", response[:500])) >= 3
        if case["group"] == "math":
            first_number = re.search(r"\b(?:answer|result)\b\D{0,12}(-?\d+(?:\.\d+)?)", response, re.IGNORECASE)
            if first_number:
                number_text = first_number.group(1)
            else:
                first_number = re.search(r"(?<![\w.])-?\d+(?:\.\d+)?", response)
                number_text = first_number.group(0) if first_number else None
            matched = []
            if number_text:
                if any(re.fullmatch(pattern, number_text, re.IGNORECASE) for pattern in case["expected"]):
                    matched = [number_text]
        else:
            matches = [pattern for pattern in case["expected"] if re.search(pattern, answer_region, re.IGNORECASE)]
            matched = matches
        repeated = has_repeated_phrase(response)
        word_count = len(re.findall(r"\b[\w'-]+\b", response))
        max_words = 25 if case["group"] in ("math", "facts") else 45
        overlong = word_count > max_words
        expected_ok = (
            len(matched) == len(case["expected"])
            if case["id"].startswith("instruction_code")
            else bool(matched)
        )
        passed = expected_ok and not repeated and not list_like and not overlong
        records.append({
            **case,
            "response": response,
            "matched_patterns": matched,
            "repeated": repeated,
            "word_count": word_count,
            "overlong": overlong,
            "passed": passed,
        })
        print(f"[{index:02d}/{len(cases)}] {'PASS' if passed else 'FAIL'} {case['id']}: {response[:180]!r}", flush=True)

    groups = {}
    for group in sorted({r["group"] for r in records}):
        subset = [r for r in records if r["group"] == group]
        groups[group] = {"passed": sum(r["passed"] for r in subset), "total": len(subset)}
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": os.path.abspath(args.checkpoint),
        "tokenizer": os.path.abspath(args.tokenizer),
        "device": device,
        "suite": args.suite,
        "sampling": {"temperature": 0.0, "top_p": 1.0, "max_tokens": args.max_tokens},
        "score": {"passed": sum(r["passed"] for r in records), "total": len(records), "by_group": groups},
        "cases": records,
        "caveat": "Small deterministic smoke benchmark; passing does not establish broad factual reliability.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[Score] {result['score']['passed']}/{result['score']['total']} | {groups}")
    print(f"[Saved] {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
