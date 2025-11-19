import argparse
import json
import sys
import os
import urllib.request
import urllib.error
import time
import socket
import re
import math
from collections import Counter


# IMPORTANT: Set your OpenAI API key using one of these methods:
# 1. Create a file named 'openai_api_key.txt' in the same directory as this script with just your API key
# 2. Set environment variable: export OPENAI_API_KEY="sk-..."
# 3. Set inline: OPENAI_API_KEY="sk-..." python translation_ai_agent.py ...

def load_api_key():
    """Load OpenAI API key from file or environment variable."""
    # First, try to load from file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    api_key_file = os.path.join(script_dir, "openai_api_key.txt")
    
    if os.path.exists(api_key_file):
        try:
            with open(api_key_file, 'r', encoding='utf-8') as f:
                key = f.read().strip()
                if key:
                    return key
        except Exception as e:
            print(f"Warning: Could not read API key from file: {e}")
    
    # Fall back to environment variable
    return os.getenv("OPENAI_API_KEY")

OPENAI_API_KEY = load_api_key()


def call_openai_chat(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.7, timeout: int = 20, retries: int = 3, backoff_sec: float = 2.0) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Connection": "close",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }

    data = json.dumps(body).encode("utf-8")
    last_err = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            socket.setdefaulttimeout(timeout)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_text = resp.read().decode("utf-8")
                parsed = json.loads(resp_text)
                try:
                    return parsed["choices"][0]["message"]["content"].strip()
                except Exception as e:
                    raise RuntimeError(f"Unexpected API response: {json.dumps(parsed)[:1000]}") from e
        except urllib.error.HTTPError as e:
            err_text = e.read().decode("utf-8", errors="replace")
            last_err = RuntimeError(f"OpenAI API HTTPError {e.code}: {err_text}")
        except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
            last_err = RuntimeError(f"OpenAI API network/timeout error: {e}")
        except Exception as e:
            last_err = RuntimeError(f"OpenAI API unknown error: {e}")

        if attempt < retries:
            time.sleep(backoff_sec * attempt)
        else:
            if last_err:
                raise last_err

    raise RuntimeError("OpenAI API: exhausted retries")


def load_ground_truth_translations(filepath: str) -> dict:
    """Load ground truth translations and create a mapping from ASL gloss to English translation."""
    translations = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or ' → ' not in line:
                continue
            asl_part, english_part = line.split(' → ', 1)
            # Clean up the ASL part (remove extra spaces, normalize)
            asl_clean = ' '.join(asl_part.split())
            translations[asl_clean] = english_part.strip()
    return translations


def normalize_spacing(text: str) -> str:
    """Normalize spacing around parentheses, commas, and other punctuation."""
    # Remove extra spaces around parentheses
    text = re.sub(r'\s*\(\s*', '(', text)
    text = re.sub(r'\s*\)\s*', ')', text)
    # Remove extra spaces around commas
    text = re.sub(r'\s*,\s*', ',', text)
    # Normalize multiple spaces to single space
    text = ' '.join(text.split())
    return text


def preprocess_text(text: str) -> str:
    """Clean and preprocess text for BLEU calculation"""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters that might interfere with tokenization
    text = re.sub(r'[^\w\s\?\.\!]', '', text)
    return text


def get_ngrams(tokens: list, n: int) -> list:
    """Get n-grams from a list of tokens"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


def calculate_bleu_score_simple(reference: str, hypothesis: str) -> float:
    """Calculate BLEU score using simple implementation"""
    # Tokenize
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    
    if len(hyp_tokens) == 0:
        return 0.0
    
    # Calculate precision for n-grams (1-4)
    precisions = []
    for n in range(1, 5):
        ref_ngrams = get_ngrams(ref_tokens, n)
        hyp_ngrams = get_ngrams(hyp_tokens, n)
        
        if len(hyp_ngrams) == 0:
            precisions.append(0.0)
            continue
        
        # Count matches
        ref_counter = Counter(ref_ngrams)
        hyp_counter = Counter(hyp_ngrams)
        
        matches = 0
        for ngram in hyp_counter:
            matches += min(hyp_counter[ngram], ref_counter[ngram])
        
        precision = matches / len(hyp_ngrams)
        precisions.append(precision)
    
    # Calculate brevity penalty
    if len(hyp_tokens) < len(ref_tokens):
        bp = math.exp(1 - len(ref_tokens) / len(hyp_tokens))
    else:
        bp = 1.0
    
    # Calculate BLEU score
    if any(p == 0 for p in precisions):
        return 0.0
    
    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    return bleu


def find_ground_truth_translation(asl_sentence: str, translations: dict) -> str:
    """Find the ground truth translation for an ASL sentence."""
    # Normalize the input sentence
    asl_normalized = normalize_spacing(asl_sentence)
    
    # Try exact match first
    if asl_sentence in translations:
        return translations[asl_sentence]
    if asl_normalized in translations:
        return translations[asl_normalized]
    
    # Collect all potential matches with their similarity scores
    potential_matches = []
    
    for key, value in translations.items():
        key_normalized = normalize_spacing(key)
        
        # Exact match (highest priority)
        if asl_normalized == key_normalized:
            return value
        
        # Case-insensitive exact match (high priority)
        if asl_normalized.lower() == key_normalized.lower():
            return value
        
        # Calculate similarity score for partial matches
        similarity_score = 0
        
        # Check if input is contained in key (prefer more specific matches)
        if asl_normalized.lower() in key_normalized.lower():
            similarity_score = len(asl_normalized) / len(key_normalized)
            potential_matches.append((similarity_score, value))
        
        # Check if key is contained in input (less preferred)
        elif key_normalized.lower() in asl_normalized.lower():
            similarity_score = len(key_normalized) / len(asl_normalized) * 0.5  # Lower score
            potential_matches.append((similarity_score, value))
    
    # Return the best match (highest similarity score)
    if potential_matches:
        # Sort by similarity score (descending) and return the best match
        potential_matches.sort(key=lambda x: x[0], reverse=True)
        return potential_matches[0][1]
    
    # If no match found, return a placeholder
    return "GROUND_TRUTH_NOT_FOUND"


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a prompt (or many from a file) to OpenAI and save answers to a text file.")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt string. If omitted and --input_file not set, read from stdin.")
    parser.add_argument("--input_file", type=str, default=None, help="Path to input file. If filename contains 'alignment_without_facial_exp.txt', each line is ASL gloss. Otherwise, format is 'GROUND_TRUTH - ASL_GLOSS'.")
    parser.add_argument("--ground_truth_file", type=str, default="gnd_truth_translations.txt", help="Path to ground truth translations file.")
    parser.add_argument("--output", type=str, default="ground_truth_vs_llm_comparison.txt", help="Path to output file (format: ground_truth_translation - LLM_translation).")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model, e.g., gpt-4o-mini.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0-2).")
    parser.add_argument("--timeout", type=int, default=20, help="Per-request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts per request.")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("ERROR: Please set your OpenAI API key using one of these methods:")
        print("1. Create a file 'openai_api_key.txt' in the same directory as this script")
        print("   with just your API key (e.g., 'sk-proj-...')")
        print("2. Set environment variable: export OPENAI_API_KEY='sk-...'")
        print("3. Set inline: OPENAI_API_KEY='sk-...' python translation_ai_agent.py ...")
        sys.exit(1)

    # Batch mode: read aligned_sentences file and send the text after '-'
    if args.input_file:
        # Load ground truth translations
        print("Loading ground truth translations...")
        ground_truth_translations = load_ground_truth_translations(args.ground_truth_file)
        print(f"Loaded {len(ground_truth_translations)} ground truth translations")
        
        # Determine input format based on filename
        input_filename = os.path.basename(args.input_file)
        is_simple_format = (
            "alignment_without_facial_exp.txt" in input_filename
            or "alignment_without_LM.txt" in input_filename
        )
        stop_on_blank_line = "alignment_without_LM.txt" in input_filename
        
        total = 0
        success = 0
        bleu_scores = []  # Store BLEU scores for each translation pair
        
        with open(args.input_file, "r", encoding="utf-8") as fin, \
             open(args.output, "w", encoding="utf-8") as fout:
            
            for line in fin:
                line = line.strip()
                if not line:
                    if stop_on_blank_line:
                        print("Encountered blank line; stopping further processing.")
                        break
                    continue
                total += 1
                
                # Parse input based on file format
                if is_simple_format:
                    # Format: "Ground_truth_ASL_gloss - recognized_ASL"
                    # Example: "MY FATHER WANT(sad) SELL(sad) HIS CAR(mm) - MY FATHER WANT SELL HIS CAR"
                    if " - " in line:
                        try:
                            ground_truth_gloss, recognized_gloss = line.split(" - ", 1)
                            prompt_text = recognized_gloss.strip()  # Send recognized ASL to AI
                            asl_gloss_for_matching = ground_truth_gloss.strip()  # Use ground truth ASL for matching
                        except ValueError:
                            # If parsing fails, treat entire line as recognized ASL
                            prompt_text = line.strip()
                            asl_gloss_for_matching = line.strip()
                    else:
                        # No separator, treat entire line as recognized ASL
                        prompt_text = line.strip()
                        asl_gloss_for_matching = line.strip()
                else:
                    # Format: "GROUND TRUTH - ALIGNED SENTENCE"
                    if " - " in line:
                        try:
                            ground_truth_part, after = line.split(" - ", 1)
                            prompt_text = after.strip()
                            asl_gloss_for_matching = prompt_text  # Use the ASL part for matching
                        except ValueError:
                            prompt_text = line.strip()
                            asl_gloss_for_matching = prompt_text
                    else:
                        prompt_text = line.strip()
                        asl_gloss_for_matching = prompt_text
                
                try:
                    prompt1 = '''Imagine you are a sign language interpreter. You are an ASL translator converting gloss notation to natural English sentences.
                                
                                Two most important rules: 
                                1. Dont add any sign words that are not in the sentence.
                                2. In sentences like "BOOK (raise) Classifier (mm) YOU BUY(th) AGAIN (raise)", 
                                you need to associate the expression with the immeditae facial expression i.e., BUY and not with any prior or later word in the sentence.
                                Notation Guide
                                Non-Manual Markers (NMMs) - in parentheses after signs:

                                (raise): Question/conditional clause
                                (happy): Positive emotion
                                (sad): Negative emotion/regret
                                (angry): Frustration/emphasis
                                (furrow): WH-question marker
                                (shake): Negation ("don't/doesn't")

                                Verb Modifiers:

                                (cs): Recently completed ("recently")
                                (th): Carelessly done ("carelessly/sloppily")
                                (mm): Routine action ("normally/routinely/regularly")

                                Classifiers: (mm) = medium-sized, (cha) = large-sized
                                Translation Rules

                                If (raise) is associated with the first/ second word of the sentence, it just means topicalisation but 
                                if it anywhere else = Yes/No question or "if" conditional
                                (furrow) + WHY = "Why" question
                                (shake) = negation
                                Reorder ASL Topic-Comment → English Subject-Verb-Object
                                Integrate emotions and modifiers naturally (like below) 
                                

                                Example Translations
                                BOOK (raise) Classifier (raise, mm) YOU BUY AGAIN → You bought the medium-size book again.
                                BOOK (raise) Classifier (raise, cha) YOU (happy) BUY(happy) AGAIN(happy) → You bought the large-size book again.
                                BOOK (raise) Classifier (raise, mm) YOU (angry) BUY(angry) AGAIN(raise) → Did you buy the medium-sized book again?
                                BOOK (raise) Classifier (cha) YOU BUY(th) AGAIN (sad) → You carelessly bought the large-size book again.
                                BOOK (raise) Classifier (mm) YOU BUY(cs) AGAIN (happy) → You recently bought the medium-size book again.
                                BOOK (raise) Classifier (cha) YOU BUY(cs) AGAIN (angry) → You recently bought the large-size book again.
                                BOOK (raise) Classifier (mm) YOU BUY(th) AGAIN (sad) → You carelessly bought the medium-size book again.
                                BOOK (raise) Classifier (mm) YOU BUY(th) AGAIN (raise) → Did you carelessly buy the medium-sized book again?
                                BOOK (raise) Classifier (cha) YOU BUY(cs) AGAIN → You recently bought the large-size book again.
                                MY FATHER WANT(shake) SELL HIS CAR → My father don't want to sell his car
                                MY FATHER WANT(shake) SELL HIS CAR (cha) → My father don't want to sell his large car
                                MY FATHER WANT SELL(cs) HIS (happy) CAR (happy) → My father recently wants to sell his car
                                MY FATHER WANT SELL HIS (angry) CAR (angry) → My father wants to sell his car
                                MY FATHER WANT (sad) SELL (sad) HIS CAR (mm) → My father recently wants to sell his midium-size car
                                MY FATHER WANT SELL (cs) HIS CAR (raise) → Does my father recently want to sell his car?
                                MY FATHER (sad) WANT (sad) SELL (sad) HIS CAR (raise) → Does my father want to sell his car?
                                MY FATHER (angry) WANT (angry) SELL (cs) HIS CAR (raise) → Does my father recently want to sell his car?
                                MY FATHER (happy) WANT (happy) SELL (happy) HIS CAR (raise) → Does my father want to sell his car?
                                I (rasie) DRIVE (rasie) HOME (rasie) FUTURE HE HAPPY (happy) → If I drive home, he will be happy
                                I (rasie) DRIVE (rasie, th) HOME FUTURE HE angry (angry) → If I drive home carelessly, he will be angry
                                I (rasie) DRIVE (rasie, mm) HOME FUTURE HE sad (sad) → If I drive home routinely, he will be sad
                                I (rasie) DRIVE (rasie, cs) HOME FUTURE HE happy (happy) → If I drive home recently, he will be happy
                                I (rasie) DRIVE (rasie) HOME (raise, cs) FUTURE HE angry (angry) → If I drive close-distance home, he will be angry
                                I (rasie) DRIVE (rasie) HOME (rasie) FUTURE HE ANGRY (angry) → If I drive home, he will be angry
                                I (rasie) DRIVE (rasie) HOME (rasie) FUTURE HE SAD (sad) → If I drive home, he will be sad
                                MY FRIEND WANT GO-OUT IF(raise) STORE OPEN → My friend wants to go out if store is open
                                MY FRIEND WANT GO-OUT (happy) IF(raise) STORE (cs) OPEN → My friend wants to go out if near store is open
                                MY FRIEND WANT (shake) GO-OUT (sad) IF(raise) STORE (cha) CLOSE → My friend don't want to go out if the big store is closed.
                                MY FRIEND WANT (shake) GO-OUT (angry) IF(raise) STORE (cha) CLOSE → My friend don't want to go out if the big store is closed.
                                MY FRIEND WANT (shake) GO-OUT IF(raise) STORE (cha) CLOSE → My friend don't want to go out if the big store is closed.
                                MY FRIEND WANT (shake) GO-OUT (mm) IF(raise) STORE (happy) OPEN (happy) → My friend don't want to go out regularly/routinely if the store opens.
                                MY FRIEND WANT (shake) GO-OUT (cs) IF(raise) STORE (angry) CLOSE (angry) → My friend don't want to go out regularly/routinely if the store is closing soon.
                                MY FRIEND WANT (shake) GO-OUT (sad) IF(raise) STORE CLOSE (cs) → My friend don't want to go out if the store is recently closed.
                                HE HERE ALONE WHY (raise) YOU LEAVE → He is here alone because you leave
                                HE HERE ALONE (happy) WHY (raise) YOU LEAVE → He is here alone because you leave
                                HE HERE ALONE (sad) WHY (raise) YOU LEAVE → He is here alone because you leave
                                HE HERE ALONE (angry) WHY (raise) YOU LEAVE (cs) → He is here alone because you leave recently
                                HE HERE ALONE (sad) WHY (raise) YOU LEAVE (cs) → He is here alone because you leave recently
                                HE HERE ALONE (happy) WHY (raise) YOU LEAVE (cs) → He is here alone because you leave recently
                                HE HERE ALONE (angry) WHY (raise) YOU LEAVE (th) → He is here alone because you leave carelessly
                                HE HERE ALONE (sad) WHY (raise) YOU LEAVE (th) → He is here alone because you leave carelessly
                                HE HERE ALONE (happy) WHY (raise) YOU LEAVE (mm) → He is here alone because you leave routinely
                                MY MOTHER ARRIVE LATE WHY (furrow) → Why did my mother arrive late?
                                MY MOTHER ARRIVE (angry) LATE WHY (furrow) → Why did my mother arrive late?
                                MY MOTHER ARRIVE (sad) LATE WHY (furrow) → Why did my mother arrive late?
                                MY MOTHER ARRIVE (angry) LATE (cha) WHY (furrow) → Why did my mother arrive very late?
                                MY MOTHER ARRIVE (sad) LATE (cha) WHY (furrow) → Why did my mother arrive very late?
                                MY MOTHER ARRIVE (th) LATE WHY (furrow) → Why did my mother arrive late carelessly?
                                MY MOTHER ARRIVE (th) LATE (angry) WHY (furrow) → Why did my mother arrive late carelessly?
                                MY MOTHER ARRIVE (mm) LATE (angry) WHY (furrow) → Why did my mother arrive late normally?
                                MY MOTHER ARRIVE (mm) LATE (sad) WHY (furrow) → Why did my mother arrive normally late?
                                MY MOTHER ARRIVE (mm) LATE WHY (furrow) → Why did my mother arrive late?
                                I HAPPY (happy) WHY (raise) MY SON HOMEWORK FINISH (cs) → I am happy because my son recenlty finished his homework.
                                I HAPPY (happy) WHY (raise) MY SON HOMEWORK FINISH (mm) → I am happy because my son normally finished his homework.
                                I HAPPY (happy) WHY (raise) MY SON HOMEWORK (cha) FINISH → I am happy because my son finished a lot of homework.
                                I HAPPY (happy) WHY (raise) MY SON HOMEWORK (mm) FINISH → I am happy because my son finished his homework normally.
                                I angry (angry) WHY (raise) MY SON HOMEWORK FINISH (th) → I am angry because my son sloppily finished his homework.
                                I angry (angry) WHY (raise) MY SON HOMEWORK (cha) FINISH NOT-YET(th) → I am angry because my son hasn't yet finished a lot of homework, and he's doing it sloppily.
                                I sad (sad) WHY (raise) MY SON HOMEWORK (cha) FINISH NOT-YET(th) → I am sad because my son hasn't yet finished a lot of homework, done in a sloppy manner.
                                I sad (sad) WHY (raise) MY SON HOMEWORK FINISH (th) → I am sad because my son sloppily finished his homework.
                                Task: Translate ASL gloss to fluent English capturing literal meaning and emotional/grammatical nuances. 
                                I want you to strictly stick to the sentences. Pls dont add any sign words that are not in the sentence. 
                                You can add extra words due to expressions, questions but pls not add any sign word not in the sentence.
                                The sentence you need to trsanslate is: ''' + prompt_text
                    answer = call_openai_chat(
                        prompt=prompt1,
                        model=args.model,
                        temperature=args.temperature,
                        timeout=args.timeout,
                        retries=args.retries,
                    )
                    success += 1
                except Exception as e:
                    answer = f"ERROR: {e}"
                
                # Find ground truth translation for the ASL gloss
                ground_truth_translation = find_ground_truth_translation(asl_gloss_for_matching, ground_truth_translations)
                
                # Calculate BLEU score if we have valid translations (skip if ground truth not found or error)
                if ground_truth_translation != "GROUND_TRUTH_NOT_FOUND" and not answer.startswith("ERROR:"):
                    try:
                        ref_processed = preprocess_text(ground_truth_translation)
                        hyp_processed = preprocess_text(answer)
                        if ref_processed and hyp_processed:
                            bleu_score = calculate_bleu_score_simple(ref_processed, hyp_processed)
                            bleu_scores.append(bleu_score)
                    except Exception as e:
                        # If BLEU calculation fails, skip this pair
                        pass
                
                # Write single output file: ground_truth_translation - LLM_translation
                fout.write(f"{ground_truth_translation} - {answer}\n")
                
                if total % 5 == 0:
                    print(f"Progress: {total} lines processed, {success} succeeded...")
            
            # Calculate and append average BLEU score at the end
            if bleu_scores:
                avg_bleu = sum(bleu_scores) / len(bleu_scores)
                fout.write("\n" + "=" * 80 + "\n")
                fout.write("BLEU SCORE METRICS\n")
                fout.write("=" * 80 + "\n")
                fout.write(f"Number of translation pairs with valid BLEU scores: {len(bleu_scores)}\n")
                fout.write(f"Average BLEU score: {avg_bleu:.4f}\n")
                fout.write(f"Average BLEU score (percentage): {avg_bleu * 100:.2f}%\n")
                fout.write("=" * 80 + "\n")
        
        print(f"Processed {total} lines from {args.input_file}. Successful responses: {success}.")
        if bleu_scores:
            avg_bleu = sum(bleu_scores) / len(bleu_scores)
            print(f"Average BLEU score: {avg_bleu:.4f} ({avg_bleu * 100:.2f}%)")
        print(f"Saved output to: {args.output}")
        return

    # Single prompt mode
    prompt = args.prompt
    if prompt is None:
        prompt = sys.stdin.read().strip()
        if not prompt:
            print("ERROR: No prompt provided via --prompt, --input_file, or stdin.")
            sys.exit(1)

    answer = call_openai_chat(
        prompt=prompt,
        model=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        retries=args.retries,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(answer + "\n")

    print(f"Saved answer to: {args.output}")


if __name__ == "__main__":
    main()


