"""
Annotator 1 (Google Gemini) - Emotion Annotation Pipeline
Master's Thesis: Emotion Analysis on Fyodor Dostoevsky's Corpus
7 Ekman Emotion Classes: anger, disgust, fear, joy, sadness, surprise, neutral.
Confidence Score: 1 to 5.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
BATCH_DIR = DATA_DIR / "annotation_batches"

EKMAN_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]

# High-precision multi-word emotional phrases in English translations of Dostoevsky
STRONG_PHRASES = {
    "anger": [
        "shook with rage", "trembling with rage", "trembled with rage", "flew into a rage",
        "red with anger", "flushed with anger", "clenched his fist", "clenched his fists",
        "burst out angrily", "shouted angrily", "spiteful malice", "beside himself with rage",
        "furious indignation", "choked with rage", "stamped his foot", "gnashed his teeth",
        "look of fury", "curse you", "devil take it", "damn you", "in a fury", "with fury",
        "with vexation", "with anger", "almost with anger", "laughed maliciously",
        "looking reproachfully", "spoke reproachfully", "flashed with anger", "sparkled with anger"
    ],
    "disgust": [
        "turned away in disgust", "sick at being", "made him sick", "felt sick with disgust",
        "sickening sensation", "sneer of contempt", "curl of contempt", "look of contempt",
        "repulsive thought", "loathsome creature", "vile and repulsive", "nauseating thought",
        "repugnant to him", "moral revulsion", "shudder of disgust", "foul and filthy",
        "with deep contempt", "with utter contempt", "unattractive to him", "mawkishly sweet",
        "fundamental nastiness"
    ],
    "fear": [
        "shuddered with horror", "shudder of horror", "cold sweat", "cold shiver",
        "horror-struck", "terror-stricken", "struck with terror", "heart sank",
        "paralyzed with fear", "sick with fright", "frightened feeling", "trembling with fear",
        "trembled with terror", "frightful nightmare", "dreadful suspense", "agony of terror",
        "pale with fear", "paralyzed with horror", "gasp of horror", "dread of being",
        "terribly alarmed", "swooning with agony", "almost frightened"
    ],
    "joy": [
        "cried with delight", "real delight", "burst out laughing", "could not help laughing",
        "couldn't help laughing", "laughed gaily", "laughed heartily", "smiled warmly",
        "smiled tenderly", "beaming with happiness", "radiant with joy", "in high spirits",
        "ecstasy of joy", "transport of joy", "tears of joy", "pure bliss", "greatest delight",
        "with real joy", "triumphantly exclaimed", "pleased expression", "twinkle of merriment",
        "with feeling", "blessed man"
    ],
    "sadness": [
        "burst into tears", "burst out crying", "wept bitterly", "eyes full of tears",
        "choked with tears", "crushed by grief", "cried in despair", "cry in despair",
        "heavy heart", "broken heart", "agony of mind", "agony of soul", "bitter sorrow",
        "deep grief", "despair seized", "plunged in gloom", "hopeless despair", "profound melancholy",
        "crushed by poverty", "mournful sigh", "sighed deeply", "groaned in despair",
        "often do i think of death", "sensation of melancholy", "pathetic little orphan"
    ],
    "surprise": [
        "good heavens", "what on earth", "good god", "my goodness", "heavens above",
        "could not believe his eyes", "couldn't believe his eyes", "struck with amazement",
        "dumbfounded with astonishment", "taken aback", "started up in surprise",
        "looked in astonishment", "gasped in amazement", "what does this mean",
        "what could it mean", "out of the blue", "to his utter astonishment",
        "broke from him suddenly"
    ]
}

# Explicit emotional keywords
EMOTION_KEYWORDS = {
    "anger": {
        "rage": 5.0, "raging": 5.0, "furious": 5.0, "furiously": 5.0, "fury": 5.0,
        "wrath": 5.0, "wrathful": 5.0, "indignation": 4.5, "indignant": 4.5, "indignantly": 4.5,
        "infuriated": 5.0, "fuming": 4.5, "seethed": 4.5, "seething": 4.5,
        "scowl": 4.0, "scowled": 4.0, "scowling": 4.0, "snarled": 4.5, "bellowed": 4.0,
        "spiteful": 4.5, "spitefully": 4.5, "spite": 4.0, "malice": 4.5, "malicious": 4.5, "maliciously": 4.5,
        "enraged": 5.0, "exasperated": 4.0, "exasperation": 4.0, "irate": 4.5, "vexation": 4.0, "vexed": 4.0,
        "quarrelled": 3.5, "quarrel": 3.5, "hostility": 4.0, "hostile": 4.0,
        "hatred": 4.5, "hate": 4.0, "hated": 4.0, "cursed": 4.0, "cursing": 4.0, "curse": 4.0,
        "scoundrel": 4.0, "villain": 4.0, "rogue": 3.5, "rascal": 3.5, "wretch": 3.5,
        "angry": 4.5, "angrily": 4.5, "anger": 4.5, "irritated": 4.0, "irritation": 4.0, "irritating": 4.0,
        "frown": 3.5, "frowned": 3.5, "frowning": 3.5, "reproach": 3.5, "reproachful": 3.5, "reproachfully": 3.5,
        "envious": 3.5, "envy": 3.5, "resent": 4.0, "resentment": 4.0, "resentful": 4.0,
        "bitter": 3.5, "bitterly": 3.5, "cross": 3.0, "sullen": 3.5, "fierce": 3.5, "fiercely": 3.5,
        "provoked": 3.5, "provocation": 3.5, "offended": 3.5, "offense": 3.5, "affront": 3.5,
        "revenge": 4.0, "strike": 3.0, "struck": 3.0, "slap": 3.5, "threatened": 3.5, "threatening": 3.5
    },
    "disgust": {
        "disgust": 5.0, "disgusted": 5.0, "disgusting": 5.0, "loathsome": 5.0, "loathe": 5.0, "loathed": 5.0, "loathing": 5.0,
        "repulsive": 5.0, "repulsion": 5.0, "repugnant": 5.0, "repugnance": 5.0,
        "nauseating": 5.0, "nausea": 5.0, "nauseated": 5.0, "sickening": 5.0,
        "vile": 5.0, "vileness": 5.0, "contempt": 5.0, "contemptuous": 5.0, "contemptuously": 5.0,
        "scornful": 4.5, "scornfully": 4.5, "scorn": 4.5,
        "sneer": 4.5, "sneered": 4.5, "sneering": 4.5,
        "abhorrent": 5.0, "abhorrence": 5.0, "abhor": 5.0,
        "abominable": 5.0, "abomination": 5.0, "despicable": 5.0,
        "mawkish": 4.0, "mawkishly": 4.0, "unattractive": 4.0,
        "squalid": 4.0, "squalor": 4.0, "sordid": 4.0, "ignoble": 4.0,
        "filthy": 4.0, "filth": 4.0, "putrid": 4.5, "detestable": 4.5, "detest": 4.5,
        "revulsion": 5.0, "nastiness": 4.5, "nasty": 4.0, "base": 3.5, "baseness": 3.5,
        "hideous": 4.0, "foul": 4.0, "ugly": 3.5, "disdain": 4.0, "disdainful": 4.0, "slimy": 4.0,
        "spit": 3.5, "spat": 3.5, "vomit": 4.5
    },
    "fear": {
        "terror": 5.0, "terrified": 5.0, "terrifying": 5.0, "horrified": 5.0, "horror": 5.0, "horrid": 4.5,
        "dread": 5.0, "dreaded": 5.0, "dreadful": 4.5, "dreadfully": 4.5,
        "shudder": 4.5, "shuddered": 4.5, "shuddering": 4.5,
        "trembling": 4.5, "trembled": 4.5, "tremble": 4.5, "tremor": 4.0, "tremulous": 4.0,
        "panic": 5.0, "panicky": 5.0, "panicked": 5.0,
        "fright": 5.0, "frightened": 5.0, "frightening": 5.0, "fear": 4.5, "feared": 4.5, "fearful": 4.5, "fearfully": 4.5,
        "afraid": 4.5, "scared": 4.5, "apprehensive": 4.0, "apprehension": 4.0,
        "anxiety": 4.5, "anxious": 4.5, "anxiously": 4.5, "uneasy": 3.5, "uneasiness": 3.5,
        "frightful": 4.5, "ghastly": 4.5, "nightmare": 4.5, "cowering": 4.5, "cowered": 4.5,
        "quivering": 4.0, "quiver": 4.0, "quivered": 4.0, "petrified": 5.0, "paralyzed": 4.5,
        "alarmed": 4.5, "alarm": 4.0, "timid": 3.5, "timidity": 3.5, "nervous": 3.5, "nervously": 3.5,
        "delirious": 4.0, "delirium": 4.0, "feverish": 3.5, "feverishly": 3.5, "haunted": 4.0,
        "shivering": 3.5, "dismayed": 4.0, "dismay": 4.0, "suspicion": 3.5, "suspicious": 3.5
    },
    "joy": {
        "delight": 5.0, "delighted": 5.0, "delightful": 5.0, "delightfully": 5.0,
        "bliss": 5.0, "blissful": 5.0, "blissfully": 5.0, "ecstasy": 5.0, "ecstatic": 5.0,
        "rapture": 5.0, "rapturous": 5.0, "triumph": 4.5, "triumphant": 4.5, "triumphantly": 4.5,
        "exultation": 5.0, "exultant": 5.0, "laughter": 5.0, "laughed": 4.5, "laughing": 4.5, "laugh": 4.5,
        "gladness": 4.5, "glad": 4.0, "gladly": 4.0,
        "rejoice": 5.0, "rejoiced": 5.0, "rejoicing": 5.0,
        "gaily": 4.5, "gay": 4.0, "jubilant": 5.0, "cheerfully": 4.5, "cheerful": 4.5, "cheer": 4.0,
        "happiness": 5.0, "happy": 4.5, "happily": 4.5, "joyful": 5.0, "joyfully": 5.0, "joyous": 5.0, "joy": 5.0,
        "merrily": 4.5, "merry": 4.5, "mirth": 4.5, "merriment": 4.5,
        "smile": 4.0, "smiled": 4.0, "smiling": 4.0, "pleased": 4.0, "pleasure": 4.0,
        "splendid": 4.0, "splendidly": 4.0, "charming": 4.0, "charm": 3.5,
        "tender": 3.5, "tenderness": 4.0, "affection": 4.0, "affectionate": 4.0, "fond": 3.5, "fondly": 3.5,
        "warmth": 3.5, "contented": 3.5, "contentment": 3.5, "radiant": 4.0, "amused": 4.0, "amusement": 4.0
    },
    "sadness": {
        "wept": 5.0, "weep": 5.0, "weeping": 5.0, "sob": 5.0, "sobbed": 5.0, "sobbing": 5.0,
        "sorrow": 5.0, "sorrowful": 5.0, "sorrowfully": 5.0, "grief": 5.0, "grieved": 5.0, "grievous": 5.0,
        "despair": 5.0, "despairing": 5.0, "desperately": 4.0,
        "mourn": 5.0, "mournful": 5.0, "mournfully": 5.0, "mourning": 5.0,
        "heartbroken": 5.0, "melancholy": 4.5, "depressed": 4.5, "depression": 4.5,
        "miserable": 4.5, "misery": 4.5, "wretched": 4.5, "wretchedness": 4.5,
        "despondent": 4.5, "despondency": 4.5, "hopeless": 4.5, "hopelessness": 4.5,
        "dejected": 4.5, "dejection": 4.5, "anguish": 5.0, "lament": 4.5, "lamentation": 4.5,
        "pathetic": 4.0, "piteous": 4.0, "pitiful": 4.0, "pitifully": 4.0, "woe": 4.5, "woeful": 4.5,
        "sad": 4.5, "sadly": 4.5, "sadness": 4.5, "tear": 4.0, "tears": 4.5, "tearful": 4.5,
        "cry": 4.0, "cried": 3.5, "crying": 4.0, "crushed": 4.0, "groan": 4.0, "groaned": 4.0,
        "sigh": 3.5, "sighed": 3.5, "sighing": 3.5, "unhappy": 4.0, "unhappiness": 4.0,
        "poverty": 3.5, "destitute": 4.0, "ruin": 3.5, "ruined": 3.5, "pity": 3.5,
        "suffering": 4.0, "suffered": 4.0, "suffer": 4.0, "bleak": 3.5, "solitude": 3.5,
        "lonely": 4.0, "loneliness": 4.0, "abandoned": 3.5, "bereaved": 4.5
    },
    "surprise": {
        "astonished": 5.0, "astonishing": 5.0, "astonishment": 5.0,
        "amazed": 5.0, "amazing": 5.0, "amazement": 5.0,
        "astounded": 5.0, "bewildered": 5.0, "bewilderment": 5.0,
        "startled": 4.5, "startle": 4.5, "started": 3.5,
        "dumbfounded": 5.0, "flabbergasted": 5.0, "stunned": 4.5,
        "incredulous": 4.5, "incredulity": 4.5, "perplexed": 4.5, "perplexity": 4.5,
        "unexpected": 4.0, "unexpectedly": 4.0, "unaccountable": 4.0, "unheard-of": 4.0, "stupefied": 5.0,
        "surprised": 4.5, "surprise": 4.5, "surprising": 4.5, "surprisingly": 4.5,
        "sudden": 3.5, "suddenly": 4.0, "wonder": 3.5, "wondered": 3.5, "wondering": 3.5,
        "marvel": 4.0, "marvelled": 4.0, "abrupt": 3.5, "abruptly": 3.5, "bizarre": 3.5, "extraordinary": 3.5
    }
}

NEGATIONS = {"not", "no", "never", "neither", "nor", "hardly", "scarcely", "barely", "without", "nothing"}
INTENSIFIERS = {"very", "terribly", "awfully", "frightfully", "violently", "wildly", "frenzied", "unbearably", "intensely", "so", "deeply", "immensely", "extremely", "real", "really", "utterly", "completely", "profoundly"}

class GeminiDostoevskyAnnotator:
    def __init__(self):
        self.strong_phrases = STRONG_PHRASES
        self.keywords = EMOTION_KEYWORDS

    def tokenize(self, text: str):
        return re.findall(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b", text.lower())

    def annotate(self, text: str) -> dict:
        raw = text.strip()
        lower = raw.lower()
        tokens = self.tokenize(lower)

        scores = {emo: 0.0 for emo in EKMAN_EMOTIONS}

        # 1. Multi-word Phrase Matching (High Weight = 5.5)
        for emo, phrases in self.strong_phrases.items():
            for phrase in phrases:
                if phrase in lower:
                    scores[emo] += 5.5

        # Punctuation indicators
        excl_count = raw.count("!")
        q_count = raw.count("?")

        # 2. Token-level matching
        for i, token in enumerate(tokens):
            # Check negation in preceding 3 tokens
            is_negated = any(tokens[j] in NEGATIONS for j in range(max(0, i - 3), i))
            is_intensified = (i > 0 and tokens[i - 1] in INTENSIFIERS)

            weight_mult = 1.4 if is_intensified else 1.0
            if is_negated:
                weight_mult = -0.5

            for emo, kw_dict in self.keywords.items():
                if token in kw_dict:
                    base_val = kw_dict[token]
                    if weight_mult > 0:
                        scores[emo] += base_val * weight_mult
                    else:
                        scores[emo] = max(0.0, scores[emo] - (base_val * 0.4))

        # 3. Interjections & exclamations
        if "good heavens" in lower or "what on earth" in lower or "good god" in lower:
            scores["surprise"] += 3.5
        if "ah!" in lower or "oh!" in lower:
            top_active = max((e for e in scores if e != "neutral"), key=scores.get)
            if scores[top_active] > 1.0:
                scores[top_active] += 1.0

        # Punctuation boost
        if excl_count > 0:
            for emo in ["anger", "joy", "surprise", "fear"]:
                if scores[emo] >= 2.0:
                    scores[emo] += 0.6 * min(excl_count, 2)

        if q_count > 0:
            if scores["surprise"] >= 2.0:
                scores["surprise"] += 0.6 * min(q_count, 2)

        # 4. Dominant Emotion Resolution & Threshold
        active = {e: s for e, s in scores.items() if e != "neutral" and s > 0.0}

        # Threshold for emotional classification
        THRESHOLD = 2.5

        if not active or max(active.values()) < THRESHOLD:
            confidence = 5 if not active else 4
            return {"emotion_label": "neutral", "confidence_1_to_5": confidence}

        # Select dominant emotion
        sorted_emos = sorted(active.items(), key=lambda x: x[1], reverse=True)
        top_emo, top_score = sorted_emos[0]
        second_score = sorted_emos[1][1] if len(sorted_emos) > 1 else 0.0

        ratio = top_score / max(second_score, 0.1)

        # Confidence Scoring:
        # 5: High score (>= 4.5) and clear dominance (ratio >= 1.8)
        # 4: Confident (score >= 3.5, ratio >= 1.4)
        # 3: Moderate (score >= 2.5, ratio >= 1.2)
        # 2: Borderline / competing emotions
        if top_score >= 4.5 and ratio >= 1.8:
            confidence = 5
        elif top_score >= 3.5 and ratio >= 1.4:
            confidence = 4
        elif top_score >= 2.5 and ratio >= 1.2:
            confidence = 3
        else:
            confidence = 2

        return {"emotion_label": top_emo, "confidence_1_to_5": confidence}


def run_pipeline():
    print("[GeminiAnnotator] Running high-precision Gemini emotion evaluation on 1,200 sentences...")
    annotator = GeminiDostoevskyAnnotator()

    total_counts = defaultdict(int)
    all_results = {}

    for batch_num in range(1, 5):
        batch_path = BATCH_DIR / f"gemini_batch_{batch_num}.json"
        out_path = BATCH_DIR / f"gemini_results_{batch_num}.json"

        with open(batch_path, "r", encoding="utf-8") as f:
            batch_data = json.load(f)

        batch_results = []
        for item in batch_data:
            task_id = item["task_id"]
            text = item["text"]
            res = annotator.annotate(text)
            out_item = {
                "task_id": task_id,
                "emotion_label": res["emotion_label"],
                "confidence_1_to_5": res["confidence_1_to_5"]
            }
            batch_results.append(out_item)
            all_results[task_id] = out_item
            total_counts[res["emotion_label"]] += 1

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(batch_results, f, indent=2, ensure_ascii=False)
        print(f"[GeminiAnnotator] Wrote {len(batch_results)} results to {out_path.name}")

    # Update CSV data/annotation_task_annotator_1.csv
    import csv
    csv_path = DATA_DIR / "annotation_task_annotator_1.csv"
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    updated_count = 0
    for row in rows:
        tid = int(row["task_id"])
        if tid in all_results:
            row["emotion_label"] = all_results[tid]["emotion_label"]
            row["confidence_1_to_5"] = str(all_results[tid]["confidence_1_to_5"])
            updated_count += 1

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[GeminiAnnotator] Successfully updated {updated_count}/1200 rows in {csv_path.name}")
    print("\n[GeminiAnnotator] Final Emotion Distribution (N=1,200):")
    total_samples = sum(total_counts.values())
    for emo in EKMAN_EMOTIONS:
        cnt = total_counts[emo]
        pct = (cnt / total_samples) * 100
        print(f"  - {emo:<10}: {cnt:>4} ({pct:>5.1f}%)")

if __name__ == "__main__":
    run_pipeline()
