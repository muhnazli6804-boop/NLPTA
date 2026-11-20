"""
Method Comparison Module: Lexicon (NRC/VADER) vs. Transformer (RoBERTa/BERT)
Project: "Analisis Sentimen dan Perbandingan Distribusi Emosi pada Karya Fiksi Fyodor Dostoevsky"

Tasks:
1. Spearman Rank Correlation (sentence-level, title-level, narrative arc-level) on 6 intersecting Ekman emotions.
2. Jensen-Shannon Divergence (JSD) distance matrices between novels (inter-novel emotional divergence) and cross-methods.
3. Inter-annotator agreement (Cohen's Kappa) on 400 overlap sentences, plus method-to-method consensus agreement.
4. Quantitative and qualitative benchmark evaluation against Picca, Schnyder & Romele (2024) on 'Crime and Punishment'.
"""

import sys
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import cohen_kappa_score, classification_report, accuracy_score, f1_score

# Ensure root workspace is in sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.config import RESULTS_DIR, DATA_DIR, SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Intersecting 6 Ekman emotions between NRC (8 emotions) and Transformer (7 Ekman + Neutral)
INTERSECTING_EMOTIONS = ["anger", "disgust", "fear", "joy", "sadness", "surprise"]
TRANSFORMER_7 = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
NRC_8 = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]


def compute_spearman_correlations(lex_df: pd.DataFrame, trans_df: pd.DataFrame, 
                                  lex_arc_df: pd.DataFrame, trans_arc_df: pd.DataFrame):
    """
    Computes Spearman rank correlation between NRC Lexicon and Transformer (RoBERTa)
    across:
    1. Sentence-level (66,410 sentences)
    2. Narrative arc progression (100 bins per novel across 12 novels)
    3. Title-level overall emotion percentages
    """
    logger.info("Computing Spearman rank correlations on 6 intersecting emotions...")
    results = {
        "sentence_level": {},
        "narrative_arc_per_title": {},
        "overall_narrative_arc": {},
        "title_level": {}
    }

    # 1. Sentence-Level Correlations
    # Merge on sentence_id
    lex_sent_cols = ["sentence_id", "title"] + [f"nrc_norm_{e}" for e in INTERSECTING_EMOTIONS]
    trans_sent_cols = ["sentence_id"] + [f"roberta_prob_{e}" for e in INTERSECTING_EMOTIONS]

    merged_sent = pd.merge(
        lex_df[lex_sent_cols],
        trans_df[trans_sent_cols],
        on="sentence_id",
        how="inner"
    )

    sent_records = []
    for emo in INTERSECTING_EMOTIONS:
        lex_col = f"nrc_norm_{emo}"
        trans_col = f"roberta_prob_{emo}"
        corr, pval = spearmanr(merged_sent[lex_col], merged_sent[trans_col])
        results["sentence_level"][emo] = {
            "spearman_rho": round(float(corr), 4),
            "p_value": float(pval),
            "statistically_significant": bool(pval < 0.05),
            "n_sentences": len(merged_sent)
        }
        sent_records.append({
            "emotion": emo,
            "spearman_rho": round(float(corr), 4),
            "p_value": float(pval),
            "significant_p05": pval < 0.05
        })

    sent_corr_df = pd.DataFrame(sent_records)
    sent_corr_df.to_csv(RESULTS_DIR / "spearman_correlation_sentences.csv", index=False)
    logger.info("Saved sentence-level Spearman correlations to results/spearman_correlation_sentences.csv")

    # 2. Narrative Arc Correlations (Per Title and Overall)
    # Merge narrative arc bins
    lex_arc_cols = ["title", "narrative_bin", "narrative_progress_pct"] + [f"nrc_prop_{e}" for e in INTERSECTING_EMOTIONS]
    trans_arc_cols = ["title", "narrative_bin"] + [f"roberta_prob_{e}" for e in INTERSECTING_EMOTIONS]

    merged_arc = pd.merge(
        lex_arc_df[lex_arc_cols],
        trans_arc_df[trans_arc_cols],
        on=["title", "narrative_bin"],
        how="inner"
    )

    arc_records = []
    titles = merged_arc["title"].unique()

    for title in titles:
        sub = merged_arc[merged_arc["title"] == title]
        results["narrative_arc_per_title"][title] = {}
        for emo in INTERSECTING_EMOTIONS:
            lex_col = f"nrc_prop_{emo}"
            trans_col = f"roberta_prob_{emo}"
            corr, pval = spearmanr(sub[lex_col], sub[trans_col])
            results["narrative_arc_per_title"][title][emo] = {
                "spearman_rho": round(float(corr), 4),
                "p_value": float(pval),
                "significant": bool(pval < 0.05)
            }
            arc_records.append({
                "title": title,
                "emotion": emo,
                "spearman_rho": round(float(corr), 4),
                "p_value": float(pval),
                "significant_p05": pval < 0.05
            })

    # Overall arc correlation (across all 1200 bins)
    for emo in INTERSECTING_EMOTIONS:
        corr, pval = spearmanr(merged_arc[f"nrc_prop_{emo}"], merged_arc[f"roberta_prob_{emo}"])
        results["overall_narrative_arc"][emo] = {
            "spearman_rho": round(float(corr), 4),
            "p_value": float(pval),
            "significant": bool(pval < 0.05)
        }

    arc_corr_df = pd.DataFrame(arc_records)
    arc_corr_df.to_csv(RESULTS_DIR / "spearman_correlation_narrative_arc.csv", index=False)
    logger.info("Saved narrative-arc Spearman correlations to results/spearman_correlation_narrative_arc.csv")

    return results


def compute_jsd_matrices(lex_title_df: pd.DataFrame, trans_title_df: pd.DataFrame):
    """
    Computes Jensen-Shannon Divergence (JSD) distance matrices:
    1. Inter-novel JSD for RoBERTa (7 classes and 6 intersecting classes)
    2. Inter-novel JSD for NRC Lexicon (8 classes and 6 intersecting classes)
    3. Cross-method JSD for each title: JSD(NRC_title_i, RoBERTa_title_i)
    """
    logger.info("Computing Jensen-Shannon Divergence (JSD) matrices...")
    titles = sorted(trans_title_df["title"].unique())
    n = len(titles)

    # Prepare normalized probability distributions over 6 intersecting emotions
    # 1. RoBERTa intersecting 6
    roberta_probs = {}
    for _, row in trans_title_df.iterrows():
        t = row["title"]
        vec = np.array([row[f"roberta_mean_prob_{e}"] for e in INTERSECTING_EMOTIONS], dtype=np.float64)
        vec_norm = vec / vec.sum() if vec.sum() > 0 else np.ones(len(INTERSECTING_EMOTIONS)) / len(INTERSECTING_EMOTIONS)
        roberta_probs[t] = vec_norm

    # 2. NRC intersecting 6
    nrc_probs = {}
    for _, row in lex_title_df.iterrows():
        t = row["title"]
        vec = np.array([row[f"nrc_prop_{e}"] for e in INTERSECTING_EMOTIONS], dtype=np.float64)
        vec_norm = vec / vec.sum() if vec.sum() > 0 else np.ones(len(INTERSECTING_EMOTIONS)) / len(INTERSECTING_EMOTIONS)
        nrc_probs[t] = vec_norm

    # 1. Pairwise Inter-Novel JSD Matrix for RoBERTa
    jsd_roberta_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                jsd_roberta_mat[i, j] = 0.0
            else:
                dist = jensenshannon(roberta_probs[titles[i]], roberta_probs[titles[j]], base=2)
                jsd_roberta_mat[i, j] = round(float(dist ** 2), 5)

    df_jsd_roberta = pd.DataFrame(jsd_roberta_mat, index=titles, columns=titles)
    df_jsd_roberta.to_csv(RESULTS_DIR / "jsd_matrix_roberta.csv")

    # 2. Pairwise Inter-Novel JSD Matrix for NRC
    jsd_nrc_mat = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                jsd_nrc_mat[i, j] = 0.0
            else:
                dist = jensenshannon(nrc_probs[titles[i]], nrc_probs[titles[j]], base=2)
                jsd_nrc_mat[i, j] = round(float(dist ** 2), 5)

    df_jsd_nrc = pd.DataFrame(jsd_nrc_mat, index=titles, columns=titles)
    df_jsd_nrc.to_csv(RESULTS_DIR / "jsd_matrix_nrc.csv")

    # 3. Cross-Method JSD (NRC vs RoBERTa for the same novel)
    cross_jsd = []
    for t in titles:
        dist = jensenshannon(nrc_probs[t], roberta_probs[t], base=2)
        jsd_val = float(dist ** 2)
        cross_jsd.append({
            "title": t,
            "jsd_cross_method": round(jsd_val, 5),
            "jsd_distance": round(float(dist), 5)
        })

    df_cross_jsd = pd.DataFrame(cross_jsd).sort_values("jsd_cross_method", ascending=False)
    df_cross_jsd.to_csv(RESULTS_DIR / "jsd_cross_methods.csv", index=False)

    logger.info("Saved JSD matrices to results/jsd_matrix_roberta.csv, results/jsd_matrix_nrc.csv, results/jsd_cross_methods.csv")
    return {
        "jsd_matrix_roberta": df_jsd_roberta.to_dict(),
        "jsd_matrix_nrc": df_jsd_nrc.to_dict(),
        "cross_methods": df_cross_jsd.to_dict(orient="records")
    }


def evaluate_annotation_agreement(roberta_sent_df: pd.DataFrame, lex_sent_df: pd.DataFrame):
    """
    Evaluates:
    1. Human Inter-Annotator Agreement (Cohen's Kappa) on the 400 overlap sentences.
    2. Model-to-Model and Model-to-Lexicon Agreement on the 2,000 sampled sentences.
    """
    logger.info("Evaluating annotation tasks and inter-method agreement...")
    annot_master_path = DATA_DIR / "annotation_master_2000.csv"
    task1_path = DATA_DIR / "annotation_task_annotator_1.csv"
    task2_path = DATA_DIR / "annotation_task_annotator_2.csv"

    out = {
        "human_annotation_status": "in_progress",
        "cohens_kappa_annotator_1_vs_2": None,
        "n_overlap_sentences": 400,
        "model_lexicon_agreement_2000": {}
    }

    if not annot_master_path.exists():
        logger.warning(f"Master annotation sample not found at {annot_master_path}")
        return out

    master_df = pd.read_csv(annot_master_path)
    t1_df = pd.read_csv(task1_path) if task1_path.exists() else None
    t2_df = pd.read_csv(task2_path) if task2_path.exists() else None

    # Check if annotators have entered labels
    has_t1_labels = t1_df is not None and "emotion_label" in t1_df.columns and t1_df["emotion_label"].notna().sum() > 50
    has_t2_labels = t2_df is not None and "emotion_label" in t2_df.columns and t2_df["emotion_label"].notna().sum() > 50

    if has_t1_labels and has_t2_labels:
        # Check overlap on the 400 sentences via sample_id
        overlap_ids = set(t1_df["sample_id"]).intersection(set(t2_df["sample_id"]))
        t1_sub = t1_df[t1_df["sample_id"].isin(overlap_ids)].sort_values("sample_id").reset_index(drop=True)
        t2_sub = t2_df[t2_df["sample_id"].isin(overlap_ids)].sort_values("sample_id").reset_index(drop=True)
        
        valid = (t1_sub["emotion_label"].notna()) & (t2_sub["emotion_label"].notna())
        if valid.sum() > 0:
            kappa = cohen_kappa_score(t1_sub.loc[valid, "emotion_label"], t2_sub.loc[valid, "emotion_label"])
            acc = accuracy_score(t1_sub.loc[valid, "emotion_label"], t2_sub.loc[valid, "emotion_label"])
            out["human_annotation_status"] = "completed_via_llm"
            out["annotator_1_model"] = "Google Gemini (gemini-3.8-flash-high)"
            out["annotator_2_model"] = "Anthropic Claude (claude-sonnet-4-6)"
            out["cohens_kappa_annotator_1_vs_2"] = round(float(kappa), 4)
            out["cohens_kappa_gemini_vs_claude"] = round(float(kappa), 4)
            out["raw_agreement_gemini_vs_claude"] = round(float(acc), 4)
            out["n_evaluated_overlap"] = int(valid.sum())
            logger.info(f"Cohen's Kappa (Gemini vs Claude on {valid.sum()} overlap): Kappa={kappa:.4f}, Acc={acc:.4f}")

        # Build Gold Consensus for 2,000 sentences
        t1_indexed = t1_df.set_index("sample_id")
        t2_indexed = t2_df.set_index("sample_id")
        gold_labels = []
        gold_sources = []

        for _, row in master_df.iterrows():
            sid = row["sample_id"]
            assign = row.get("assignment", "overlap")
            if assign == "overlap":
                l1 = t1_indexed.loc[sid, "emotion_label"] if sid in t1_indexed.index else None
                c1 = t1_indexed.loc[sid, "confidence_1_to_5"] if sid in t1_indexed.index else 3
                l2 = t2_indexed.loc[sid, "emotion_label"] if sid in t2_indexed.index else None
                c2 = t2_indexed.loc[sid, "confidence_1_to_5"] if sid in t2_indexed.index else 3
                if l1 == l2 and l1 is not None:
                    gold_labels.append(l1)
                    gold_sources.append("agreed_both")
                else:
                    if c1 > c2 and l1 is not None:
                        gold_labels.append(l1)
                        gold_sources.append("gemini_higher_conf")
                    elif c2 > c1 and l2 is not None:
                        gold_labels.append(l2)
                        gold_sources.append("claude_higher_conf")
                    else:
                        if l1 == "neutral" and l2 is not None and l2 != "neutral":
                            gold_labels.append(l2)
                            gold_sources.append("claude_nuanced")
                        elif l2 == "neutral" and l1 is not None and l1 != "neutral":
                            gold_labels.append(l1)
                            gold_sources.append("gemini_nuanced")
                        else:
                            gold_labels.append(l2 or l1)
                            gold_sources.append("claude_default")
            elif assign == "annotator_1_only":
                l1 = t1_indexed.loc[sid, "emotion_label"] if sid in t1_indexed.index else None
                gold_labels.append(l1 or "neutral")
                gold_sources.append("gemini_solo")
            else:
                l2 = t2_indexed.loc[sid, "emotion_label"] if sid in t2_indexed.index else None
                gold_labels.append(l2 or "neutral")
                gold_sources.append("claude_solo")

        master_df["gold_emotion_label"] = gold_labels
        master_df["consensus_source"] = gold_sources
        gold_csv_path = DATA_DIR / "annotation_gold_consensus_2000.csv"
        master_df.to_csv(gold_csv_path, index=False)
        logger.info(f"Saved Gold Consensus 2,000 dataset to {gold_csv_path}")

        # Evaluate RoBERTa vs Gold Consensus on 2,000 sentences
        merged_gold_rob = pd.merge(master_df, roberta_sent_df[["sentence_id", "roberta_pred_emotion"]], on="sentence_id")
        out["roberta_vs_gold_consensus_2000"] = {
            "accuracy": round(float(accuracy_score(merged_gold_rob["gold_emotion_label"], merged_gold_rob["roberta_pred_emotion"])), 4),
            "macro_f1": round(float(f1_score(merged_gold_rob["gold_emotion_label"], merged_gold_rob["roberta_pred_emotion"], average="macro")), 4),
            "cohens_kappa": round(float(cohen_kappa_score(merged_gold_rob["gold_emotion_label"], merged_gold_rob["roberta_pred_emotion"])), 4),
            "n_sentences": len(merged_gold_rob)
        }

        # Evaluate BERT vs Gold Consensus on 2,000 sentences
        bert_sample_path = RESULTS_DIR / "transformer_scores_sample_2000_bert.csv"
        if bert_sample_path.exists():
            sub_bert = pd.read_csv(bert_sample_path)
            merged_gold_bert = pd.merge(master_df, sub_bert[["sample_id", "bert_pred_emotion"]], on="sample_id")
            out["bert_vs_gold_consensus_2000"] = {
                "accuracy": round(float(accuracy_score(merged_gold_bert["gold_emotion_label"], merged_gold_bert["bert_pred_emotion"])), 4),
                "macro_f1": round(float(f1_score(merged_gold_bert["gold_emotion_label"], merged_gold_bert["bert_pred_emotion"], average="macro")), 4),
                "cohens_kappa": round(float(cohen_kappa_score(merged_gold_bert["gold_emotion_label"], merged_gold_bert["bert_pred_emotion"])), 4),
                "n_sentences": len(merged_gold_bert)
            }
    else:
        out["human_annotation_status"] = "pending_manual_annotation"
        out["note"] = (
            "Templates for Annotator 1 and Annotator 2 are generated with 400 overlap sentences. "
            "Automated pipeline will calculate Cohen's Kappa immediately when annotated CSVs are provided."
        )

    # Compare RoBERTa vs NRC on the 2,000 sentences
    sample_ids = master_df["sentence_id"].unique()
    sub_roberta = roberta_sent_df[roberta_sent_df["sentence_id"].isin(sample_ids)].sort_values("sentence_id")
    sub_lex = lex_sent_df[lex_sent_df["sentence_id"].isin(sample_ids)].sort_values("sentence_id")

    # Determine dominant NRC emotion for the 6 intersecting classes
    dominant_nrc = []
    for _, row in sub_lex.iterrows():
        scores = {e: row[f"nrc_norm_{e}"] for e in INTERSECTING_EMOTIONS}
        max_e = max(scores, key=scores.get)
        dominant_nrc.append(max_e if scores[max_e] > 0 else "neutral")

    sub_roberta = sub_roberta.copy()
    sub_roberta["dominant_nrc"] = dominant_nrc

    # Compute agreement
    pred_roberta = sub_roberta["roberta_pred_emotion"]
    pred_nrc = sub_roberta["dominant_nrc"]

    # Filter to intersecting where neither is neutral
    mask_intersect = (pred_roberta.isin(INTERSECTING_EMOTIONS)) & (pred_nrc.isin(INTERSECTING_EMOTIONS))
    if mask_intersect.sum() > 0:
        kappa_model_lex = cohen_kappa_score(pred_roberta[mask_intersect], pred_nrc[mask_intersect])
        acc_model_lex = accuracy_score(pred_roberta[mask_intersect], pred_nrc[mask_intersect])
        out["model_lexicon_agreement_2000"] = {
            "n_intersecting_sentences": int(mask_intersect.sum()),
            "cohens_kappa_roberta_vs_nrc": round(float(kappa_model_lex), 4),
            "raw_agreement_accuracy": round(float(acc_model_lex), 4)
        }
        logger.info(f"RoBERTa vs NRC Agreement on 2,000 sample (non-neutral): Kappa={kappa_model_lex:.4f}, Acc={acc_model_lex:.4f}")

    # Also evaluate BERT vs RoBERTa agreement if BERT sample predictions exist
    bert_sample_path = RESULTS_DIR / "transformer_scores_sample_2000_bert.csv"
    if bert_sample_path.exists():
        sub_bert = pd.read_csv(bert_sample_path).sort_values("sentence_id")
        pred_bert = sub_bert["bert_pred_emotion"]
        kappa_bert_roberta = cohen_kappa_score(pred_roberta, pred_bert)
        acc_bert_roberta = accuracy_score(pred_roberta, pred_bert)
        out["bert_vs_roberta_agreement_2000"] = {
            "cohens_kappa": round(float(kappa_bert_roberta), 4),
            "raw_agreement_accuracy": round(float(acc_bert_roberta), 4),
            "n_sentences": len(pred_bert)
        }
        logger.info(f"BERT vs RoBERTa Agreement on 2,000 sample: Kappa={kappa_bert_roberta:.4f}, Acc={acc_bert_roberta:.4f}")

    return out


def compare_picca_crime_and_punishment(trans_arc_df: pd.DataFrame, lex_arc_df: pd.DataFrame):
    """
    Quantitative & qualitative benchmark evaluation against Picca, Schnyder & Romele (2024)
    for 'Crime and Punishment'.

    Key narrative crisis checkpoints identified in Picca et al. (2024):
    1. Checkpoint 1 (15-20%): Murder of Alyona Ivanovna & Lizaveta (Climax Part I)
       -> Expected: Sharp spike in Fear & Anger, dramatic trough in Joy.
    2. Checkpoint 2 (30-35%): Marmeladov's tragedy & funeral (Part II)
       -> Expected: Prominent peak in Sadness.
    3. Checkpoint 3 (60-75%): Psychological cat-and-mouse duel with Porfiry Petrovich (Part IV-V)
       -> Expected: Sustained high Suspense/Surprise, high Fear, fluctuating Anger.
    4. Checkpoint 4 (85-90%): Svidrigailov's nightmare/suicide & Raskolnikov's confession (Part VI)
       -> Expected: Dramatic climax of distress (Fear, Anger, Sadness).
    5. Checkpoint 5 (95-100%): Siberian exile & spiritual rebirth with Sonia (Epilogue)
       -> Expected: Sudden surge in Joy, collapse in Fear and Sadness.
    """
    logger.info("Evaluating Crime and Punishment narrative progression against Picca et al. (2024)...")
    cp_trans = trans_arc_df[trans_arc_df["title"] == "Crime and Punishment"].sort_values("narrative_progress_pct").reset_index(drop=True)
    cp_lex = lex_arc_df[lex_arc_df["title"] == "Crime and Punishment"].sort_values("narrative_progress_pct").reset_index(drop=True)

    checkpoints = [
        {
            "name": "Murder of Alyona Ivanovna (Part I Climax)",
            "bin_range": (15, 20),
            "expected_dynamics": "High Fear & Anger, Low Joy",
        },
        {
            "name": "Marmeladov Death & Sonia Introduction (Part II)",
            "bin_range": (30, 35),
            "expected_dynamics": "Peak Sadness, Elevated Fear",
        },
        {
            "name": "Psychological Interrogations by Porfiry Petrovich (Part IV-V)",
            "bin_range": (60, 75),
            "expected_dynamics": "Peak Surprise / Suspense, Sustained Fear",
        },
        {
            "name": "Svidrigailov Suicide & Confession (Part VI)",
            "bin_range": (85, 90),
            "expected_dynamics": "Climactic Distress, High Fear & Anger",
        },
        {
            "name": "Siberian Prison & Epilogue Redemption (Epilogue)",
            "bin_range": (95, 100),
            "expected_dynamics": "Sharp Joy Recovery, Dramatic Drop in Fear/Sadness",
        }
    ]

    evaluated_points = []
    for cp in checkpoints:
        low, high = cp["bin_range"]
        sub_t = cp_trans[(cp_trans["narrative_progress_pct"] >= low) & (cp_trans["narrative_progress_pct"] <= high)]
        sub_l = cp_lex[(cp_lex["narrative_progress_pct"] >= low) & (cp_lex["narrative_progress_pct"] <= high)]

        roberta_metrics = {e: round(float(sub_t[f"roberta_prob_{e}"].mean() * 100), 2) for e in INTERSECTING_EMOTIONS}
        nrc_metrics = {e: round(float(sub_l[f"nrc_prop_{e}"].mean() * 100), 2) for e in INTERSECTING_EMOTIONS}

        evaluated_points.append({
            "narrative_checkpoint": cp["name"],
            "novel_progress_pct": f"{low}%-{high}%",
            "picca_expected_pattern": cp["expected_dynamics"],
            "empirical_roberta_probs_pct": roberta_metrics,
            "empirical_nrc_pct": nrc_metrics,
            "dominant_roberta_emotion": max(roberta_metrics, key=roberta_metrics.get),
            "dominant_nrc_emotion": max(nrc_metrics, key=nrc_metrics.get)
        })

    picca_summary = {
        "reference_study": "Picca, Schnyder & Romele (2024) - Computational Literary Studies on Crime and Punishment",
        "novel": "Crime and Punishment",
        "checkpoints_evaluated": evaluated_points,
        "epilogue_joy_surge_detected": bool(
            cp_trans[cp_trans["narrative_progress_pct"] >= 95]["roberta_prob_joy"].mean() > 
            cp_trans[(cp_trans["narrative_progress_pct"] >= 80) & (cp_trans["narrative_progress_pct"] <= 90)]["roberta_prob_joy"].mean()
        ),
        "murder_fear_spike_detected": bool(
            cp_trans[(cp_trans["narrative_progress_pct"] >= 15) & (cp_trans["narrative_progress_pct"] <= 20)]["roberta_prob_fear"].mean() >
            cp_trans[cp_trans["narrative_progress_pct"] <= 10]["roberta_prob_fear"].mean()
        )
    }

    out_picca = RESULTS_DIR / "picca_comparison_crime_and_punishment.json"
    with open(out_picca, "w", encoding="utf-8") as f:
        json.dump(picca_summary, f, indent=2)
    logger.info(f"Saved Picca et al. benchmark comparison to {out_picca}")

    return picca_summary


def run_comparative_analysis():
    """
    Main orchestration routine for Tahap 5: Comparative Analysis.
    """
    logger.info("=== STARTING TAHAP 5: COMPARATIVE ANALYSIS ===")

    # 1. Load Lexicon datasets
    lex_sent_file = RESULTS_DIR / "lexicon_scores_sentences.csv"
    lex_title_file = RESULTS_DIR / "lexicon_scores_by_title.csv"
    lex_arc_file = RESULTS_DIR / "lexicon_scores_narrative_arc.csv"

    # 2. Load Transformer datasets
    trans_sent_file = RESULTS_DIR / "transformer_scores_sentences.csv"
    trans_title_file = RESULTS_DIR / "transformer_scores_by_title.csv"
    trans_arc_file = RESULTS_DIR / "transformer_scores_narrative_arc.csv"

    if not trans_sent_file.exists():
        # Fallback to roberta specific file if present
        trans_sent_file = RESULTS_DIR / "transformer_scores_sentences_roberta.csv"
    if not trans_title_file.exists():
        trans_title_file = RESULTS_DIR / "transformer_scores_by_title_roberta.csv"
    if not trans_arc_file.exists():
        trans_arc_file = RESULTS_DIR / "transformer_scores_narrative_arc_roberta.csv"

    logger.info(f"Reading Lexicon scores from: {lex_sent_file}")
    lex_sent = pd.read_csv(lex_sent_file)
    lex_title = pd.read_csv(lex_title_file)
    lex_arc = pd.read_csv(lex_arc_file)

    logger.info(f"Reading Transformer scores from: {trans_sent_file}")
    trans_sent = pd.read_csv(trans_sent_file)
    trans_title = pd.read_csv(trans_title_file)
    trans_arc = pd.read_csv(trans_arc_file)

    # 1. Spearman Correlations
    spearman_res = compute_spearman_correlations(lex_sent, trans_sent, lex_arc, trans_arc)

    # 2. Jensen-Shannon Divergence
    jsd_res = compute_jsd_matrices(lex_title, trans_title)

    # 3. Inter-Annotator Agreement & Method Agreement
    annot_res = evaluate_annotation_agreement(trans_sent, lex_sent)

    # 4. Picca et al. Benchmark Comparison
    picca_res = compare_picca_crime_and_punishment(trans_arc, lex_arc)

    # Master Comparative JSON Report
    master_report = {
        "spearman_correlations": spearman_res,
        "jensen_shannon_divergence": {
            "cross_method_rankings": jsd_res["cross_methods"]
        },
        "annotation_and_agreement": annot_res,
        "picca_benchmark_validation": picca_res
    }

    master_report_path = RESULTS_DIR / "comparative_analysis_metrics.json"
    with open(master_report_path, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2)

    logger.info(f"=== Comparative analysis successfully complete! Saved master metrics to {master_report_path} ===")
    return master_report


if __name__ == "__main__":
    run_comparative_analysis()
