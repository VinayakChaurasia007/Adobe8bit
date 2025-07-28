import argparse
import os
import glob
import time
import json
import re
import numpy as np
import heapq
from datetime import datetime

import spacy

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import pypdf

CONFIG_FILE = 'input/config.json'
CONFIG = {}
try:
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        CONFIG = json.load(f)
    print(f"Configuration loaded from {CONFIG_FILE}")
except FileNotFoundError:
    print(f"Error: {CONFIG_FILE} not found. Please create it in the same directory as run.py.")
    exit(1)
except json.JSONDecodeError as e:
    print(f"Error decoding {CONFIG_FILE}: {e}. Please check JSON syntax.")
    exit(1)

MODEL_PATH = CONFIG.get('MODEL_PATH', './model_weights/bge-base-en-v1.5')
SPACY_MODEL_NAME = CONFIG.get('SPACY_MODEL_NAME', 'en_core_web_sm')

MIN_SECTION_LENGTH = CONFIG.get('MIN_SECTION_LENGTH', 50)
MAX_SECTION_LENGTH = CONFIG.get('MAX_SECTION_LENGTH', 4000)
MAX_SENTENCES_FOR_REFINED_TEXT = CONFIG.get('MAX_SENTENCES_FOR_REFINED_TEXT', 3)
TOP_N_SECTIONS_FOR_ANALYSIS = CONFIG.get('TOP_N_SECTIONS_FOR_ANALYSIS', 5)

QUERY_PREFIX = CONFIG.get('QUERY_PREFIX', "Represent this sentence for searching relevant passages:")
DOCUMENT_PREFIX = CONFIG.get('DOCUMENT_PREFIX', "Represent this passage for searching relevant passages:")

GENERAL_NEGATION_PATTERNS = CONFIG.get('GENERAL_NEGATION_PATTERNS', [])

CATEGORY_ASPECTS = CONFIG.get('CATEGORY_ASPECTS', {})

MAIN_QUERY_WEIGHT = CONFIG.get('MAIN_QUERY_WEIGHT', 1.0)
POSITIVE_ASPECT_WEIGHT = CONFIG.get('POSITIVE_ASPECT_WEIGHT', 0.5)
NEGATIVE_ASPECT_WEIGHT = CONFIG.get('NEGATIVE_ASPECT_WEIGHT', 0.7)
ASPECT_SIMILARITY_THRESHOLD = CONFIG.get('ASPECT_SIMILARITY_THRESHOLD', 0.2)

sentence_model = None
try:
    print(f"Loading Sentence Transformer model from {MODEL_PATH}...")
    sentence_model = SentenceTransformer(MODEL_PATH, device='cpu')
    print("Sentence Transformer model loaded successfully (on CPU).")
except Exception as e:
    print(f"Error loading Sentence Transformer model from {MODEL_PATH}: {e}")
    print("Proceeding without semantic embeddings (using keyword fallback).")

spacy_nlp_model = None
try:
    print(f"Loading spaCy model '{SPACY_MODEL_NAME}'...")
    spacy_nlp_model = spacy.load(SPACY_MODEL_NAME)
    print(f"spaCy model '{SPACY_MODEL_NAME}' loaded successfully.")
except OSError:
    print(f"spaCy model '{SPACY_MODEL_NAME}' not found. Please run 'python -m spacy download {SPACY_MODEL_NAME}' to download it.")
    print("Proceeding without spaCy for keyword extraction and sentence tokenization; will use regex fallback.")
except Exception as e:
    print(f"Error loading spaCy model '{SPACY_MODEL_NAME}': {e}")
    print("Proceeding without spaCy for keyword extraction and sentence tokenization; will use regex fallback.")

def preprocess_documents(document_paths):
    processed_docs = []

    for doc_path in document_paths:
        try:
            reader = pypdf.PdfReader(doc_path)
            doc_name = os.path.basename(doc_path)

            for page_num in range(len(reader.pages)):
                page_text = reader.pages[page_num].extract_text()
                if not page_text:
                    continue

                sections = re.split(r'\n{2,}', page_text)

                current_section_idx = 0
                for section_text_raw in sections:
                    section_text = section_text_raw.strip()
                    if not section_text or len(section_text) < MIN_SECTION_LENGTH:
                        continue

                    section_title_candidate = f"Page {page_num+1} - Section {current_section_idx + 1}"
                    first_line = section_text.split('\n')[0].strip()

                    if 5 < len(first_line) < 100 and \
                       (first_line.isupper() or re.match(r'^\s*(\d+(\.\d+)*)?\s*[A-Z][a-zA-Z0-9\s,&/-]*$', first_line)):
                        section_title_candidate = first_line

                    processed_docs.append({
                        "document": doc_name,
                        "page_number": page_num + 1,
                        "text_content": section_text,
                        "section_title_candidate": section_title_candidate
                    })
                    current_section_idx += 1

        except Exception as e:
            print(f"Error processing document {doc_path}: {e}")
            continue
    return processed_docs

def get_query_vectors(persona_description, job_to_be_done):
    combined_main_query_text = f"{persona_description}. {job_to_be_done}"
    main_query_embedding = None
    positive_aspect_embedding = None
    negative_aspect_embedding = None

    combined_query_lower = combined_main_query_text.lower()

    positive_aspect_concept_text = None
    negative_aspect_concept_text = None

    for pattern in GENERAL_NEGATION_PATTERNS:
        match = re.search(pattern, combined_query_lower)
        if match:
            negative_aspect_concept_text = match.group(1).strip()
            break

    for category_name, aspects in CATEGORY_ASPECTS.items():
        is_triggered = False
        for trigger in aspects["query_triggers"]:
            if trigger in combined_query_lower:
                is_triggered = True
                break

        if is_triggered:
            if aspects["positive_semantic_space"]:
                positive_aspect_concept_text = aspects["positive_semantic_space"]

            if aspects["negative_semantic_space"]:
                if not negative_aspect_concept_text:
                    negative_aspect_concept_text = aspects["negative_semantic_space"]

            break

    if sentence_model:
        main_query_embedding = sentence_model.encode(QUERY_PREFIX + combined_main_query_text)

        if positive_aspect_concept_text:
            positive_aspect_embedding = sentence_model.encode(QUERY_PREFIX + positive_aspect_concept_text)

        if negative_aspect_concept_text:
            negative_aspect_embedding = sentence_model.encode(QUERY_PREFIX + negative_aspect_concept_text)

    else:
        print("Sentence Transformer not available. Fallback to basic keyword extraction.")

        main_query_embedding = set()
        if spacy_nlp_model:
            doc = spacy_nlp_model(combined_main_query_text.lower())
            main_query_embedding.update([
                token.lemma_ for token in doc
                if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB']
            ])
            for chunk in doc.noun_chunks:
                 main_query_embedding.add(chunk.text.lower())
        else:
            main_query_embedding = set(re.findall(r'\b\w{3,}\b', combined_main_query_text.lower()))
        main_query_embedding = list(main_query_embedding)

        if positive_aspect_concept_text:
            positive_aspect_embedding = set()
            if spacy_nlp_model:
                doc = spacy_nlp_model(positive_aspect_concept_text.lower())
                positive_aspect_embedding.update([
                    token.lemma_ for token in doc
                    if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB']
                ])
                for chunk in doc.noun_chunks:
                     positive_aspect_embedding.add(chunk.text.lower())
            else:
                positive_aspect_embedding = set(re.findall(r'\b\w{3,}\b', positive_aspect_concept_text.lower()))
            positive_aspect_embedding = list(positive_aspect_embedding)

        if negative_aspect_concept_text:
            negative_aspect_embedding = set()
            if spacy_nlp_model:
                doc = spacy_nlp_model(negative_aspect_concept_text.lower())
                negative_aspect_embedding.update([
                    token.lemma_ for token in doc
                    if token.is_alpha and not token.is_stop and token.pos_ in ['NOUN', 'PROPN', 'ADJ', 'VERB']
                ])
                for chunk in doc.noun_chunks:
                     negative_aspect_embedding.add(chunk.text.lower())
            else:
                negative_aspect_embedding = set(re.findall(r'\b\w{3,}\b', negative_aspect_concept_text.lower()))
            negative_aspect_embedding = list(negative_aspect_embedding)

    return main_query_embedding, positive_aspect_embedding, negative_aspect_embedding

def rank_sections(processed_docs, main_query_representation, positive_aspect_representation, negative_aspect_representation):
    ranked_sections = []

    if sentence_model is not None and isinstance(main_query_representation, np.ndarray):
        main_query_embedding = main_query_representation.reshape(1, -1)

        section_embeddings = []
        for doc_section in processed_docs:
            try:
                text_to_encode = doc_section["text_content"]
                if not text_to_encode or len(text_to_encode) < MIN_SECTION_LENGTH:
                     section_embeddings.append(np.zeros(sentence_model.get_sentence_embedding_dimension()))
                     continue

                if len(text_to_encode) > MAX_SECTION_LENGTH:
                    text_to_encode = text_to_encode[:MAX_SECTION_LENGTH]

                section_embeddings.append(sentence_model.encode(DOCUMENT_PREFIX + text_to_encode))
            except Exception as e:
                print(f"Warning: Could not encode section text for '{doc_section['document']}' (Page {doc_section['page_number']}): {e}")
                section_embeddings.append(np.zeros(sentence_model.get_sentence_embedding_dimension()))

        if len(section_embeddings) == 0:
            return []

        main_similarities = cosine_similarity(main_query_embedding, np.array(section_embeddings))[0]

        positive_aspect_similarities = None
        if positive_aspect_representation is not None and isinstance(positive_aspect_representation, np.ndarray):
            positive_aspect_embedding = positive_aspect_representation.reshape(1, -1)
            positive_aspect_similarities = cosine_similarity(positive_aspect_embedding, np.array(section_embeddings))[0]

        negative_aspect_similarities = None
        if negative_aspect_representation is not None and isinstance(negative_aspect_representation, np.ndarray):
            negative_aspect_embedding = negative_aspect_representation.reshape(1, -1)
            negative_aspect_similarities = cosine_similarity(negative_aspect_embedding, np.array(section_embeddings))[0]

        for i, main_sim_score in enumerate(main_similarities):
            doc_section = processed_docs[i]
            final_score = float(main_sim_score * MAIN_QUERY_WEIGHT)

            if positive_aspect_similarities is not None:
                pos_aspect_sim = positive_aspect_similarities[i]
                if pos_aspect_sim > ASPECT_SIMILARITY_THRESHOLD:
                    final_score += (pos_aspect_sim * POSITIVE_ASPECT_WEIGHT)

            if negative_aspect_similarities is not None:
                neg_aspect_sim = negative_aspect_similarities[i]
                if neg_aspect_sim > ASPECT_SIMILARITY_THRESHOLD:
                    final_score -= (neg_aspect_sim * NEGATIVE_ASPECT_WEIGHT)

            ranked_sections.append({
                "document": doc_section["document"],
                "page_number": doc_section["page_number"],
                "section_title": doc_section["section_title_candidate"],
                "importance_score": final_score,
                "text_content": doc_section["text_content"]
            })

        ranked_sections = [s for s in ranked_sections if s["importance_score"] > 0]
        ranked_sections.sort(key=lambda x: x["importance_score"], reverse=True)

    else:
        print("Using keyword-based ranking (semantic model not available).")
        main_keywords = set(main_query_representation)
        positive_aspect_keywords = set(positive_aspect_representation) if positive_aspect_representation else set()
        negative_aspect_keywords = set(negative_aspect_representation) if negative_aspect_representation else set()

        for doc_section in processed_docs:
            score = 0
            section_text_lower = doc_section["text_content"].lower()

            for keyword in main_keywords:
                if len(keyword) > 2:
                    if ' ' in keyword:
                        score += section_text_lower.count(keyword) * 2
                    else:
                        score += len(re.findall(r'\b' + re.escape(keyword) + r'\b', section_text_lower))

            title_lower = doc_section["section_title_candidate"].lower()
            for keyword in main_keywords:
                if keyword in title_lower:
                    score += 5

            if positive_aspect_keywords:
                for boost_keyword in positive_aspect_keywords:
                    if len(boost_keyword) > 2:
                        if ' ' in boost_keyword:
                            score += section_text_lower.count(boost_keyword) * 15
                        else:
                            score += len(re.findall(r'\b' + re.escape(boost_keyword) + r'\b', section_text_lower)) * 8

            if negative_aspect_keywords:
                for neg_keyword in negative_aspect_keywords:
                    if len(neg_keyword) > 2:
                        if ' ' in neg_keyword:
                            score -= section_text_lower.count(neg_keyword) * 20
                        else:
                            score -= len(re.findall(r'\b' + re.escape(neg_keyword) + r'\b', section_text_lower)) * 10

            ranked_sections.append({
                "document": doc_section["document"],
                "page_number": doc_section["page_number"],
                "section_title": doc_section["section_title_candidate"],
                "importance_score": score,
                "text_content": doc_section["text_content"]
            })
        ranked_sections = [s for s in ranked_sections if s["importance_score"] > 0]
        ranked_sections.sort(key=lambda x: x["importance_score"], reverse=True)

    for i, section in enumerate(ranked_sections):
        section["importance_rank"] = i + 1

    return ranked_sections

def analyze_subsections(ranked_sections, main_query_representation, positive_aspect_representation, negative_aspect_representation):
    sub_section_analysis_results = []

    main_query_embedding = None
    positive_aspect_embedding = None
    negative_aspect_embedding = None

    main_query_keywords = set()
    positive_aspect_keywords = set()
    negative_aspect_keywords = set()

    if sentence_model and isinstance(main_query_representation, np.ndarray):
        main_query_embedding = main_query_representation.reshape(1, -1)
        if positive_aspect_representation is not None and isinstance(positive_aspect_representation, np.ndarray):
            positive_aspect_embedding = positive_aspect_representation.reshape(1, -1)
        if negative_aspect_representation is not None and isinstance(negative_aspect_representation, np.ndarray):
            negative_aspect_embedding = negative_aspect_representation.reshape(1, -1)
    else:
        main_query_keywords = set(main_query_representation)
        if positive_aspect_representation:
            positive_aspect_keywords = set(positive_aspect_representation)
        if negative_aspect_representation:
            negative_aspect_keywords = set(negative_aspect_representation)
        print(f"Warning: Sub-section analysis using keyword-based approach.")


    for i, section in enumerate(ranked_sections):
        if i >= TOP_N_SECTIONS_FOR_ANALYSIS:
            break

        refined_text_list = []

        try:
            sentences = []
            if spacy_nlp_model:
                doc = spacy_nlp_model(section["text_content"])
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            else:
                sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', section["text_content"])
                sentences = [s.strip() for s in sentences if s.strip()]

            meaningful_sentences = [s for s in sentences if len(s) > 15 and s.strip()]
            if not meaningful_sentences:
                print(f"No meaningful sentences found for {section['document']} (Page {section['page_number']}). Skipping refined text for this section.")
                continue

            if sentence_model and main_query_embedding is not None:
                sentences_with_prefix = [DOCUMENT_PREFIX + s for s in meaningful_sentences]
                sentence_embeddings = sentence_model.encode(sentences_with_prefix)
                main_sentence_similarities = cosine_similarity(main_query_embedding, sentence_embeddings)[0]

                sentence_scores_with_indices = []

                positive_aspect_sentence_similarities = None
                if positive_aspect_embedding is not None:
                    positive_aspect_sentence_similarities = cosine_similarity(positive_aspect_embedding, sentence_embeddings)[0]

                negative_aspect_sentence_similarities = None
                if negative_aspect_embedding is not None:
                    negative_aspect_sentence_similarities = cosine_similarity(negative_aspect_embedding, sentence_embeddings)[0]

                for idx, main_sim_score in enumerate(main_sentence_similarities):
                    final_sentence_score = float(main_sim_score * MAIN_QUERY_WEIGHT)

                    if positive_aspect_sentence_similarities is not None:
                        pos_aspect_sim = positive_aspect_sentence_similarities[idx]
                        if pos_aspect_sim > ASPECT_SIMILARITY_THRESHOLD:
                            final_sentence_score += (pos_aspect_sim * POSITIVE_ASPECT_WEIGHT)

                    if negative_aspect_sentence_similarities is not None:
                        neg_aspect_sim = negative_aspect_sentence_similarities[idx]
                        if neg_aspect_sim > ASPECT_SIMILARITY_THRESHOLD:
                            final_sentence_score -= (neg_aspect_sim * NEGATIVE_ASPECT_WEIGHT)

                    sentence_scores_with_indices.append((final_sentence_score, idx))

                sentence_scores_with_indices = [s for s in sentence_scores_with_indices if s[0] > 0]

                selected_sentences_with_scores = heapq.nlargest(
                    min(MAX_SENTENCES_FOR_REFINED_TEXT, len(sentence_scores_with_indices)),
                    sentence_scores_with_indices,
                    key=lambda x: x[0]
                )

                sorted_selected_sentences = sorted(selected_sentences_with_scores, key=lambda x: x[1])
                for score, idx in sorted_selected_sentences:
                    refined_text_list.append(meaningful_sentences[idx])

            else:
                sentence_scores = []
                for sent_idx, sentence in enumerate(meaningful_sentences):
                    score = 0
                    sentence_words = re.findall(r'\b\w+\b', sentence.lower())

                    for keyword in main_query_keywords:
                        if len(keyword) > 2:
                            if ' ' in keyword:
                                score += sentence.lower().count(keyword) * 2
                            else:
                                score += sentence_words.count(keyword)

                    if positive_aspect_keywords:
                        for boost_keyword in positive_aspect_keywords:
                            if len(boost_keyword) > 2:
                                if ' ' in boost_keyword:
                                    score += sentence.lower().count(boost_keyword) * 15
                                else:
                                    score += sentence_words.count(boost_keyword) * 8

                    if negative_aspect_keywords:
                        for neg_keyword in negative_aspect_keywords:
                            if len(neg_keyword) > 2:
                                if ' ' in neg_keyword:
                                    score -= sentence.lower().count(neg_keyword) * 20
                                else:
                                    score -= sentence_words.count(neg_keyword) * 10

                    score += (len(meaningful_sentences) - sent_idx) * 0.1
                    sentence_scores.append((score, sent_idx))

                sentence_scores = [s for s in sentence_scores if s[0] > 0]

                top_sentences_with_scores = heapq.nlargest(
                    min(MAX_SENTENCES_FOR_REFINED_TEXT, len(sentence_scores)),
                    sentence_scores,
                    key=lambda x: x[0]
                )

                sorted_top_sentences = sorted(top_sentences_with_scores, key=lambda x: x[1])
                for score, idx in sorted_top_sentences:
                    refined_text_list.append(meaningful_sentences[idx])

            if refined_text_list:
                sub_section_analysis_results.append({
                    "document": section["document"],
                    "section_title": section["section_title"],
                    "refined_text": " ".join(refined_text_list),
                    "page_number": section["page_number"]
                })
        except Exception as e:
            print(f"Error during sub-section analysis for {section['document']} (Page {section['page_number']}): {e}")
            continue

    return sub_section_analysis_results

def generate_output_json(input_documents, persona, job_to_be_done, extracted_sections_data, subsection_analysis_data):
    output_data = {
        "metadata": {
            "input_documents": input_documents,
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "sub_section_analysis": []
    }

    for section in extracted_sections_data[:TOP_N_SECTIONS_FOR_ANALYSIS]:
        output_data["extracted_sections"].append({
            "document": section["document"],
            "page_number": section["page_number"],
            "section_title": section["section_title"],
            "importance_rank": section["importance_rank"]
        })

    for sub_section in subsection_analysis_data:
        output_data["sub_section_analysis"].append({
            "document": sub_section["document"],
            "section_title": sub_section["section_title"],
            "refined_text": sub_section["refined_text"],
            "page_number": sub_section["page_number"]
        })

    return json.dumps(output_data, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Intelligent Document Analyst")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input JSON file (e.g., challenge1b_input.json).")
    parser.add_argument("--pdf_dir", type=str, required=True,
                        help="Path to the directory containing PDF documents (e.g., PDFs/).")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save the output JSON file.")
    parser.add_argument("--config_file", type=str, default="input/config.json",
                        help="Path to the JSON configuration file (default: config.json).")

    args = parser.parse_args()

    global CONFIG, MODEL_PATH, SPACY_MODEL_NAME, MIN_SECTION_LENGTH, MAX_SECTION_LENGTH, MAX_SENTENCES_FOR_REFINED_TEXT, TOP_N_SECTIONS_FOR_ANALYSIS, QUERY_PREFIX, DOCUMENT_PREFIX, GENERAL_NEGATION_PATTERNS, CATEGORY_ASPECTS, MAIN_QUERY_WEIGHT, POSITIVE_ASPECT_WEIGHT, NEGATIVE_ASPECT_WEIGHT, ASPECT_SIMILARITY_THRESHOLD
    if args.config_file != CONFIG_FILE:
        try:
            with open(args.config_file, 'r', encoding='utf-8') as f:
                CONFIG = json.load(f)
            print(f"Configuration loaded from {args.config_file}")
            MODEL_PATH = CONFIG.get('MODEL_PATH', './model_weights/bge-base-en-v1.5')
            SPACY_MODEL_NAME = CONFIG.get('SPACY_MODEL_NAME', 'en_core_web_sm')
            MIN_SECTION_LENGTH = CONFIG.get('MIN_SECTION_LENGTH', 50)
            MAX_SECTION_LENGTH = CONFIG.get('MAX_SECTION_LENGTH', 4000)
            MAX_SENTENCES_FOR_REFINED_TEXT = CONFIG.get('MAX_SENTENCES_FOR_REFINED_TEXT', 3)
            TOP_N_SECTIONS_FOR_ANALYSIS = CONFIG.get('TOP_N_SECTIONS_FOR_ANALYSIS', 5)
            QUERY_PREFIX = CONFIG.get('QUERY_PREFIX', "Represent this sentence for searching relevant passages:")
            DOCUMENT_PREFIX = CONFIG.get('DOCUMENT_PREFIX', "Represent this passage for searching relevant passages:")
            GENERAL_NEGATION_PATTERNS = CONFIG.get('GENERAL_NEGATION_PATTERNS', [])
            CATEGORY_ASPECTS = CONFIG.get('CATEGORY_ASPECTS', {})
            MAIN_QUERY_WEIGHT = CONFIG.get('MAIN_QUERY_WEIGHT', 1.0)
            POSITIVE_ASPECT_WEIGHT = CONFIG.get('POSITIVE_ASPECT_WEIGHT', 0.5)
            NEGATIVE_ASPECT_WEIGHT = CONFIG.get('NEGATIVE_ASPECT_WEIGHT', 0.7)
            ASPECT_SIMILARITY_THRESHOLD = CONFIG.get('ASPECT_SIMILARITY_THRESHOLD', 0.2)
        except FileNotFoundError:
            print(f"Error: Custom config file {args.config_file} not found.")
            exit(1)
        except json.JSONDecodeError as e:
            print(f"Error decoding custom config file {args.config_file}: {e}.")
            exit(1)

    start_time = time.time()

    # Read the input JSON file
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        print(f"Input data loaded from {args.input_file}")
    except FileNotFoundError:
        print(f"Error: Input file {args.input_file} not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding input JSON file {args.input_file}: {e}. Please check JSON syntax.")
        return

    # Extract persona, job_to_be_done, and document information from the input JSON
    persona_description = input_data["persona"]["role"]
    job_to_be_done = input_data["job_to_be_done"]["task"]
    input_documents_info = input_data["documents"]

    if not persona_description:
        print("Warning: Persona description is empty. This may affect relevance.")
    if not job_to_be_done:
        print("Warning: Job-to-be-Done is empty. This may affect relevance.")

    document_paths = []
    for doc_info in input_documents_info:
        doc_path = os.path.join(args.pdf_dir, doc_info["filename"])
        if os.path.exists(doc_path):
            document_paths.append(doc_path)
        else:
            print(f"Warning: Document '{doc_info['filename']}' not found at '{doc_path}'. Skipping.")

    if not document_paths:
        print(f"Error: No PDF documents found based on the input JSON and PDF directory {args.pdf_dir}. Please ensure the files exist.")
        return

    print(f"Starting analysis for {len(document_paths)} documents.")

    print("Step 1/4: Preprocessing documents by identifying logical sections...")
    processed_docs = preprocess_documents(document_paths)
    if not processed_docs:
        print("No content extracted or valid sections identified from documents. Exiting.")
        return
    print(f"Preprocessing complete. Extracted {len(processed_docs)} sections.")

    print("Step 2/4: Understanding Persona and Job-to-be-Done (Main, Positive Aspect, and Negative Aspect queries)...")
    main_query_representation, positive_aspect_representation, negative_aspect_representation = \
        get_query_vectors(persona_description, job_to_be_done)
    print("Query understanding complete.")

    print("Step 3/4: Ranking document sections based on relevance with dynamic aspect boosting and exclusion...")
    ranked_sections = rank_sections(processed_docs, main_query_representation,
                                    positive_aspect_representation, negative_aspect_representation)

    if not ranked_sections:
        print("No sections could be ranked. Exiting.")
        return

    print(f"Sections ranked. Displaying top {min(TOP_N_SECTIONS_FOR_ANALYSIS, len(ranked_sections))} sections:")
    for i, section in enumerate(ranked_sections[:min(TOP_N_SECTIONS_FOR_ANALYSIS, len(ranked_sections))]):
        print(f"- Rank {section['importance_rank']:.0f} (Score: {section['importance_score']:.4f}): '{section['section_title']}' (Doc: {section['document']}, Page: {section['page_number']})")

    print("Step 4/4: Performing sub-section analysis on top sections for refined text extraction with dynamic aspect boosting and exclusion...")
    subsection_analysis_results = analyze_subsections(
        ranked_sections,
        main_query_representation,
        positive_aspect_representation,
        negative_aspect_representation
    )
    print(f"Sub-section analysis complete for {len(subsection_analysis_results)} top sections.")

    # Pass the original documents info from the JSON to generate_output_json
    output_json = generate_output_json(
        input_documents_info, # Use the document info from the input JSON
        persona_description,
        job_to_be_done,
        ranked_sections,
        subsection_analysis_results
    )

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"\nProcessing finished successfully.")
        print(f"Output saved to {args.output}")
    except Exception as e:
        print(f"Error saving output JSON to {args.output}: {e}")

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Total processing time: {processing_time:.2f} seconds.")

if __name__ == "__main__":
    main()