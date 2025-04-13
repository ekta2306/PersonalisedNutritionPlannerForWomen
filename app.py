from flask import Flask, render_template, request
import pandas as pd
import networkx as nx
import os
import random
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

app = Flask(__name__)

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=10)
symptom_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

DATASET_FOLDER = "datasets"
HEALTH_DATA_PATH = os.path.join(DATASET_FOLDER, "detailed_meals_macros_.csv")
FOOD_DATA_PATH = os.path.join(DATASET_FOLDER, "FOOD-DATA-GROUP1.csv")

POSSIBLE_NUTRIENTS = [
    "Vitamin A", "Vitamin B1", "Vitamin B2", "Vitamin B3", "Vitamin B5", "Vitamin B6", "Vitamin B11", "Vitamin B12",
    "Vitamin C", "Vitamin D", "Vitamin E", "Vitamin K", "Calcium", "Iron", "Magnesium", "Manganese", "Phosphorus",
    "Potassium", "Zinc", "Protein", "Carbohydrates", "Sugars", "Fiber", "Fat", "Sodium"
]

G = None
def build_knowledge_graph(health_data_path, food_data_group_path):
    print("[DEBUG] Building knowledge graph...")
    health_data = pd.read_csv(health_data_path)
    food_data = pd.read_csv(food_data_group_path)
    print("[DEBUG] Food CSV Columns:", food_data.columns.tolist())

    G = nx.Graph()
    for idx, row in health_data.iterrows():
        disease = row['Disease']
        if pd.notna(disease):
            age = row.get('Age')
            weight = row.get('Weight')
            height = row.get('Height')
            gender = row.get('Gender')
            calorie_target = pd.to_numeric(row.get('Daily Calorie Target', None), errors='coerce')
            label = f"{disease} | Age:{age}, Wt:{weight}, Ht:{height}, Cal:{calorie_target}"
            G.add_node(label, type="disease", age=age, weight=weight, height=height, gender=gender, daily_calorie_target=calorie_target)

            for nutrient in health_data.columns[1:]:
                if nutrient in ["Age", "Weight", "Height", "Gender", "Daily Calorie Target"]:
                    continue
                value = pd.to_numeric(row[nutrient], errors='coerce')
                if pd.notna(value) and value > 0:
                    if nutrient not in G.nodes:
                        G.add_node(nutrient, type="nutrient")
                    G.add_edge(label, nutrient, weight=value)

    for _, row in food_data.iterrows():
        food = row['food']
        caloric_value = pd.to_numeric(row.get("Caloric Value", None), errors='coerce')
        G.add_node(food, type="food", caloric_value=caloric_value)
        for nutrient in food_data.columns[1:]:
            if nutrient == "Caloric Value":
                continue
            value = pd.to_numeric(row[nutrient], errors='coerce')
            if pd.notna(value) and value > 0:
                if nutrient not in G.nodes:
                    G.add_node(nutrient, type="nutrient")
                G.add_edge(food, nutrient, weight=value)

    print("[INFO] Knowledge graph built.")
    return G

def infer_required_nutrients_from_label(label_text):
    result = symptom_classifier(label_text)
    inferred_nutrients = set()
    for item in result:
        for nutrient in POSSIBLE_NUTRIENTS:
            if nutrient.lower() in item['label'].lower():
                inferred_nutrients.add(nutrient)
    print(f"[INFO] Inferred nutrients: {inferred_nutrients}")
    return inferred_nutrients

def map_symptom_to_diseases(symptom, G):
    result = symptom_classifier(symptom)
    disease_nodes = [node for node in G.nodes if G.nodes[node].get('type') == 'disease']
    base_diseases = list(sorted(set(node.split('|')[0].strip() for node in disease_nodes)))

    disease_mapping = []
    for res in result:
        label = res['label']
        score = res['score']
        try:
            label_index = int(label.split('_')[-1])
            if 0 <= label_index < len(base_diseases):
                matched_base = base_diseases[label_index]
                matches = [node for node in disease_nodes if matched_base.lower() in node.lower()]
                for node in matches:
                    disease_mapping.append((node, score))
        except Exception as e:
            print("[ERROR] Failed to parse label:", label, e)
    print(f"[DEBUG] Disease mapping: {disease_mapping}")
    return disease_mapping

def get_personalized_recommendations(symptom, G, user_age=None, user_weight=None, user_height=None):
    disease_mapping = map_symptom_to_diseases(symptom, G)
    if not disease_mapping:
        print("[INFO] No disease mapping found.")
        return ["No suitable foods found."]

    best_disease_label, best_score = max(disease_mapping, key=lambda x: x[1])
    print(f"[INFO] Selected disease: {best_disease_label} with score {best_score}")

    matched_disease_nodes = [
        node for node in G.nodes
        if G.nodes[node].get("type") == "disease" and node.lower().startswith(best_disease_label.lower())
    ]

    if matched_disease_nodes:
        def metadata_match_score(node):
            meta = G.nodes[node]
            score = 0
            if user_age is not None and meta.get("age") is not None:
                score -= abs(user_age - meta["age"])
            if user_weight is not None and meta.get("weight") is not None:
                score -= abs(user_weight - meta["weight"])
            if user_height is not None and meta.get("height") is not None:
                score -= abs(user_height - meta["height"])
            return score

        matched_disease_node = max(
            matched_disease_nodes,
            key=lambda node: (G.nodes[node].get("daily_calorie_target") is not None, metadata_match_score(node))
        )

        print(f"[INFO] Matched disease node (metadata-based): {matched_disease_node}")
        calorie_target = G.nodes[matched_disease_node].get("daily_calorie_target")
        print(f"[DEBUG] Extracted daily calorie target: {calorie_target}")

        neighbors = list(G.neighbors(matched_disease_node))
        required_nutrients = {node for node in neighbors if G.nodes[node].get("type") == "nutrient"}

        print(f"[DEBUG] Nutrients linked to disease node:")
        for nutrient in required_nutrients:
            print(f"  - {nutrient}")

    else:
        print(f"[INFO] Disease label '{best_disease_label}' not found in graph. Inferring nutrients...")
        required_nutrients = infer_required_nutrients_from_label(best_disease_label)
        calorie_target = None

    if not required_nutrients:
        print("[INFO] No nutrients found for disease.")
        return ["No suitable foods found."]

    food_recs = set()
    for nutrient in required_nutrients:
        if nutrient in G.nodes and G.nodes[nutrient]['type'] == "nutrient":
            linked_foods = [food for food in G.neighbors(nutrient) if G.nodes[food]['type'] == "food"]
            food_recs.update(linked_foods)

    print(f"[DEBUG] Candidate foods before calorie filter: {len(food_recs)}")
    print(f"[DEBUG] Extracted daily calorie target: {calorie_target}")
    if calorie_target:       
        min_cal = calorie_target * 0.25
        max_cal = calorie_target  # Allow anything below the target
        print(f"[DEBUG] Filtering foods with calorie between {min_cal} and {max_cal}")
        filtered_foods = []
        for food in food_recs:
            food_cal = G.nodes[food].get("caloric_value")
            if food_cal is not None:
                print(f"  - {food}: {food_cal} cal")
                if min_cal <= food_cal <= max_cal:
                    filtered_foods.append(food)


        food_recs = filtered_foods
        print(f"[DEBUG] Foods after calorie filter: {len(food_recs)}")
    else:
        print("[INFO] Skipping calorie filter as no target was found.")

    if not food_recs:
        return ["No suitable foods found."]

    return random.sample(food_recs, min(10, len(food_recs)))

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/recommendation", methods=["GET", "POST"])
def recommendation():
    global G
    recommended_foods = []
    if request.method == "POST":
        symptom = request.form.get("symptom", "").strip()
        recommended_foods = get_personalized_recommendations(symptom, G)
    return render_template("recommendation.html", recommended_foods=recommended_foods)

if __name__ == "__main__":
    G = build_knowledge_graph(HEALTH_DATA_PATH, FOOD_DATA_PATH)
    app.run(debug=True)
