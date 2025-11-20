import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="n_jobs value.*overridden")

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import seaborn as sns
from umap import UMAP
from model.data_loader import DataLoader
from model.heuristic import HeuristicRecommender
from model.random_forest import RandomForestRecommender
from model.knn import KNNRecommender
from model.content_based import ContentBasedRecommender
from model.random_recommender import RandomRecommender

BG_COLOR = '#8B4513'
TEXT_COLOR = 'white'
PLOT_COLOR = "#4BAA4B"
KDE_COLOR = '#FFFFE0'

def calculate_mse(user_vector, recommended_df, feature_cols):
    """
    Calculate Mean Squared Error between user vector and average of recommendations.
    Lower MSE = better alignment.
    """
    if recommended_df.height == 0:
        return 10.0
        
    rec_matrix = recommended_df.select(feature_cols).to_numpy()
    avg_rec_vector = np.mean(rec_matrix, axis=0)
    mse = np.mean((user_vector - avg_rec_vector) ** 2)
    return mse

def create_ground_truth(user_vec, all_features, df, top_n=30):
    """
    Create ground truth by finding plants most similar to user preferences.
    Returns list of scientific names of top_n most similar plants.
    """
    distances = np.linalg.norm(all_features - user_vec, axis=1)
    top_indices = np.argsort(distances)[:top_n]
    return df[top_indices.tolist()]["scientific_name"].to_list()

def precision_at_k(recommendations, ground_truth, k=3):
    """
    Calculate Precision@k: proportion of recommended items that are relevant.
    """
    if len(recommendations) == 0:
        return 0.0
    relevant_count = len(set(recommendations[:k]) & set(ground_truth))
    return relevant_count / min(k, len(recommendations))

def recall_at_k(recommendations, ground_truth, k=3):
    """
    Calculate Recall@k: proportion of relevant items that were recommended.
    """
    if len(ground_truth) == 0:
        return 0.0
    relevant_count = len(set(recommendations[:k]) & set(ground_truth))
    return relevant_count / len(ground_truth)

def average_precision(recommendations, ground_truth, k=3):
    """
    Calculate Average Precision for a single query.
    """
    if len(recommendations) == 0:
        return 0.0
    
    precision_sum = 0.0
    relevant_count = 0
    
    for i in range(min(k, len(recommendations))):
        if recommendations[i] in ground_truth:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    
    if relevant_count == 0:
        return 0.0
    
    return precision_sum / min(len(ground_truth), k)

def ndcg_at_k(recommendations, ground_truth, all_features, user_vec, df, k=3):
    """
    Calculate NDCG@k (Normalized Discounted Cumulative Gain).
    Uses distance as relevance score (closer = more relevant).
    """
    if len(recommendations) == 0:
        return 0.0
    
    def get_relevance_score(plant_name):
        plant_row = df.filter(pl.col("scientific_name") == plant_name)
        if plant_row.height == 0:
            return 0.0
        feature_cols = ['light_level', 'water_need', 'humidity_need', 'temp_tolerance']
        plant_vec = plant_row.select(feature_cols).to_numpy()[0]
        distance = np.linalg.norm(plant_vec - user_vec)
        return 1.0 / (1.0 + distance)
    
    dcg = 0.0
    for i in range(min(k, len(recommendations))):
        relevance = get_relevance_score(recommendations[i])
        dcg += relevance / np.log2(i + 2)
    
    ideal_scores = sorted([get_relevance_score(name) for name in ground_truth], reverse=True)
    idcg = 0.0
    for i in range(min(k, len(ideal_scores))):
        idcg += ideal_scores[i] / np.log2(i + 2)
    
    if idcg == 0.0:
        return 0.0
    
    return dcg / idcg

def get_inputs_from_vector(user_vec):
    """
    Maps the numeric user vector back to the questionnaire strings 
    that your models expect.
    Vector format: [light, water, humidity, temp]
    """
    light_val, water_val, humid_val, temp_val = user_vec
    
    inputs = {
        "flowers": "Dont Care", 
        "toxic": "Dont Care"
    }

    if light_val <= 1.5:
        inputs["light"] = "South"
    elif light_val <= 2.5:
        inputs["light"] = "East"
    elif light_val <= 3.5:
        inputs["light"] = "North"
    else:
        inputs["light"] = "North"

    if water_val <= 1.6:
        inputs["care"] = "High"    # Frequent watering
    elif water_val <= 2.4:
        inputs["care"] = "Medium"
    else:
        inputs["care"] = "Low"     # Infrequent

    if humid_val <= 1.8:
        inputs["room"] = "Bathroom" # High humidity
    elif humid_val <= 2.4:
        inputs["room"] = "Kitchen"
    else:
        inputs["room"] = "Living Room" # Avg/Low humidity
        
    return inputs

def alignment_test_questionnaire(df, models, n_users=100, k=3):
    """
    Test questionnaire models with both MSE and standard recommendation metrics.
    """
    feature_cols = ['light_level', 'water_need', 'humidity_need', 'temp_tolerance']
    all_features = df.select(feature_cols).to_numpy()
    
    results = {
        name: {
            'mse': [],
            'precision': [],
            'recall': [],
            'map': [],
            'ndcg': []
        } for name in models.keys()
    }
    
    print(f"Running Questionnaire Alignment Test with {n_users} synthetic users...")
    
    for i in range(n_users):
        user_vec = np.array([
            np.random.uniform(1, 4),
            np.random.uniform(1, 3),
            np.random.uniform(1, 3),
            np.random.uniform(1, 3)
        ])
        
        ground_truth = create_ground_truth(user_vec, all_features, df, top_n=30)
        inputs = get_inputs_from_vector(user_vec)
        
        for model_name, model in models.items():
            try:
                recs = model.recommend(inputs, top_k=k)
                if recs.height > 0:
                    # MSE
                    mse = calculate_mse(user_vec, recs, feature_cols)
                    results[model_name]['mse'].append(mse)
                    
                    # Get recommendations as list
                    rec_names = recs["scientific_name"].to_list()
                    
                    # Precision@k
                    prec = precision_at_k(rec_names, ground_truth, k)
                    results[model_name]['precision'].append(prec)
                    
                    # Recall@k
                    rec = recall_at_k(rec_names, ground_truth, k)
                    results[model_name]['recall'].append(rec)
                    
                    # Average Precision
                    ap = average_precision(rec_names, ground_truth, k)
                    results[model_name]['map'].append(ap)
                    
                    # NDCG@k
                    ndcg = ndcg_at_k(rec_names, ground_truth, all_features, user_vec, df, k)
                    results[model_name]['ndcg'].append(ndcg)
                    
            except Exception as e:
                pass
    
    return results

def alignment_test_itembased(df, models, n_users=100, k=3):
    """
    Test item-based models with both MSE and standard recommendation metrics.
    """
    feature_cols = ['light_level', 'water_need', 'humidity_need', 'temp_tolerance']
    all_features = df.select(feature_cols).to_numpy()
    
    results = {
        name: {
            'mse': [],
            'precision': [],
            'recall': [],
            'map': [],
            'ndcg': []
        } for name in models.keys()
    }
    
    print(f"Running Item-Based Alignment Test with {n_users} synthetic users...")
    
    for i in range(n_users):
        # Generate random user preference vector
        user_vec = np.array([
            np.random.uniform(1, 4),
            np.random.uniform(1, 3),
            np.random.uniform(1, 3),
            np.random.uniform(1, 3)
        ])
        
        # Create ground truth
        ground_truth = create_ground_truth(user_vec, all_features, df, top_n=30)
        
        # Find closest plant as seed
        distances = np.linalg.norm(all_features - user_vec, axis=1)
        closest_idx = int(np.argmin(distances))
        seed_plant = df[closest_idx]["scientific_name"][0]
        
        inputs = {"plants": [seed_plant], "flowers": "Dont Care", "toxic": "Dont Care"}
        
        for model_name, model in models.items():
            try:
                recs = model.recommend(inputs, top_k=k)
                if recs.height > 0:
                    # MSE
                    mse = calculate_mse(user_vec, recs, feature_cols)
                    results[model_name]['mse'].append(mse)
                    
                    # Get recommendations as list
                    rec_names = recs["scientific_name"].to_list()
                    
                    # Precision@k
                    prec = precision_at_k(rec_names, ground_truth, k)
                    results[model_name]['precision'].append(prec)
                    
                    # Recall@k
                    rec = recall_at_k(rec_names, ground_truth, k)
                    results[model_name]['recall'].append(rec)
                    
                    # Average Precision
                    ap = average_precision(rec_names, ground_truth, k)
                    results[model_name]['map'].append(ap)
                    
                    # NDCG@k
                    ndcg = ndcg_at_k(rec_names, ground_truth, all_features, user_vec, df, k)
                    results[model_name]['ndcg'].append(ndcg)
                    
            except:
                pass
    
    return results

def coverage_test_all_models(df, all_models, n_users=100):
    """
    Test catalog coverage for all models.
    """
    coverage = {name: set() for name in all_models.keys()}
    
    print(f"Running Coverage Test with {n_users} synthetic users...")
    
    # Test questionnaire models
    lights = ["North", "South", "East", "West", "Grow Light"]
    cares = ["Low", "Medium", "High"]
    rooms = ["Living Room", "Bathroom", "Bedroom", "Kitchen"]
    
    for i in range(n_users // 2):
        inputs = {
            "light": np.random.choice(lights),
            "care": np.random.choice(cares),
            "room": np.random.choice(rooms),
            "flowers": "Dont Care",
            "toxic": "Dont Care"
        }
        
        for model_name in ["Heuristic", "Random Forest", "Random"]:
            if model_name in all_models:
                try:
                    recs = all_models[model_name].recommend(inputs, top_k=5)
                    coverage[model_name].update(recs["scientific_name"].to_list())
                except:
                    pass
    
    # Test item-based models
    for i in range(n_users // 2):
        seed_plant = df.sample(1)["scientific_name"][0]
        inputs = {"plants": [seed_plant], "flowers": "Dont Care", "toxic": "Dont Care"}
        
        for model_name in ["KNN", "Content-Based", "Random"]:
            if model_name in all_models:
                try:
                    recs = all_models[model_name].recommend(inputs, top_k=5)
                    coverage[model_name].update(recs["scientific_name"].to_list())
                except:
                    pass
    
    total_plants = df.height
    results = {}
    for model_name, plants in coverage.items():
        results[model_name] = {
            'coverage': len(plants),
            'pct': (len(plants) / total_plants) * 100
        }
    
    return results

def umap_visualization_3d(df, cb_model):
    """
    Create interactive 3D UMAP visualization showing recommendations cluster around user input.
    """
    feature_cols = ['light_level', 'water_need', 'humidity_need', 'temp_tolerance']
    
    print("Creating 3D UMAP visualization...")
    
    # Fit UMAP to 3D
    all_features = df.select(feature_cols).to_numpy()
    umap_model = UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
    coords_3d = umap_model.fit_transform(all_features)
    
    # Create a specific user preference
    user_vec = np.array([[2.5, 1.5, 2.0, 2.0]])  # Mid-low light, low water
    user_3d = umap_model.transform(user_vec)[0]
    
    # Find closest plant as seed
    distances = np.linalg.norm(all_features - user_vec[0], axis=1)
    closest_idx = int(np.argmin(distances))
    seed_plant = df[closest_idx]["scientific_name"][0]
    
    # Get recommendations
    inputs = {"plants": [seed_plant], "flowers": "Dont Care", "toxic": "Dont Care"}
    recs = cb_model.recommend(inputs, top_k=10)
    
    # Prepare data for plotting
    plot_df = df.with_columns([
        pl.lit(coords_3d[:, 0]).alias("umap_x"),
        pl.lit(coords_3d[:, 1]).alias("umap_y"),
        pl.lit(coords_3d[:, 2]).alias("umap_z"),
        pl.lit("All Plants").alias("category")
    ])
    
    # Mark recommended plants
    rec_names = recs["scientific_name"].to_list()
    plot_df = plot_df.with_columns(
        pl.when(pl.col("scientific_name").is_in(rec_names))
        .then(pl.lit("Recommended"))
        .otherwise(pl.col("category"))
        .alias("category")
    )
    
    # Create plotly figure
    fig = go.Figure()
    
    # Plot all plants (gray)
    all_plants = plot_df.filter(pl.col("category") == "All Plants")
    
    # Prepare hover data with all desired fields
    hover_data_all = []
    for row in all_plants.iter_rows(named=True):
        hover_data_all.append([
            row['light_level'],
            row['water_need'],
            row['humidity_need'],
            row['scientific_name'],
            'Yes' if row.get('has_flowers', 0) == 1 else 'No'
        ])
    
    fig.add_trace(go.Scatter3d(
        x=all_plants["umap_x"].to_list(),
        y=all_plants["umap_y"].to_list(),
        z=all_plants["umap_z"].to_list(),
        mode='markers',
        marker=dict(size=3, color='green', opacity=0.5),
        name='All Plants',
        hovertemplate='<b>%{hovertext}</b><br>' +
                     'Scientific: %{customdata[3]}<br>' +
                     'Light: %{customdata[0]:.1f}<br>' +
                     'Water: %{customdata[1]:.1f}<br>' +
                     'Humidity: %{customdata[2]:.1f}<br>' +
                     'Has Flowers: %{customdata[4]}<extra></extra>',
        hovertext=all_plants["common_name"].to_list(),
        customdata=hover_data_all
    ))
    
    
    fig.update_layout(
        title={
            'text': 'Content-Based Recommendations in 3D Feature Space (UMAP)',
            'font': {'size': 16, 'color': TEXT_COLOR}
        },
        template='seaborn',
        scene=dict(
            xaxis=dict(title='UMAP 1', backgroundcolor='rgb(50,50,50)', gridcolor='gray'),
            yaxis=dict(title='UMAP 2', backgroundcolor='rgb(50,50,50)', gridcolor='gray'),
            zaxis=dict(title='UMAP 3', backgroundcolor='rgb(50,50,50)', gridcolor='gray'),
            bgcolor='rgb(30,30,30)'
        ),
        paper_bgcolor='rgb(30,30,30)',
        font=dict(color=TEXT_COLOR),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor=TEXT_COLOR,
            borderwidth=1
        ),
        height=800
    )
    
    # Save as HTML
    os.makedirs('evaluation_outputs/visualizations', exist_ok=True)
    fig.write_html('evaluation_outputs/visualizations/umap_3d_visualization.html')
    print("  Saved: evaluation_outputs/visualizations/umap_3d_visualization.html")
    
    return fig

def plot_metrics_comparison(q_alignment, i_alignment):
    """
    Create bar charts comparing all models across metrics.
    """
    # Prepare data for plotting
    metrics = ['precision', 'recall', 'map', 'ndcg']
    metric_labels = ['Precision@3', 'Recall@3', 'MAP@3', 'NDCG@3']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.set_facecolor(BG_COLOR)
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2, idx % 2]
        ax.set_facecolor(BG_COLOR)
        
        # Combine questionnaire and item-based results
        model_names = []
        scores = []
        model_types = []
        
        # Questionnaire models
        for model_name, metrics_dict in q_alignment.items():
            if metrics_dict[metric]:
                model_names.append(f"{model_name}\n(Q)")
                scores.append(np.mean(metrics_dict[metric]))
                model_types.append('Questionnaire')
        
        # Item-based models
        for model_name, metrics_dict in i_alignment.items():
            if metrics_dict[metric]:
                model_names.append(f"{model_name}\n(Item)")
                scores.append(np.mean(metrics_dict[metric]))
                model_types.append('Item-Based')
        
        # Create bar plot
        bars = ax.bar(range(len(model_names)), scores, color=PLOT_COLOR, alpha=0.8, edgecolor=TEXT_COLOR, linewidth=1.5)
        
        # Color questionnaire vs item-based differently
        for i, bar in enumerate(bars):
            if model_types[i] == 'Questionnaire':
                bar.set_color('#1B5234')  # Dark green for questionnaire
            else:
                bar.set_color(PLOT_COLOR)  # Light green for item-based
        
        # Styling
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=0, color=TEXT_COLOR, fontsize=9)
        ax.set_ylabel(label, color=TEXT_COLOR, fontsize=12, fontweight='bold')
        ax.set_title(f'{label} Comparison', color=TEXT_COLOR, fontsize=14, fontweight='bold')
        ax.tick_params(colors=TEXT_COLOR, which='both')
        ax.grid(True, alpha=0.2, color=TEXT_COLOR, axis='y')
        
        for spine in ax.spines.values():
            spine.set_edgecolor(TEXT_COLOR)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{score:.3f}',
                   ha='center', va='bottom', color=TEXT_COLOR, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('evaluation_outputs/visualizations/metrics_comparison_bars.png', facecolor=BG_COLOR, dpi=150)
    print("  Saved: evaluation_outputs/visualizations/metrics_comparison_bars.png")
    
    return fig

def plot_coverage_comparison(coverage_results):
    """
    Create bar chart for coverage comparison.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    model_names = list(coverage_results.keys())
    coverages = [coverage_results[m]['pct'] for m in model_names]
    
    bars = ax.bar(range(len(model_names)), coverages, color=PLOT_COLOR, alpha=0.8, edgecolor=TEXT_COLOR, linewidth=2)
    
    # Styling
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=15, color=TEXT_COLOR, fontsize=11)
    ax.set_ylabel('Catalog Coverage (%)', color=TEXT_COLOR, fontsize=12, fontweight='bold')
    ax.set_title('Model Coverage Comparison', color=TEXT_COLOR, fontsize=14, fontweight='bold')
    ax.tick_params(colors=TEXT_COLOR, which='both')
    ax.grid(True, alpha=0.2, color=TEXT_COLOR, axis='y')
    ax.set_ylim(0, 100)
    
    for spine in ax.spines.values():
        spine.set_edgecolor(TEXT_COLOR)
    
    # Add value labels
    for bar, cov in zip(bars, coverages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{cov:.1f}%',
               ha='center', va='bottom', color=TEXT_COLOR, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('evaluation_outputs/visualizations/coverage_comparison_bar.png', facecolor=BG_COLOR, dpi=150)
    print("  Saved: evaluation_outputs/visualizations/coverage_comparison_bar.png")
    
    return fig

def plot_mse_comparison_all(questionnaire_results, itembased_results):
    """
    Create comparison plot for MSE scores across all models.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.set_facecolor(BG_COLOR)
    
    # Questionnaire models
    ax1.set_facecolor(BG_COLOR)
    q_models = list(questionnaire_results.keys())
    q_data = [questionnaire_results[m]['mse'] for m in q_models]  # Extract MSE values
    
    parts = ax1.violinplot(q_data, positions=range(1, len(q_models) + 1),
                          showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor(PLOT_COLOR)
        pc.set_alpha(0.7)
    
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor(TEXT_COLOR)
            vp.set_linewidth(2)
    
    ax1.set_xticks(range(1, len(q_models) + 1))
    ax1.set_xticklabels(q_models, color=TEXT_COLOR, rotation=15)
    ax1.set_ylabel('MSE (Lower is Better)', color=TEXT_COLOR, fontsize=12)
    ax1.set_title('Questionnaire Models: Alignment Test', 
                 color=TEXT_COLOR, fontsize=14, fontweight='bold')
    ax1.tick_params(colors=TEXT_COLOR, which='both')
    for spine in ax1.spines.values():
        spine.set_edgecolor(TEXT_COLOR)
    ax1.grid(True, alpha=0.2, color=TEXT_COLOR)
    
    # Item-based models
    ax2.set_facecolor(BG_COLOR)
    i_models = list(itembased_results.keys())
    i_data = [itembased_results[m]['mse'] for m in i_models]  # Extract MSE values
    
    parts = ax2.violinplot(i_data, positions=range(1, len(i_models) + 1),
                          showmeans=True, showmedians=True)
    
    for pc in parts['bodies']:
        pc.set_facecolor(PLOT_COLOR)
        pc.set_alpha(0.7)
    
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor(TEXT_COLOR)
            vp.set_linewidth(2)
    
    ax2.set_xticks(range(1, len(i_models) + 1))
    ax2.set_xticklabels(i_models, color=TEXT_COLOR, rotation=15)
    ax2.set_ylabel('MSE (Lower is Better)', color=TEXT_COLOR, fontsize=12)
    ax2.set_title('Item-Based Models: Alignment Test', 
                 color=TEXT_COLOR, fontsize=14, fontweight='bold')
    ax2.tick_params(colors=TEXT_COLOR, which='both')
    for spine in ax2.spines.values():
        spine.set_edgecolor(TEXT_COLOR)
    ax2.grid(True, alpha=0.2, color=TEXT_COLOR)
    
    plt.tight_layout()
    plt.savefig('evaluation_outputs/visualizations/mse_comparison.png', facecolor=BG_COLOR, dpi=150)
    print("  Saved: evaluation_outputs/visualizations/mse_comparison.png")
    
    return fig

def run_full_evaluation(n_users=150):
    """
    Run comprehensive evaluation on all models.
    """
    print("="*80)
    print("SYNTHETIC EVALUATION FRAMEWORK - ALL MODELS")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()
    
    # Setup
    loader = DataLoader()
    df = loader.load_data()
    
    all_models = {
        "Heuristic": HeuristicRecommender(df),
        "Random Forest": RandomForestRecommender(df),
        "KNN": KNNRecommender(df),
        "Content-Based": ContentBasedRecommender(df),
        "Random": RandomRecommender(df)
    }
    
    # Test 1: Alignment (MSE) - Questionnaire Models
    print("\n[1/4] ALIGNMENT TEST - QUESTIONNAIRE MODELS")
    print("-"*80)
    q_models = {k: v for k, v in all_models.items() if k in ["Heuristic", "Random Forest", "Random"]}
    q_alignment = alignment_test_questionnaire(df, q_models, n_users)
    
    print(f"\nQuestionnaire Model Results:")
    for model_name, metrics in q_alignment.items():
        if metrics['mse']:
            print(f"  {model_name:15s} MSE: {np.mean(metrics['mse']):.4f} | "
                  f"P@3: {np.mean(metrics['precision']):.3f} | "
                  f"R@3: {np.mean(metrics['recall']):.3f} | "
                  f"MAP@3: {np.mean(metrics['map']):.3f} | "
                  f"NDCG@3: {np.mean(metrics['ndcg']):.3f}")
    
    # Test 2: Alignment (MSE) - Item-Based Models
    print("\n[2/4] ALIGNMENT TEST - ITEM-BASED MODELS")
    print("-"*80)
    i_models = {k: v for k, v in all_models.items() if k in ["KNN", "Content-Based", "Random"]}
    i_alignment = alignment_test_itembased(df, i_models, n_users, k=3)
    
    print(f"\nItem-Based Model Results:")
    for model_name, metrics in i_alignment.items():
        if metrics['mse']:
            print(f"  {model_name:15s} MSE: {np.mean(metrics['mse']):.4f} | "
                  f"P@3: {np.mean(metrics['precision']):.3f} | "
                  f"R@3: {np.mean(metrics['recall']):.3f} | "
                  f"MAP@3: {np.mean(metrics['map']):.3f} | "
                  f"NDCG@3: {np.mean(metrics['ndcg']):.3f}")
    
    # Test 3: Coverage
    print("\n[3/4] CATALOG COVERAGE TEST")
    print("-"*80)
    coverage_results = coverage_test_all_models(df, all_models, n_users)
    
    print(f"\nCoverage Results:")
    for model_name, cov_metrics in coverage_results.items():
        print(f"  {model_name:15s} {cov_metrics['coverage']}/{df.height} ({cov_metrics['pct']:.1f}%)")
    
    # Test 4: 3D UMAP Visualization
    print("\n[4/4] 3D UMAP VISUALIZATION")
    print("-"*80)
    umap_visualization_3d(df, all_models["Content-Based"])
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    plot_mse_comparison_all(q_alignment, i_alignment)
    plot_metrics_comparison(q_alignment, i_alignment)
    plot_coverage_comparison(coverage_results)
    
    # Write summary to file
    os.makedirs('evaluation_outputs/summaries', exist_ok=True)
    with open('evaluation_outputs/summaries/synthetic_evaluation_summary.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("SYNTHETIC EVALUATION SUMMARY - ALL MODELS\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Synthetic Users: {n_users}\n")
        f.write(f"Evaluation at k=3 recommendations\n")
        f.write("="*80 + "\n\n")
        
        f.write("QUESTIONNAIRE MODELS:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<15} {'MSE':<8} {'P@3':<8} {'R@3':<8} {'MAP@3':<8} {'NDCG@3':<8}\n")
        f.write("-"*80 + "\n")
        for model_name, metrics in q_alignment.items():
            if metrics['mse']:
                f.write(f"{model_name:<15} "
                       f"{np.mean(metrics['mse']):<8.4f} "
                       f"{np.mean(metrics['precision']):<8.3f} "
                       f"{np.mean(metrics['recall']):<8.3f} "
                       f"{np.mean(metrics['map']):<8.3f} "
                       f"{np.mean(metrics['ndcg']):<8.3f}\n")
        
        f.write("\nITEM-BASED MODELS:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<15} {'MSE':<8} {'P@3':<8} {'R@3':<8} {'MAP@3':<8} {'NDCG@3':<8}\n")
        f.write("-"*80 + "\n")
        for model_name, metrics in i_alignment.items():
            if metrics['mse']:
                f.write(f"{model_name:<15} "
                       f"{np.mean(metrics['mse']):<8.4f} "
                       f"{np.mean(metrics['precision']):<8.3f} "
                       f"{np.mean(metrics['recall']):<8.3f} "
                       f"{np.mean(metrics['map']):<8.3f} "
                       f"{np.mean(metrics['ndcg']):<8.3f}\n")
        
        f.write("\nCATALOG COVERAGE:\n")
        f.write("-"*80 + "\n")
        for model_name, cov_metrics in coverage_results.items():
            f.write(f"{model_name:<15} {cov_metrics['coverage']}/{df.height} ({cov_metrics['pct']:.1f}%)\n")
        
        # Find best models
        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("-"*80 + "\n")
        
        # Best by different metrics
        if q_alignment:
            best_q_prec = max(q_alignment.items(), key=lambda x: np.mean(x[1]['precision']) if x[1]['precision'] else 0)
            best_q_ndcg = max(q_alignment.items(), key=lambda x: np.mean(x[1]['ndcg']) if x[1]['ndcg'] else 0)
            f.write(f"Best Questionnaire Model (Precision@3): {best_q_prec[0]} ({np.mean(best_q_prec[1]['precision']):.3f})\n")
            f.write(f"Best Questionnaire Model (NDCG@3):      {best_q_ndcg[0]} ({np.mean(best_q_ndcg[1]['ndcg']):.3f})\n")
        
        if i_alignment:
            best_i_prec = max(i_alignment.items(), key=lambda x: np.mean(x[1]['precision']) if x[1]['precision'] else 0)
            best_i_ndcg = max(i_alignment.items(), key=lambda x: np.mean(x[1]['ndcg']) if x[1]['ndcg'] else 0)
            f.write(f"Best Item-Based Model (Precision@3):    {best_i_prec[0]} ({np.mean(best_i_prec[1]['precision']):.3f})\n")
            f.write(f"Best Item-Based Model (NDCG@3):         {best_i_ndcg[0]} ({np.mean(best_i_ndcg[1]['ndcg']):.3f})\n")
        
        # Best coverage
        best_cov = max(coverage_results.items(), key=lambda x: x[1]['pct'])
        f.write(f"Best Coverage:                           {best_cov[0]} ({best_cov[1]['pct']:.1f}%)\n")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - evaluation_outputs/visualizations/umap_3d_visualization.html (interactive 3D)")
    print("  - evaluation_outputs/visualizations/mse_comparison.png")
    print("  - evaluation_outputs/visualizations/metrics_comparison_bars.png")
    print("  - evaluation_outputs/visualizations/coverage_comparison_bar.png")
    print("  - evaluation_outputs/summaries/synthetic_evaluation_summary.txt")

if __name__ == "__main__":
    run_full_evaluation(n_users=150)
