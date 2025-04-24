# @title
# Behavioral Categories Data Exploration
# This script analyzes the relationship between behavioral indicators and emotional states
# Updated to work with processed CVAT annotations

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import yaml
from sklearn.metrics import confusion_matrix
from collections import Counter
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm

class BehavioralDataExplorer:
    def __init__(self, config_path="config.yaml", excel_path=None):
        """
        Initialize the behavioral data explorer

        Args:
            config_path (str): Path to configuration YAML file
            excel_path (str, optional): Path to the Condensed Behavioral Categories Excel file
                                       If None, will use path from config
        """
        self.config = self._load_config(config_path)
        self.excel_path = excel_path or os.path.join(
            self.config['data']['base_dir'],
            "Condensed Behavioral Categories.xlsx"
        )
        self.behavioral_data = self._load_behavioral_data()
        self.annotation_data = self._load_annotation_data()

    def _load_config(self, config_path):
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _load_behavioral_data(self):
        """Load data from Excel file"""
        try:
            # Load the Excel file
            df = pd.read_excel(self.excel_path)
            print(f"Successfully loaded behavioral data with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading behavioral data: {e}")
            return None

    def _load_annotation_data(self):
        """Load processed annotation data from CSV files"""
        base_dir = self.config['data']['base_dir']
        processed_dir = os.path.join(base_dir, self.config['data']['processed_data_dir'])

        annotations = {}
        # Load annotations for each split
        for split in ['train', 'val', 'test']:
            annotations_path = os.path.join(processed_dir, split, 'annotations.csv')
            if os.path.exists(annotations_path):
                try:
                    df = pd.read_csv(annotations_path)
                    annotations[split] = df
                    print(f"Loaded {split} annotations: {df.shape[0]} records")
                except Exception as e:
                    print(f"Error loading {split} annotations: {e}")

        # Combine all annotations for analysis
        if annotations:
            combined = pd.concat(annotations.values(), ignore_index=True)
            print(f"Combined annotations: {combined.shape[0]} records")
            return combined
        else:
            print("No annotation data found. Please run the CVAT processing pipeline first.")
            return None

    def display_emotional_categories(self):
        """Display all emotional categories in the dataset"""
        if self.annotation_data is None:
            print("No annotation data available")
            return None

        # Get emotional states from annotation data
        emotion_col = 'emotional_state'
        if emotion_col not in self.annotation_data.columns:
            print(f"Column '{emotion_col}' not found in annotation data")
            return None

        emotions = self.annotation_data[emotion_col].unique()
        counts = self.annotation_data[emotion_col].value_counts()

        print(f"Found {len(emotions)} unique emotional states:")
        for i, emotion in enumerate(counts.index, 1):
            print(f"{i}. {emotion}: {counts[emotion]} instances ({counts[emotion]/len(self.annotation_data)*100:.1f}%)")

        return counts

    def display_behavioral_indicators(self):
        """Display all behavioral indicators from the annotation data"""
        if self.annotation_data is None:
            print("No annotation data available")
            return

        # Look for columns that are behavior indicators (starting with 'behavior_')
        behavior_cols = [col for col in self.annotation_data.columns if col.startswith('behavior_')]

        if not behavior_cols:
            print("No behavioral indicator columns found in annotation data")
            return

        print(f"Found {len(behavior_cols)} behavioral indicator columns:")

        # Calculate frequency of each behavior
        behavior_counts = {}
        for col in behavior_cols:
            behavior_name = col.replace('behavior_', '')
            count = self.annotation_data[col].sum()
            behavior_counts[behavior_name] = count

        # Sort and display
        sorted_behaviors = sorted(behavior_counts.items(), key=lambda x: x[1], reverse=True)
        for behavior, count in sorted_behaviors:
            print(f"  {behavior}: {count} instances ({count/len(self.annotation_data)*100:.1f}%)")

        return behavior_counts

    def extract_safety_categories(self):
        """Extract safety categories from the behavioral data"""
        if self.behavioral_data is None:
            print("No behavioral data available")
            return None

        # Safety categories are typically in the first row
        safety_row = self.behavioral_data.iloc[0]
        safety_categories = safety_row.dropna().unique().tolist()

        # Remove any non-category values that might be in the header
        safety_categories = [cat for cat in safety_categories if cat not in
                            ['Friendliness Scale', 'Behavioral Indicators']]

        print(f"Found {len(safety_categories)} safety categories:")
        for i, category in enumerate(safety_categories, 1):
            print(f"{i}. {category}")

        return safety_categories

    def analyze_emotion_behavior_relationship(self):
        """Analyze the relationship between emotions and behaviors"""
        if self.annotation_data is None:
            print("No annotation data available")
            return

        emotion_col = 'emotional_state'
        behavior_cols = [col for col in self.annotation_data.columns if col.startswith('behavior_')]

        if not behavior_cols or emotion_col not in self.annotation_data.columns:
            print("Required columns not found in annotation data")
            return

        print(f"Analyzing relationship between {emotion_col} and {len(behavior_cols)} behavioral indicators")

        # Create a correlation matrix between emotions and behaviors
        # One-hot encode the emotional states for correlation
        emotions_encoded = pd.get_dummies(self.annotation_data[emotion_col])

        # Combine with behavior columns
        behavior_data = self.annotation_data[behavior_cols]
        combined_data = pd.concat([emotions_encoded, behavior_data], axis=1)

        # Calculate correlation matrix
        corr_matrix = combined_data.corr()

        # Extract correlations between emotions and behaviors
        emotion_behavior_corr = corr_matrix.loc[emotions_encoded.columns, behavior_cols]

        # Get top correlated behaviors for each emotion
        results = {}
        for emotion in emotions_encoded.columns:
            # Get correlations for this emotion
            correlations = emotion_behavior_corr.loc[emotion].sort_values(ascending=False)

            # Get top positively correlated behaviors
            top_positive = correlations[correlations > 0.2].head(5)

            # Get top negatively correlated behaviors
            top_negative = correlations[correlations < -0.2].head(5)

            results[emotion] = {
                'positive': top_positive.to_dict(),
                'negative': top_negative.to_dict()
            }

            print(f"\nEmotion: {emotion}")
            if not top_positive.empty:
                print("  Top positive correlations:")
                for behavior, corr in top_positive.items():
                    print(f"    {behavior.replace('behavior_', '')}: {corr:.3f}")
            else:
                print("  No strong positive correlations found")

            if not top_negative.empty:
                print("  Top negative correlations:")
                for behavior, corr in top_negative.items():
                    print(f"    {behavior.replace('behavior_', '')}: {corr:.3f}")
            else:
                print("  No strong negative correlations found")

        return results

    def visualize_emotion_distribution(self, save_path=None):
        """Visualize the distribution of emotional states"""
        if self.annotation_data is None:
            print("No annotation data available")
            return

        emotion_col = 'emotional_state'
        if emotion_col not in self.annotation_data.columns:
            print(f"Column '{emotion_col}' not found in annotation data")
            return

        plt.figure(figsize=(12, 6))
        ax = plt.subplot(111)

        # Create emotion count data
        emotion_counts = self.annotation_data[emotion_col].value_counts()

        # Sort by frequency
        emotion_counts = emotion_counts.sort_values(ascending=False)

        # Create color palette based on count
        colors = sns.color_palette("viridis", len(emotion_counts))

        # Create bar plot
        bars = sns.barplot(x=emotion_counts.index, y=emotion_counts.values, palette=colors, ax=ax)

        # Add count labels on top of bars
        for i, (emotion, count) in enumerate(emotion_counts.items()):
            ax.text(i, count + 5, str(count), ha='center')

        # Customize plot
        plt.title('Distribution of Emotional States', fontsize=16)
        plt.xlabel('Emotional State', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, ha='right')

        # Add percentage labels
        total = emotion_counts.sum()
        for i, p in enumerate(ax.patches):
            percentage = f"{100 * p.get_height() / total:.1f}%"
            ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height() / 2),
                        ha = 'center', va = 'center', xytext = (0, 0),
                        textcoords = 'offset points', color='white', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

        plt.close()

    def visualize_behavior_distribution(self, top_n=15, save_path=None):
        """Visualize the distribution of behavioral indicators"""
        if self.annotation_data is None:
            print("No annotation data available")
            return

        # Get behavior columns
        behavior_cols = [col for col in self.annotation_data.columns if col.startswith('behavior_')]

        if not behavior_cols:
            print("No behavioral indicator columns found in annotation data")
            return

        # Sum the occurrences of each behavior
        behavior_counts = {}
        for col in behavior_cols:
            behavior_name = col.replace('behavior_', '')
            count = self.annotation_data[col].sum()
            behavior_counts[behavior_name] = count

        # Convert to DataFrame for easier plotting
        behavior_df = pd.DataFrame({
            'behavior': list(behavior_counts.keys()),
            'count': list(behavior_counts.values())
        })

        # Sort and take top N
        behavior_df = behavior_df.sort_values('count', ascending=False).head(top_n)

        # Create the plot
        plt.figure(figsize=(12, 8))
        ax = plt.subplot(111)

        # Create color palette
        colors = sns.color_palette("viridis", len(behavior_df))

        # Create bar plot
        sns.barplot(x='count', y='behavior', data=behavior_df, palette=colors, ax=ax)

        # Add count labels
        for i, row in behavior_df.iterrows():
            ax.text(row['count'] + 5, i, str(int(row['count'])), va='center')

        # Customize plot
        plt.title(f'Top {top_n} Behavioral Indicators', fontsize=16)
        plt.xlabel('Count', fontsize=12)
        plt.ylabel('Behavioral Indicator', fontsize=12)

        # Add percentage labels
        total = behavior_df['count'].sum()
        for i, p in enumerate(ax.patches):
            percentage = f"{100 * p.get_width() / sum(behavior_counts.values()):.1f}%"
            ax.annotate(percentage, (p.get_width() / 2, p.get_y() + p.get_height() / 2),
                        ha = 'center', va = 'center', xytext = (0, 0),
                        textcoords = 'offset points', color='white', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()

        plt.close()

    def create_emotion_behavior_heatmap(self, save_path=None):
        """Create a heatmap showing correlations between emotions and behaviors"""
        if self.annotation_data is None:
            print("No annotation data available")
            return

        emotion_col = 'emotional_state'
        behavior_cols = [col for col in self.annotation_data.columns if col.startswith('behavior_')]

        if not behavior_cols or emotion_col not in self.annotation_data.columns:
            print("Required columns not found in annotation data")
            return

        # Check if we have enough behaviors with variation
        behavior_variation = self.annotation_data[behavior_cols].nunique() > 1
        valid_behaviors = [col for i, col in enumerate(behavior_cols) if behavior_variation.iloc[i]]

        if len(valid_behaviors) < 1:
            print("Not enough behavioral indicators with variation for heatmap")
            if save_path:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, "Insufficient behavior variation for heatmap",
                         ha='center', va='center', fontsize=16)
                plt.axis('off')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            return None

        # Use only behaviors with variation
        behavior_cols = valid_behaviors

        try:
            # Prepare data for heatmap
            # Count occurrences of each behavior for each emotion
            emotion_behavior_matrix = np.zeros((
                len(self.annotation_data[emotion_col].unique()),
                len(behavior_cols)
            ))

            # Get unique emotions
            emotions = self.annotation_data[emotion_col].unique()

            # Fill the matrix
            for i, emotion in enumerate(emotions):
                emotion_data = self.annotation_data[self.annotation_data[emotion_col] == emotion]
                for j, col in enumerate(behavior_cols):
                    # Calculate percentage of this emotion showing this behavior
                    if len(emotion_data) > 0:
                        emotion_behavior_matrix[i, j] = emotion_data[col].sum() / len(emotion_data)

            # Check for NaN values
            if np.isnan(emotion_behavior_matrix).any():
                print("Warning: NaN values found in emotion-behavior matrix. Filling with zeros.")
                emotion_behavior_matrix = np.nan_to_num(emotion_behavior_matrix)

            # Create heatmap
            plt.figure(figsize=(16, 10))

            # Clean up behavior names for display
            behavior_names = [col.replace('behavior_', '') for col in behavior_cols]

            # Shorten behavior names if too long
            max_name_length = 25
            behavior_names = [name[:max_name_length] + '...' if len(name) > max_name_length else name
                             for name in behavior_names]

            # Create heatmap
            sns.heatmap(
                emotion_behavior_matrix,
                annot=False,
                cmap='viridis',
                xticklabels=behavior_names,
                yticklabels=emotions,
                vmin=0,
                vmax=1
            )

            plt.title('Emotion-Behavior Association Heatmap', fontsize=16)
            plt.xlabel('Behavioral Indicators', fontsize=12)
            plt.ylabel('Emotional States', fontsize=12)
            plt.xticks(rotation=90)
            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved heatmap to {save_path}")
            else:
                plt.show()

            plt.close()

            return emotion_behavior_matrix, emotions, behavior_names

        except Exception as e:
            print(f"Error creating emotion-behavior heatmap: {e}")
            if save_path:
                # Create a simple error message image
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, f"Error creating heatmap: {str(e)}",
                         ha='center', va='center', fontsize=12, wrap=True)
                plt.axis('off')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            return None

    def create_emotion_behavior_map(self):
        """Create a mapping between behaviors and emotions"""
        if self.annotation_data is None:
            print("No annotation data available")
            return

        emotion_col = 'emotional_state'
        behavior_cols = [col for col in self.annotation_data.columns if col.startswith('behavior_')]

        if not behavior_cols or emotion_col not in self.annotation_data.columns:
            print("Required columns not found in annotation data")
            return

        # Create mapping
        emotion_behavior_map = {}

        for behavior_col in behavior_cols:
            behavior_name = behavior_col.replace('behavior_', '')
            emotion_behavior_map[behavior_name] = {}

            # Get emotions associated with this behavior
            behavior_present = self.annotation_data[self.annotation_data[behavior_col] == 1]
            if len(behavior_present) == 0:
                continue

            # Count emotions
            emotion_counts = behavior_present[emotion_col].value_counts()

            # Calculate percentages
            for emotion, count in emotion_counts.items():
                percentage = count / len(behavior_present)
                # Only include if percentage is significant
                if percentage >= 0.05:  # At least 5%
                    emotion_behavior_map[behavior_name][emotion] = percentage

        # Save the mapping to a JSON file
        output_dir = os.path.join(self.config['data']['base_dir'], 'analysis')
        os.makedirs(output_dir, exist_ok=True)

        map_path = os.path.join(output_dir, 'emotion_behavior_map.json')
        with open(map_path, 'w') as f:
            json.dump(emotion_behavior_map, f, indent=2)

        print(f"Saved emotion-behavior mapping to {map_path}")

        return emotion_behavior_map

    def analyze_co_occurring_behaviors(self, min_correlation=0.3):
        """Analyze which behaviors tend to occur together"""
        if self.annotation_data is None:
            print("No annotation data available")
            return

        behavior_cols = [col for col in self.annotation_data.columns if col.startswith('behavior_')]

        if not behavior_cols:
            print("No behavioral indicator columns found in annotation data")
            return

        # Calculate correlation matrix for behaviors
        behavior_data = self.annotation_data[behavior_cols]
        behavior_corr = behavior_data.corr()

        # Find pairs of highly correlated behaviors
        co_occurring_pairs = []

        for i, b1 in enumerate(behavior_cols):
            for j, b2 in enumerate(behavior_cols[i+1:], i+1):
                correlation = behavior_corr.loc[b1, b2]
                if correlation >= min_correlation:
                    co_occurring_pairs.append((
                        b1.replace('behavior_', ''),
                        b2.replace('behavior_', ''),
                        correlation
                    ))

        # Sort by correlation
        co_occurring_pairs.sort(key=lambda x: x[2], reverse=True)

        # Print results
        print(f"Found {len(co_occurring_pairs)} pairs of co-occurring behaviors (correlation >= {min_correlation}):")
        for b1, b2, corr in co_occurring_pairs:
            print(f"  {b1} + {b2}: {corr:.3f}")

        return co_occurring_pairs

    def create_behavior_clusters(self, min_correlation=0.3, save_path=None):
        """Create clusters of related behaviors based on correlation"""
        if self.annotation_data is None:
            print("No annotation data available")
            return

        behavior_cols = [col for col in self.annotation_data.columns if col.startswith('behavior_')]

        if not behavior_cols:
            print("No behavioral indicator columns found in annotation data")
            return

        # Calculate correlation matrix for behaviors
        behavior_data = self.annotation_data[behavior_cols]

        # Check if we have enough variation in the data
        constant_cols = [col for col in behavior_cols if behavior_data[col].nunique() <= 1]
        if constant_cols:
            print(f"Warning: {len(constant_cols)} columns have no variation. Removing from clustering.")
            for col in constant_cols:
                print(f"  - {col.replace('behavior_', '')}")
            behavior_cols = [col for col in behavior_cols if col not in constant_cols]
            behavior_data = behavior_data[behavior_cols]

        if len(behavior_cols) < 2:
            print("Not enough variable behavioral indicators for clustering")
            # Create a simple plot instead
            if save_path:
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, "Insufficient data for clustering",
                         ha='center', va='center', fontsize=16)
                plt.axis('off')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            return None

        # Calculate correlation
        behavior_corr = behavior_data.corr()

        # Handle NaN values
        if behavior_corr.isnull().any().any():
            print("Warning: NaN values found in correlation matrix. Filling with zeros.")
            behavior_corr = behavior_corr.fillna(0)

        # Clean behavior names for visualization
        behavior_names = [col.replace('behavior_', '') for col in behavior_cols]

        try:
            # Create a clustermap
            plt.figure(figsize=(16, 14))

            # Use a regular heatmap if there are issues with clustermap
            try:
                cluster_grid = sns.clustermap(
                    behavior_corr,
                    cmap='viridis',
                    xticklabels=behavior_names,
                    yticklabels=behavior_names,
                    figsize=(16, 14)
                )

                plt.title('Behavioral Indicator Clusters', fontsize=16)
            except Exception as e:
                print(f"Error creating clustermap: {e}")
                print("Falling back to standard heatmap")
                plt.figure(figsize=(16, 14))
                sns.heatmap(
                    behavior_corr,
                    cmap='viridis',
                    xticklabels=behavior_names,
                    yticklabels=behavior_names,
                    annot=False
                )
                plt.title('Behavioral Indicator Correlations', fontsize=16)

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved behavior correlations to {save_path}")
            else:
                plt.show()

            plt.close()

            return behavior_corr

        except Exception as e:
            print(f"Error in cluster visualization: {e}")
            if save_path:
                # Create a simple error message image
                plt.figure(figsize=(8, 6))
                plt.text(0.5, 0.5, f"Error creating clusters: {str(e)}",
                         ha='center', va='center', fontsize=12, wrap=True)
                plt.axis('off')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            return None

    def analyze_emotion_prediction_power(self):
        """Analyze which behaviors are most predictive of specific emotions"""
        if self.annotation_data is None:
            print("No annotation data available")
            return

        emotion_col = 'emotional_state'
        behavior_cols = [col for col in self.annotation_data.columns if col.startswith('behavior_')]

        if not behavior_cols or emotion_col not in self.annotation_data.columns:
            print("Required columns not found in annotation data")
            return

        # Initialize results
        predictive_power = {}

        # Analyze each emotion
        for emotion in self.annotation_data[emotion_col].unique():
            # Create binary target (1 for this emotion, 0 for others)
            y = (self.annotation_data[emotion_col] == emotion).astype(int)

            # Calculate metrics for each behavior
            behavior_metrics = {}

            for col in behavior_cols:
                behavior_name = col.replace('behavior_', '')
                X = self.annotation_data[col]

                # Calculate precision and recall
                # True positives: behavior present (1) and emotion correct (1)
                tp = ((X == 1) & (y == 1)).sum()

                # False positives: behavior present (1) but emotion wrong (0)
                fp = ((X == 1) & (y == 0)).sum()

                # False negatives: behavior not present (0) but emotion is correct (1)
                fn = ((X == 0) & (y == 1)).sum()

                # Calculate metrics if possible
                if tp + fp > 0:
                    precision = tp / (tp + fp)
                else:
                    precision = 0

                if tp + fn > 0:
                    recall = tp / (tp + fn)
                else:
                    recall = 0

                # Calculate F1 score
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0

                # Store metrics
                behavior_metrics[behavior_name] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }

            # Sort behaviors by F1 score
            sorted_behaviors = sorted(
                behavior_metrics.items(),
                key=lambda x: x[1]['f1'],
                reverse=True
            )

            # Store top behaviors for this emotion
            predictive_power[emotion] = {
                'top_behaviors': sorted_behaviors[:5],
                'all_behaviors': behavior_metrics
            }

            # Print results
            print(f"\nMost predictive behaviors for '{emotion}':")
            for behavior, metrics in sorted_behaviors[:5]:
                print(f"  {behavior}:")
                print(f"    Precision: {metrics['precision']:.3f}")
                print(f"    Recall: {metrics['recall']:.3f}")
                print(f"    F1 Score: {metrics['f1']:.3f}")

        return predictive_power

    def analyze_emotion_co_occurrence(self):
        """Analyze which emotions commonly occur together in multi-label scenarios"""
        if self.annotation_data is None:
            print("No annotation data available")
            return

        # Currently, our data model likely only has one emotion per annotation
        # This is a placeholder for future enhancements if we support multiple emotions
        print("Multiple emotions per annotation not currently supported in the data model.")

    def generate_emotion_profiles(self, output_dir=None):
        """Generate comprehensive profiles for each emotion"""
        if self.annotation_data is None:
            print("No annotation data available")
            return

        emotion_col = 'emotional_state'
        behavior_cols = [col for col in self.annotation_data.columns if col.startswith('behavior_')]

        if not behavior_cols or emotion_col not in self.annotation_data.columns:
            print("Required columns not found in annotation data")
            return

        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(self.config['data']['base_dir'], 'analysis', 'emotion_profiles')
        os.makedirs(output_dir, exist_ok=True)

        # Get unique emotions
        emotions = self.annotation_data[emotion_col].unique()

        # Create profiles for each emotion
        emotion_profiles = {}

        for emotion in emotions:
            # Get data for this emotion
            emotion_data = self.annotation_data[self.annotation_data[emotion_col] == emotion]

            # Count total instances
            count = len(emotion_data)

            # Count behavior occurrences
            behavior_counts = {}
            for col in behavior_cols:
                behavior_name = col.replace('behavior_', '')
                behavior_count = emotion_data[col].sum()
                if behavior_count > 0:
                    behavior_counts[behavior_name] = {
                        'count': int(behavior_count),
                        'percentage': float(behavior_count / count)
                    }

            # Sort behaviors by percentage
            sorted_behaviors = {k: v for k, v in sorted(
                behavior_counts.items(),
                key=lambda item: item[1]['percentage'],
                reverse=True
            )}

            # Create profile
            profile = {
                'emotion': emotion,
                'count': int(count),
                'percentage_of_dataset': float(count / len(self.annotation_data)),
                'behaviors': sorted_behaviors
            }

            emotion_profiles[emotion] = profile

            # Save individual profile
            profile_path = os.path.join(output_dir, f"{emotion.lower().replace(' ', '_')}_profile.json")
            with open(profile_path, 'w') as f:
                json.dump(profile, f, indent=2)

            # Print summary
            print(f"\nProfile for '{emotion}':")
            print(f"  Total instances: {count} ({profile['percentage_of_dataset']:.1%} of dataset)")
            print("  Top behaviors:")
            for behavior, stats in list(sorted_behaviors.items())[:5]:
                print(f"    {behavior}: {stats['count']} instances ({stats['percentage']:.1%})")

        # Save all profiles to one file
        all_profiles_path = os.path.join(output_dir, "all_emotion_profiles.json")
        with open(all_profiles_path, 'w') as f:
            json.dump(emotion_profiles, f, indent=2)

        print(f"\nAll emotion profiles saved to {all_profiles_path}")

        return emotion_profiles

    def create_dashboard(self, output_dir=None):
        """Create a comprehensive dashboard of visualizations"""
        if self.annotation_data is None:
            print("No annotation data available")
            return

        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(self.config['data']['base_dir'], 'analysis', 'dashboard')
        os.makedirs(output_dir, exist_ok=True)

        print("Generating visualization dashboard...")

        # 1. Emotion Distribution
        emotion_path = os.path.join(output_dir, "emotion_distribution.png")
        self.visualize_emotion_distribution(save_path=emotion_path)

        # 2. Behavior Distribution
        behavior_path = os.path.join(output_dir, "behavior_distribution.png")
        self.visualize_behavior_distribution(save_path=behavior_path)

        # 3. Emotion-Behavior Heatmap
        heatmap_path = os.path.join(output_dir, "emotion_behavior_heatmap.png")
        self.create_emotion_behavior_heatmap(save_path=heatmap_path)

        # 4. Behavior Clusters
        clusters_path = os.path.join(output_dir, "behavior_clusters.png")
        self.create_behavior_clusters(save_path=clusters_path)

        # 5. Generate emotion profiles
        profiles_dir = os.path.join(output_dir, "emotion_profiles")
        self.generate_emotion_profiles(output_dir=profiles_dir)

        # 6. Create emotion-behavior map
        self.create_emotion_behavior_map()

        # 7. Create dashboard HTML
        dashboard_html = self._create_dashboard_html(output_dir)

        print(f"Dashboard created in {output_dir}")
        print(f"Open {os.path.join(output_dir, 'dashboard.html')} to view the dashboard")

        return output_dir

    def _create_dashboard_html(self, output_dir):
        """Create an HTML dashboard to view all visualizations"""
        # Basic HTML template
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Dog Emotion Project - Data Analysis Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2 {
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .card {
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            padding: 20px;
        }
        .visualization {
            text-align: center;
            margin: 20px 0;
        }
        .visualization img {
            max-width: 100%;
            height: auto;
        }
        .col-2 {
            display: flex;
            flex-wrap: wrap;
        }
        .col-2 .card {
            flex: 1 1 45%;
            margin: 10px;
            min-width: 300px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dog Emotion Project - Data Analysis Dashboard</h1>

        <div class="card">
            <h2>Dataset Overview</h2>
            <p>Total annotations: {total_annotations}</p>
            <p>Unique emotions: {unique_emotions}</p>
            <p>Behavioral indicators: {behavior_count}</p>
        </div>

        <div class="card">
            <h2>Emotion Distribution</h2>
            <div class="visualization">
                <img src="emotion_distribution.png" alt="Emotion Distribution">
            </div>
        </div>

        <div class="card">
            <h2>Behavioral Indicators Distribution</h2>
            <div class="visualization">
                <img src="behavior_distribution.png" alt="Behavior Distribution">
            </div>
        </div>

        <div class="card">
            <h2>Emotion-Behavior Relationship</h2>
            <div class="visualization">
                <img src="emotion_behavior_heatmap.png" alt="Emotion-Behavior Heatmap">
            </div>
        </div>

        <div class="card">
            <h2>Behavioral Indicator Clusters</h2>
            <div class="visualization">
                <img src="behavior_clusters.png" alt="Behavior Clusters">
            </div>
        </div>

        <div class="card">
            <h2>Emotion Profiles</h2>
            <p>Detailed emotion profiles are available in the emotion_profiles directory.</p>
        </div>
    </div>
</body>
</html>"""

        # Fill in the template with actual data
        total_annotations = len(self.annotation_data) if self.annotation_data is not None else 0

        emotion_col = 'emotional_state'
        unique_emotions = len(self.annotation_data[emotion_col].unique()) if self.annotation_data is not None and emotion_col in self.annotation_data.columns else 0

        behavior_cols = [col for col in self.annotation_data.columns if col.startswith('behavior_')] if self.annotation_data is not None else []
        behavior_count = len(behavior_cols)

        html = html.format(
            total_annotations=total_annotations,
            unique_emotions=unique_emotions,
            behavior_count=behavior_count
        )

        # Write to file
        dashboard_path = os.path.join(output_dir, 'dashboard.html')
        with open(dashboard_path, 'w') as f:
            f.write(html)

        return dashboard_path

    def run_comprehensive_analysis(self, output_dir=None):
        """Run all analyses and create a complete report"""
        if output_dir is None:
            output_dir = os.path.join(self.config['data']['base_dir'], 'analysis')
        os.makedirs(output_dir, exist_ok=True)

        print("Starting comprehensive analysis of dog emotion data...")

        # Check if we have annotation data
        if self.annotation_data is None or len(self.annotation_data) == 0:
            print("ERROR: No annotation data available for analysis.")
            print("Please run the CVAT Annotation Processing Pipeline first to generate annotation data.")
            return None

        try:
            # 1. Display dataset information
            print("\n=== DATASET OVERVIEW ===")
            print(f"Total annotations: {len(self.annotation_data)}")

            # Split counts
            split_counts = {}
            for split in ['train', 'val', 'test']:
                annotations_path = os.path.join(
                    self.config['data']['base_dir'],
                    self.config['data']['processed_data_dir'],
                    split,
                    'annotations.csv'
                )
                if os.path.exists(annotations_path):
                    df = pd.read_csv(annotations_path)
                    split_counts[split] = len(df)

            if split_counts:
                print("Split distribution:")
                for split, count in split_counts.items():
                    print(f"  {split}: {count} annotations ({count/len(self.annotation_data)*100:.1f}%)")

            # 2. Analyze emotional categories
            print("\n=== EMOTIONAL CATEGORIES ===")
            try:
                emotion_counts = self.display_emotional_categories()
            except Exception as e:
                print(f"Error analyzing emotional categories: {e}")
                emotion_counts = None

            # 3. Analyze behavioral indicators
            print("\n=== BEHAVIORAL INDICATORS ===")
            try:
                behavior_counts = self.display_behavioral_indicators()
            except Exception as e:
                print(f"Error analyzing behavioral indicators: {e}")
                behavior_counts = None

            # 4. Extract safety categories (if available)
            print("\n=== SAFETY CATEGORIES ===")
            try:
                safety_categories = self.extract_safety_categories()
            except Exception as e:
                print(f"Error extracting safety categories: {e}")
                safety_categories = None

            # 5. Analyze emotion-behavior relationships
            print("\n=== EMOTION-BEHAVIOR RELATIONSHIPS ===")
            try:
                emotion_behavior_results = self.analyze_emotion_behavior_relationship()
            except Exception as e:
                print(f"Error analyzing emotion-behavior relationships: {e}")
                emotion_behavior_results = None

            # 6. Analyze co-occurring behaviors
            print("\n=== CO-OCCURRING BEHAVIORS ===")
            try:
                co_occurring_pairs = self.analyze_co_occurring_behaviors()
            except Exception as e:
                print(f"Error analyzing co-occurring behaviors: {e}")
                co_occurring_pairs = None

            # 7. Analyze emotion prediction power
            print("\n=== BEHAVIOR PREDICTIVE POWER ===")
            try:
                predictive_power = self.analyze_emotion_prediction_power()
            except Exception as e:
                print(f"Error analyzing behavior predictive power: {e}")
                predictive_power = None

            # 8. Create dashboard
            print("\n=== GENERATING VISUAL DASHBOARD ===")
            try:
                dashboard_dir = self.create_dashboard(output_dir=os.path.join(output_dir, 'dashboard'))
            except Exception as e:
                print(f"Error creating dashboard: {e}")
                dashboard_dir = None

            # 9. Save all results to a comprehensive report
            summary_data = {
                'dataset_overview': {
                    'total_annotations': len(self.annotation_data),
                    'split_counts': split_counts if 'split_counts' in locals() else {},
                    'emotion_counts': emotion_counts.to_dict() if emotion_counts is not None else {},
                    'behavior_counts': behavior_counts if behavior_counts is not None else {}
                },
                'safety_categories': safety_categories if safety_categories is not None else [],
                'co_occurring_behaviors': co_occurring_pairs if co_occurring_pairs is not None else []
            }

            # Save summary report
            summary_path = os.path.join(output_dir, 'analysis_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=2)

            print(f"\nAnalysis complete! Summary saved to {summary_path}")
            if dashboard_dir:
                print(f"Interactive dashboard available at {os.path.join(dashboard_dir, 'dashboard.html')}")

            return summary_data

        except Exception as e:
            print(f"Error during comprehensive analysis: {e}")
            import traceback
            traceback.print_exc()

            # Create basic error report
            error_path = os.path.join(output_dir, 'analysis_error.txt')
            with open(error_path, 'w') as f:
                f.write(f"Error during analysis: {str(e)}\n\n")
                traceback.print_exc(file=f)

            print(f"\nError recorded to {error_path}")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize explorer with config path
    explorer = BehavioralDataExplorer(config_path="config.yaml")

    # Run comprehensive analysis
    explorer.run_comprehensive_analysis()
