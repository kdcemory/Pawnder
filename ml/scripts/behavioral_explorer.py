import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Import the behavioral parser
from behavioral_parser import BehavioralCategoriesParser

class BehavioralDataExplorer:
    def __init__(self, excel_path):
        """
        Initialize the behavioral data explorer
        
        Args:
            excel_path (str): Path to the Condensed Behavioral Categories Excel file
        """
        self.excel_path = excel_path
        self.parser = BehavioralCategoriesParser(excel_path)
        
    def display_emotional_categories(self):
        """Display all emotional categories in the dataset"""
        emotions = self.parser.get_emotional_categories()
        
        print(f"Found {len(emotions)} unique emotional states:")
        for i, emotion in enumerate(sorted(emotions), 1):
            safety = self.parser.get_safety_level_for_emotion(emotion)
            friendliness = self.parser.get_friendliness_score_for_emotion(emotion)
            print(f"{i}. {emotion} (Safety: {safety}, Friendliness Score: {friendliness})")
    
    def display_behavioral_indicators(self):
        """Display all behavioral indicators grouped by body part"""
        indicators_by_part = self.parser.get_indicators_by_body_part()
        
        print(f"Found {sum(len(behaviors) for behaviors in indicators_by_part.values())} behavioral indicators across {len(indicators_by_part)} body parts:")
        
        for body_part, behaviors in indicators_by_part.items():
            print(f"\n{body_part} ({len(behaviors)}):")
            for behavior in behaviors:
                print(f"  - {behavior}")
    
    def analyze_emotion_behavior_relationship(self):
        """Analyze the relationship between emotions and behaviors"""
        emotions = self.parser.get_emotional_categories()
        
        print(f"Analyzing relationship between {len(emotions)} emotions and behaviors")
        
        for emotion in emotions:
            behaviors = self.parser.get_behavioral_indicators_for_emotion(emotion)
            safety = self.parser.get_safety_level_for_emotion(emotion)
            
            print(f"\n{emotion} (Safety Level: {safety}):")
            
            # Group behaviors by body part for more organized output
            behaviors_by_part = {}
            for behavior in behaviors:
                # Find which body part this behavior belongs to
                for part, part_behaviors in self.parser.physical_behaviors.items():
                    if behavior in part_behaviors:
                        if part not in behaviors_by_part:
                            behaviors_by_part[part] = []
                        behaviors_by_part[part].append(behavior)
                        break
            
            # Print behaviors by body part
            for part, part_behaviors in behaviors_by_part.items():
                print(f"  {part}:")
                for behavior in part_behaviors:
                    print(f"    - {behavior}")
    
    def visualize_emotion_distribution(self):
        """Visualize the distribution of emotional states by safety category"""
        emotions = self.parser.get_emotional_categories()
        safety_categories = self.parser.safety_categories
        
        # Create a dictionary to store emotion counts by safety category
        emotion_counts = {category: 0 for category in safety_categories}
        
        # Count emotions per safety category
        for emotion in emotions:
            safety = self.parser.get_safety_level_for_emotion(emotion)
            if safety in emotion_counts:
                emotion_counts[safety] += 1
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame({
            'Safety Category': list(emotion_counts.keys()),
            'Emotion Count': list(emotion_counts.values())
        })
        
        # Create plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df, x='Safety Category', y='Emotion Count')
        
        # Add data labels
        for i, v in enumerate(df['Emotion Count']):
            ax.text(i, v + 0.2, str(v), ha='center')
        
        plt.title('Distribution of Emotions by Safety Category')
        plt.xlabel('Safety Category')
        plt.ylabel('Number of Emotions')
        plt.tight_layout()
        plt.savefig('emotion_distribution_by_safety.png')
        plt.close()
        
        print(f"Saved visualization to emotion_distribution_by_safety.png")
        
        # Now visualize the friendliness scale distribution
        friendliness_scores = {}
        for emotion in emotions:
            score = self.parser.get_friendliness_score_for_emotion(emotion)
            if score is not None:
                if score not in friendliness_scores:
                    friendliness_scores[score] = []
                friendliness_scores[score].append(emotion)
        
        # Create DataFrame for friendliness distribution
        df_friendliness = pd.DataFrame({
            'Friendliness Score': list(friendliness_scores.keys()),
            'Emotion Count': [len(friendliness_scores[score]) for score in friendliness_scores.keys()]
        })
        
        # Sort by friendliness score
        df_friendliness = df_friendliness.sort_values('Friendliness Score')
        
        # Create plot
        plt.figure(figsize=(14, 6))
        ax = sns.barplot(data=df_friendliness, x='Friendliness Score', y='Emotion Count')
        
        # Add data labels
        for i, v in enumerate(df_friendliness['Emotion Count']):
            ax.text(i, v + 0.1, str(v), ha='center')
        
        plt.title('Distribution of Emotions by Friendliness Score')
        plt.xlabel('Friendliness Score (Lower = Safer)')
        plt.ylabel('Number of Emotions')
        plt.tight_layout()
        plt.savefig('emotion_distribution_by_friendliness.png')
        plt.close()
        
        print(f"Saved visualization to emotion_distribution_by_friendliness.png")
    
    def create_emotion_behavior_heatmap(self):
        """Create a heatmap showing emotion-behavior relationships"""
        # Use the parser's visualization method
        self.parser.visualize_emotion_behavior_map('emotion_behavior_heatmap.png')
        print(f"Saved emotion-behavior heatmap to emotion_behavior_heatmap.png")
    
    def analyze_behavior_distribution(self):
        """Analyze the distribution of behaviors across body parts"""
        # Get behaviors by body part
        behaviors_by_part = self.parser.get_indicators_by_body_part()
        
        # Count behaviors per body part
        behavior_counts = {part: len(behaviors) for part, behaviors in behaviors_by_part.items()}
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Body Part': list(behavior_counts.keys()),
            'Behavior Count': list(behavior_counts.values())
        })
        
        # Sort by behavior count
        df = df.sort_values('Behavior Count', ascending=False)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=df, x='Body Part', y='Behavior Count')
        
        # Add data labels
        for i, v in enumerate(df['Behavior Count']):
            ax.text(i, v + 0.2, str(v), ha='center')
        
        plt.title('Distribution of Behaviors by Body Part')
        plt.xlabel('Body Part')
        plt.ylabel('Number of Behaviors')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('behavior_distribution_by_body_part.png')
        plt.close()
        
        print(f"Saved visualization to behavior_distribution_by_body_part.png")
        
        # Analyze how many emotions each behavior is associated with
        behavior_emotion_counts = {}
        
        for behavior, indicators in self.parser.behavior_matrix.items():
            # Count how many emotions this behavior is associated with
            associated_emotions = [indicator for indicator, value in indicators.items() if value == 1]
            behavior_emotion_counts[behavior] = len(associated_emotions)
        
        # Get top 20 behaviors with most associated emotions
        top_behaviors = sorted(behavior_emotion_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        
        # Create DataFrame for plotting
        df_top = pd.DataFrame({
            'Behavior': [b[0] for b in top_behaviors],
            'Associated Emotions': [b[1] for b in top_behaviors]
        })
        
        # Create plot
        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=df_top, x='Associated Emotions', y='Behavior')
        
        plt.title('Top 20 Behaviors by Number of Associated Emotions')
        plt.xlabel('Number of Associated Emotions')
        plt.ylabel('Behavior')
        plt.tight_layout()
        plt.savefig('top_behaviors_by_emotion_count.png')
        plt.close()
        
        print(f"Saved visualization to top_behaviors_by_emotion_count.png")
    
    def export_analysis_to_html(self, output_path="behavior_analysis.html"):
        """
        Export the analysis results to an HTML report
        
        Args:
            output_path (str): Path to save the HTML report
        """
        emotions = self.parser.get_emotional_categories()
        behaviors_by_part = self.parser.get_indicators_by_body_part()
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dog Behavior Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; margin-top: 30px; }
                h3 { color: #2980b9; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                tr:hover { background-color: #f5f5f5; }
                .safety-safe { background-color: #dff0d8; }
                .safety-supervised { background-color: #fcf8e3; }
                .safety-caution { background-color: #f2dede; }
                .safety-concerning { background-color: #f2b8b8; }
                .safety-danger { background-color: #e74c3c; color: white; }
                .container { display: flex; flex-wrap: wrap; }
                .image { margin: 10px; }
            </style>
        </head>
        <body>
            <h1>Dog Behavior Analysis Report</h1>
            <p>Analysis of the Condensed Behavioral Categories dataset</p>
            
            <h2>1. Emotional Categories</h2>
            <p>The dataset contains ${len(emotions)} distinct emotional states categorized by safety level.</p>
            
            <table>
                <tr>
                    <th>Emotion</th>
                    <th>Safety Category</th>
                    <th>Friendliness Score</th>
                    <th>Associated Behaviors</th>
                </tr>
        """
        
        # Add emotion data
        for emotion in sorted(emotions):
            safety = self.parser.get_safety_level_for_emotion(emotion)
            safety_class = "safety-" + safety.lower().replace(" ", "-")
            friendliness = self.parser.get_friendliness_score_for_emotion(emotion)
            behaviors = self.parser.get_behavioral_indicators_for_emotion(emotion)
            
            html += f"""
                <tr class="{safety_class}">
                    <td>{emotion}</td>
                    <td>{safety}</td>
                    <td>{friendliness}</td>
                    <td>{len(behaviors)}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>2. Behavioral Indicators</h2>
            <p>Behaviors are organized by body part.</p>
        """
        
        # Add behavior data
        for part, behaviors in behaviors_by_part.items():
            html += f"""
                <h3>{part} ({len(behaviors)})</h3>
                <ul>
            """
            
            for behavior in sorted(behaviors):
                html += f"<li>{behavior}</li>\n"
            
            html += "</ul>\n"
        
        html += """
            <h2>3. Visualizations</h2>
            <div class="container">
                <div class="image">
                    <h3>Emotion-Behavior Heatmap</h3>
                    <img src="emotion_behavior_heatmap.png" alt="Emotion-Behavior Heatmap" width="600">
                </div>
                <div class="image">
                    <h3>Emotion Distribution by Safety Category</h3>
                    <img src="emotion_distribution_by_safety.png" alt="Emotion Distribution" width="600">
                </div>
                <div class="image">
                    <h3>Emotion Distribution by Friendliness Score</h3>
                    <img src="emotion_distribution_by_friendliness.png" alt="Emotion Distribution by Friendliness" width="600">
                </div>
                <div class="image">
                    <h3>Behavior Distribution by Body Part</h3>
                    <img src="behavior_distribution_by_body_part.png" alt="Behavior Distribution" width="600">
                </div>
                <div class="image">
                    <h3>Top Behaviors by Associated Emotions</h3>
                    <img src="top_behaviors_by_emotion_count.png" alt="Top Behaviors" width="600">
                </div>
            </div>
        </body>
        </html>
        """
        
        # Replace template variables
        html = html.replace("${len(emotions)}", str(len(emotions)))
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"Exported analysis report to {output_path}")
    
    def run_analysis(self):
        """Run all analyses"""
        print("\n=== EMOTIONAL CATEGORIES ===")
        self.display_emotional_categories()
        
        print("\n=== BEHAVIORAL INDICATORS ===")
        self.display_behavioral_indicators()
        
        print("\n=== EMOTION-BEHAVIOR RELATIONSHIPS ===")
        self.analyze_emotion_behavior_relationship()
        
        print("\n=== VISUALIZING EMOTION DISTRIBUTION ===")
        self.visualize_emotion_distribution()
        
        print("\n=== CREATING EMOTION-BEHAVIOR HEATMAP ===")
        self.create_emotion_behavior_heatmap()
        
        print("\n=== ANALYZING BEHAVIOR DISTRIBUTION ===")
        self.analyze_behavior_distribution()
        
        print("\n=== EXPORTING ANALYSIS REPORT ===")
        self.export_analysis_to_html()
        
        # Export the emotion-behavior map for use by other components
        self.parser.export_emotion_behavior_map("emotion_behavior_map.json")
        print(f"Exported emotion-behavior map to emotion_behavior_map.json")

# Example usage
if __name__ == "__main__":
    # Adjust the path to your Excel file
    excel_path = "Condensed Behavioral Categories.xlsx"
    
    explorer = BehavioralDataExplorer(excel_path)
    explorer.run_analysis()
