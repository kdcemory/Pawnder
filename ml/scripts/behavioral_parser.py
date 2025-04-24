import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class BehavioralCategoriesParser:
    """
    Parser for the Condensed Behavioral Categories Excel file
    with the specific structure outlined in the documentation.
    """
    
    def __init__(self, excel_path):
        """
        Initialize the parser
        
        Args:
            excel_path (str): Path to the Condensed Behavioral Categories Excel file
        """
        self.excel_path = excel_path
        self.data = self._load_data()
        
        if self.data is not None:
            # Parse the structure
            self.safety_categories = self._extract_safety_categories()
            self.friendliness_scale = self._extract_friendliness_scale()
            self.behavioral_indicators = self._extract_behavioral_indicators()
            self.physical_behaviors = self._extract_physical_behaviors()
            self.indicator_to_category = self._map_indicators_to_categories()
            self.behavior_matrix = self._extract_behavior_matrix()
    
    def _load_data(self):
        """Load data from Excel file"""
        try:
            # Load the Excel file without interpreting headers
            df = pd.read_excel(self.excel_path, header=None)
            print(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def _extract_safety_categories(self):
        """Extract safety categories from row 1"""
        if self.data is None:
            return []
        
        # Get row 1, starting from column B (index 1)
        categories_row = self.data.iloc[0, 1:].dropna()
        
        # Convert to list and remove duplicates caused by merged cells
        categories = []
        for cat in categories_row:
            if cat not in categories:
                categories.append(cat)
        
        print(f"Extracted {len(categories)} safety categories: {categories}")
        return categories
    
    def _extract_friendliness_scale(self):
        """Extract friendliness scale from row 3"""
        if self.data is None:
            return []
        
        # Get row 3, starting from column B (index 1)
        scale_row = self.data.iloc[2, 1:]
        
        # Convert to list and filter out NaN values
        scale = [int(x) for x in scale_row.dropna().tolist() if not pd.isna(x)]
        
        print(f"Extracted friendliness scale with {len(scale)} values: {scale}")
        return scale
    
    def _extract_behavioral_indicators(self):
        """Extract behavioral indicators from row 4"""
        if self.data is None:
            return []
        
        # Get row 4, starting from column B (index 1)
        indicators_row = self.data.iloc[3, 1:]
        
        # Convert to list and filter out NaN values
        indicators = [str(x) for x in indicators_row.dropna().tolist()]
        
        print(f"Extracted {len(indicators)} behavioral indicators")
        return indicators
    
    def _extract_physical_behaviors(self):
        """Extract physical behavior categories and subcategories from column A"""
        if self.data is None:
            return {}
        
        # Start from row 5 (index 4) for behaviors
        behaviors_col = self.data.iloc[4:, 0]
        
        # Parse into categories and behaviors
        behaviors = {}
        current_category = None
        
        for item in behaviors_col:
            if pd.isna(item):
                continue
                
            # Check if this is a category header (has NaN in the next column)
            is_category = pd.isna(self.data.iloc[behaviors_col[behaviors_col == item].index[0], 1])
            
            if is_category:
                current_category = item
                behaviors[current_category] = []
            elif current_category is not None:
                behaviors[current_category].append(item)
        
        # Count total behaviors
        total_behaviors = sum(len(behaviors[cat]) for cat in behaviors)
        print(f"Extracted {len(behaviors)} physical behavior categories with {total_behaviors} total behaviors")
        
        return behaviors
    
    def _map_indicators_to_categories(self):
        """Map behavioral indicators to safety categories"""
        if self.data is None:
            return {}
        
        indicator_map = {}
        safety_indices = {}
        
        # Find the column indices where each safety category appears in row 1
        for col_idx in range(1, len(self.data.columns)):
            val = self.data.iloc[0, col_idx]
            if not pd.isna(val) and val in self.safety_categories:
                safety_indices[col_idx] = val
        
        # Create a mapping of column indices to safety categories
        col_to_safety = {}
        current_category = None
        
        for col_idx in range(1, len(self.data.columns)):
            if col_idx in safety_indices:
                current_category = safety_indices[col_idx]
            if current_category and col_idx <= len(self.behavioral_indicators) + 1:
                col_to_safety[col_idx] = current_category
        
        # Map each indicator to its safety category
        for i, indicator in enumerate(self.behavioral_indicators):
            col_idx = i + 1  # +1 because indicators start at column B (index 1)
            if col_idx in col_to_safety:
                indicator_map[indicator] = col_to_safety[col_idx]
        
        print(f"Mapped {len(indicator_map)} indicators to safety categories")
        return indicator_map
    
    def _extract_behavior_matrix(self):
        """Extract the binary matrix mapping behaviors to indicators"""
        if self.data is None:
            return {}
        
        matrix = {}
        
        # Get the starting row for behavior data (after the header rows)
        start_row = 4
        
        # Iterate through each row starting from row 5
        for row_idx in range(start_row, len(self.data)):
            behavior = self.data.iloc[row_idx, 0]
            
            # Skip empty rows or category headers (which have NaN in column B)
            if pd.isna(behavior) or pd.isna(self.data.iloc[row_idx, 1]):
                continue
            
            # Get the binary vector for this behavior (columns B through N+1 where N=number of indicators)
            binary_vector = self.data.iloc[row_idx, 1:len(self.behavioral_indicators)+1].values
            
            # Create a dictionary mapping each indicator to its binary value
            behavior_indicators = {}
            for i, val in enumerate(binary_vector):
                # Check if value exists and is numeric
                if i < len(self.behavioral_indicators) and not pd.isna(val) and val in [0, 1]:
                    indicator = self.behavioral_indicators[i]
                    behavior_indicators[indicator] = int(val)
            
            matrix[behavior] = behavior_indicators
        
        print(f"Extracted behavior matrix with {len(matrix)} mappings")
        return matrix
    
    def get_emotional_categories(self):
        """
        Get all emotional categories (behavioral indicators)
        
        Returns:
            list: List of emotional categories
        """
        return self.behavioral_indicators
    
    def get_behavioral_indicators_for_emotion(self, emotion):
        """
        Get all behavioral indicators associated with a specific emotion
        
        Args:
            emotion (str): The emotion to get indicators for
            
        Returns:
            list: Behaviors associated with the emotion
        """
        if emotion not in self.behavioral_indicators:
            print(f"Warning: Emotion '{emotion}' not found in behavioral indicators")
            return []
        
        # Find all behaviors that have a value of 1 for this emotion
        behaviors = []
        for behavior, indicators in self.behavior_matrix.items():
            if emotion in indicators and indicators[emotion] == 1:
                behaviors.append(behavior)
        
        return behaviors
    
    def get_indicators_by_body_part(self):
        """
        Group behavioral indicators by body part
        
        Returns:
            dict: Dictionary mapping body parts to lists of behaviors
        """
        indicators_by_part = {}
        
        for category, behaviors in self.physical_behaviors.items():
            indicators_by_part[category] = behaviors
        
        return indicators_by_part
    
    def get_safety_level_for_emotion(self, emotion):
        """
        Get the safety level associated with an emotion
        
        Args:
            emotion (str): The emotion to get the safety level for
            
        Returns:
            str: Safety level for the emotion
        """
        if emotion in self.indicator_to_category:
            return self.indicator_to_category[emotion]
        return "Unknown"
    
    def get_friendliness_score_for_emotion(self, emotion):
        """
        Get the friendliness score for an emotion
        
        Args:
            emotion (str): The emotion to get the score for
            
        Returns:
            int: Friendliness score (1-13)
        """
        if emotion not in self.behavioral_indicators:
            return None
        
        idx = self.behavioral_indicators.index(emotion)
        if idx < len(self.friendliness_scale):
            return self.friendliness_scale[idx]
        return None
    
    def visualize_emotion_behavior_map(self, output_path=None):
        """
        Visualize the emotion-behavior mapping as a heatmap
        
        Args:
            output_path (str): Path to save the visualization
        """
        if not self.behavior_matrix:
            print("No behavior matrix available for visualization")
            return
        
        # Create a DataFrame for the heatmap
        behaviors = list(self.behavior_matrix.keys())
        df_data = []
        
        for behavior in behaviors:
            row = []
            for indicator in self.behavioral_indicators:
                if indicator in self.behavior_matrix[behavior]:
                    row.append(self.behavior_matrix[behavior][indicator])
                else:
                    row.append(0)
            df_data.append(row)
        
        heatmap_df = pd.DataFrame(df_data, index=behaviors, columns=self.behavioral_indicators)
        
        # Create the heatmap
        plt.figure(figsize=(16, 12))
        sns.heatmap(heatmap_df, cmap="YlGnBu", cbar_kws={'label': 'Association (0=None, 1=Associated)'})
        plt.title("Behavior-Emotion Associations")
        plt.ylabel("Physical Behaviors")
        plt.xlabel("Emotional States")
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def export_emotion_behavior_map(self, output_path):
        """
        Export the emotion-behavior mapping to a JSON file
        
        Args:
            output_path (str): Path to save the JSON file
        """
        # Create a mapping of emotions to behaviors
        emotion_to_behaviors = {}
        
        for emotion in self.behavioral_indicators:
            behaviors = self.get_behavioral_indicators_for_emotion(emotion)
            if behaviors:
                emotion_to_behaviors[emotion] = {
                    "behaviors": behaviors,
                    "safety_category": self.get_safety_level_for_emotion(emotion),
                    "friendliness_score": self.get_friendliness_score_for_emotion(emotion)
                }
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(emotion_to_behaviors, f, indent=2)
        
        print(f"Emotion-behavior mapping exported to {output_path}")
        
        return emotion_to_behaviors

# Example usage
if __name__ == "__main__":
    # Path to your Excel file - adjust as needed
    excel_path = /Data/Matrix/"Primary Behavior Matrix.xlsx"
    
    parser = BehavioralCategoriesParser(excel_path)
    
    # Example: Get all emotional categories
    emotions = parser.get_emotional_categories()
    print(f"\nEmotional Categories: {emotions}")
    
    # Example: Get behaviors for a specific emotion
    emotion = "Happy/Playful"  # Replace with an actual emotion from your data
    behaviors = parser.get_behavioral_indicators_for_emotion(emotion)
    print(f"\nBehaviors for {emotion}: {behaviors}")
    
    # Example: Visualize the emotion-behavior map
    parser.visualize_emotion_behavior_map("emotion_behavior_heatmap.png")
    
    # Example: Export the emotion-behavior map
    parser.export_emotion_behavior_map("emotion_behavior_map.json")
