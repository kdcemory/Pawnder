def map_emotions_to_reduced_set(df, emotion_column='emotional_state'):
    """
    Maps a larger set of emotions to a reduced set by combining similar emotions
    
    Args:
        df (pd.DataFrame): DataFrame containing emotion annotations
        emotion_column (str): Name of the column containing emotions
        
    Returns:
        pd.DataFrame: DataFrame with added column for reduced emotion set
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Define your emotion mapping (customize based on your needs)
    emotion_mapping = {
        # Map original emotions to consolidated categories
        'Happy or Playful': 'Happy/Playful',
        'Relaxed': 'Relaxed',
        'Submissive': 'Submissive/Appeasement',
        'Excited': 'Happy/Playful',
        
        'Drowsy or Bored': 'Relaxed',
        'Curious or Confused': 'Curiosity/Alertness',
        'Confident or Alert': 'Curiosity/Alertness',
        
        'Jealous': 'Stressed',
        'Stressed': 'Stressed'
        'Frustrated': 'Stressed',
        'Unsure or Uneasy': 'Fearful/Anxious',
        'Possessive, Territorial, Dominant': 'Dominant',
        'Fear or Aggression': 'Aggressive/Threatening',
        'Pain': 'Stressed'
    }
    
    # Create a new column with mapped emotions
    df['reduced_emotional_state'] = df[emotion_column].map(emotion_mapping)
    
    # Handle any emotions not in the mapping
    unknown_emotions = df[df['reduced_emotional_state'].isna()][emotion_column].unique()
    if len(unknown_emotions) > 0:
        print(f"Warning: Found unmapped emotions: {unknown_emotions}")
        # Map unknown emotions to 'Other' or a similar category
        df['reduced_emotional_state'].fillna('Other', inplace=True)
    
    return df
