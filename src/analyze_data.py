#!/usr/bin/env python3
"""
Script to analyze the landslide probability dataset
"""
import pandas as pd
import numpy as np

def analyze_landslide_data():
    """ Analyze the landslide probability CSV data"""
    print("Loading landslide probability data...")
    df = pd.read_csv('landslide_prob.csv')
    
    print(f"\nDataset Information:")
    print(f"- Shape: {df.shape}")
    print(f"- Columns: {list(df.columns)}")
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nCoordinate Ranges:")
    print(f"- Longitude (X): {df.X.min():.6f} to {df.X.max():.6f}")
    print(f"- Latitude (Y): {df.Y.min():.6f} to {df.Y.max():.6f}")
    
    print(f"\nProbability Statistics:")
    print(f"- Range: {df.Probability.min():.6f} to {df.Probability.max():.6f}")
    print(f"- Mean: {df.Probability.mean():.6f}")
    print(f"- Median: {df.Probability.median():.6f}")
    print(f"- Std Dev: {df.Probability.std():.6f}")
    
    print(f"\nProbability Distribution:")
    print(df.Probability.describe())
    
    # Check for missing values
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    
    # Sample some high and low probability areas
    print(f"\nTop 5 highest probability areas:")
    high_prob = df.nlargest(5, 'Probability')
    print(high_prob)
    
    print(f"\nTop 5 lowest probability areas:")
    low_prob = df.nsmallest(5, 'Probability')
    print(low_prob)

if __name__ == "__main__":
    analyze_landslide_data()
