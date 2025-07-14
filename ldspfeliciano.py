import pandas as pd
import json

# Load Shannon radii data
with open('shannon-radii.json', 'r') as f:
    shannon_radii = json.load(f)

# Load known perovskite datasets only
known_perovskites = pd.read_csv('Database_S1.1.csv')
known_non_perovskites = pd.read_csv('Database_S1.2.csv')

print("Known Perovskites:", len(known_perovskites))
print("Known Non-Perovskites:", len(known_non_perovskites))

def get_shannon_radius(element, oxidation_state, coordination):
    """
    Get Shannon radius for a given element, oxidation state, and coordination number.
    Returns ionic radius in Angstroms.
    """
    try:
        # Convert coordination number to Roman numeral format used in Shannon data
        coord_map = {
            1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 
            7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X', 11: 'XI', 12: 'XII',
            14: 'XIV'
        }
        
        coord_str = coord_map.get(coordination)
        if not coord_str:
            # Try alternative coordination numbers if exact match not found
            alternative_coords = [6, 8, 12, 4, 9]  # Common alternatives
            for alt_coord in alternative_coords:
                if alt_coord in coord_map and coord_map[alt_coord] in shannon_radii.get(element, {}).get(str(oxidation_state), {}):
                    coord_str = coord_map[alt_coord]
                    break
        
        if element in shannon_radii:
            if str(oxidation_state) in shannon_radii[element]:
                if coord_str and coord_str in shannon_radii[element][str(oxidation_state)]:
                    return shannon_radii[element][str(oxidation_state)][coord_str]['r_ionic']
                else:
                    # If exact coordination not found, try to find any available coordination
                    available_coords = shannon_radii[element][str(oxidation_state)]
                    if available_coords:
                        # Return the first available coordination
                        first_coord = list(available_coords.keys())[0]
                        return shannon_radii[element][str(oxidation_state)][first_coord]['r_ionic']
        
        return None
    except:
        return None

def add_shannon_radii_columns(df):
    """
    Add Shannon radii columns to the dataframe.
    """
    df = df.copy()
    
    # Initialize new columns
    df['A Radius'] = None
    df['B Radius'] = None
    df['B\' Radius'] = None
    df['X Radius'] = None
    for idx, row in df.iterrows():
        # A-site ion (typically 12-coordinate in perovskites)
        a_element = row['A']
        a_oxidation = row['A Oxidation State']
        df.at[idx, 'A Radius'] = get_shannon_radius(a_element, a_oxidation, 12)
        
        # B-site ion (typically 6-coordinate in perovskites)
        b_element = row['B']
        b_oxidation = row[' B Oxidation State']  # Note the space in column name
        df.at[idx, 'B Radius'] = get_shannon_radius(b_element, b_oxidation, 6)
        
        # B'-site ion (typically 6-coordinate in perovskites)
        b_prime_element = row['B\'']
        b_prime_oxidation = row['B\' Oxidation State']
        df.at[idx, 'B\' Radius'] = get_shannon_radius(b_prime_element, b_prime_oxidation, 6)
        
        # X-site ion (anion, coordination depends on structure)
        x_element = row['X']
        x_oxidation = row['X Oxidation State']
        # For anions, coordination is typically 2-6, try common values
        x_coords_to_try = [6, 4, 3, 2, 8]
        x_radius = None
        for coord in x_coords_to_try:
            x_radius = get_shannon_radius(x_element, x_oxidation, coord)
            if x_radius is not None:
                break
        df.at[idx, 'X Radius'] = x_radius
    
    return df

# Add Shannon radii to known perovskites
print("\nAdding Shannon radii to known perovskites...")
known_perovskites_with_radii = add_shannon_radii_columns(known_perovskites)

# Save the updated dataframe
known_perovskites_with_radii.to_csv('Database_S1.1_with_radii.csv', index=False)
print(f"\nSaved enhanced dataset with Shannon radii to 'Database_S1.1_with_radii.csv'")

# Add Shannon radii to known non-perovskites
print("\nAdding Shannon radii to known non-perovskites...")
known_non_perovskites_with_radii = add_shannon_radii_columns(known_non_perovskites)

# Save the updated dataframe
known_non_perovskites_with_radii.to_csv('Database_S1.2_with_radii.csv', index=False)
print(f"\nSaved enhanced dataset with Shannon radii to 'Database_S1.2_with_radii.csv'")

# Check for missing radii in both datasets
print("\n=== Known Perovskites Missing Radii ===")
missing_a = known_perovskites_with_radii['A Radius'].isna().sum()
missing_b = known_perovskites_with_radii['B Radius'].isna().sum()
missing_b_prime = known_perovskites_with_radii['B\' Radius'].isna().sum()
missing_x = known_perovskites_with_radii['X Radius'].isna().sum()

print(f"A Radius: {missing_a}/{len(known_perovskites_with_radii)}")
print(f"B Radius: {missing_b}/{len(known_perovskites_with_radii)}")
print(f"B' Radius: {missing_b_prime}/{len(known_perovskites_with_radii)}")
print(f"X Radius: {missing_x}/{len(known_perovskites_with_radii)}")

print("\n=== Known Non-Perovskites Missing Radii ===")
missing_a_np = known_non_perovskites_with_radii['A Radius'].isna().sum()
missing_b_np = known_non_perovskites_with_radii['B Radius'].isna().sum()
missing_b_prime_np = known_non_perovskites_with_radii['B\' Radius'].isna().sum()
missing_x_np = known_non_perovskites_with_radii['X Radius'].isna().sum()

print(f"A Radius: {missing_a_np}/{len(known_non_perovskites_with_radii)}")
print(f"B Radius: {missing_b_np}/{len(known_non_perovskites_with_radii)}")
print(f"B' Radius: {missing_b_prime_np}/{len(known_non_perovskites_with_radii)}")
print(f"X Radius: {missing_x_np}/{len(known_non_perovskites_with_radii)}")
