import pandas as pd

# 1. Load your data
# Using 'Main.csv' because it has the actual Language names (e.g., 'English')
nodes_df = pd.read_csv('Nodes.csv', encoding='latin1')
main_df = pd.read_csv('Countires and Languages.csv', encoding='latin1', names=['ISO Code', 'Country', 'Language'])

# Clean column names to avoid KeyErrors
nodes_df.columns = [c.strip() for c in nodes_df.columns]
main_df.columns = [c.strip() for c in main_df.columns]

# 2. Define which languages belong to which Colonizer
lang_to_source = {
    'English': 'GB',
    'French': 'FR',
    'Spanish': 'ES',
    'Portuguese': 'PT',
    'Dutch': 'NL',
    'Russian': 'RU',
    'German': 'DE',
    'Italian': 'IT'
}

# 3. Create the Network Edges
edges = []

for _, row in nodes_df.iterrows():
    target_iso = str(row['ISO Code']).strip()
    
    # Get the list of languages actually spoken in this country from Main.csv
    spoken_langs = main_df[main_df['ISO Code'] == target_iso]['Language'].tolist()
    
    # Parse the 'by who?' column for colonizers
    raw_who = str(row['by who?'])
    colonizers = [p.strip() for p in raw_who.split(',') if len(p.strip()) == 2 and p.strip().isupper()]
    
    for source_iso in colonizers:
        # Check for the Austria typo fix
        if source_iso == 'AS' and target_iso in ['PL', 'SK', 'CZ']: source_iso = 'AT'
        
        # Determine if any spoken language matches the colonizer's language
        # Example: If ruled by GB, is English in spoken_langs?
        is_linguistic = False
        for lang in spoken_langs:
            if lang_to_source.get(lang) == source_iso:
                is_linguistic = True
                break
        
        # Special case for US influence
        if source_iso == 'US' and 'English' in spoken_langs:
            is_linguistic = True
            
        edge_label = "Linguistic Transfer" if is_linguistic else "Colonial Rule Only"
        
        edges.append({
            'source': source_iso,
            'target': target_iso,
            'type': edge_label
        })

# 4. Save the Files
pd.DataFrame(edges).to_csv('linguistic_transfer_results.csv', index=False)

with open('final_language_flow.gml', 'w', encoding='utf-8') as f:
    f.write('graph [\n  directed 1\n')
    for _, row in nodes_df.iterrows():
        f.write(f'  node [\n    id "{row["ISO Code"]}"\n    label "{row["Country"]}"\n  ]\n')
    for e in edges:
        f.write(f'  edge [\n    source "{e["source"]}"\n    target "{e["target"]}"\n    label "{e["type"]}"\n  ]\n')
    f.write(']')

print("GML and CSV generated successfully without using the 'Degree' column.")