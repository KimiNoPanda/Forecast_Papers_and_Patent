'''
Merge substring clusters into the most representative stable cluster.

'''


import pandas as pd
def merge_substring_clusters(df):
    """
    For concepts where A is a substring of B, merge all into the 
    stable_cluster with the most concepts (most representative).
    """
    cluster_map  = dict(zip(df["norm_concept"], df["stable_cluster"]))
    cluster_sizes = df[df["stable_cluster"] != -1]["stable_cluster"].value_counts().to_dict()
    
    # group by first token for speed
    from collections import defaultdict
    first_token_groups = defaultdict(list)
    for c in df["norm_concept"]:
        first_token_groups[c.split()[0]].append(c)
    
    new_map = cluster_map.copy()
    reassigned = 0
    
    for first_tok, group in first_token_groups.items():
        if len(group) < 2:
            continue
        
        # find all substring relationships within group
        components = []
        used = set()
        
        for c1 in group:
            if c1 in used:
                continue
            component = {c1}
            for c2 in group:
                if c2 == c1:
                    continue
                if c1 in c2 or c2 in c1:  # substring either way
                    component.add(c2)
                    used.add(c2)
            if len(component) > 1:
                components.append(component)
            used.add(c1)
        
        # for each component, find the dominant stable cluster
        for component in components:
            valid_clusters = [
                cluster_map[c] for c in component 
                if cluster_map[c] != -1
            ]
            if not valid_clusters:
                continue
            
            # pick cluster with most members overall
            dominant = max(valid_clusters, 
                          key=lambda c: cluster_sizes.get(c, 0))
            
            for c in component:
                old = new_map[c]
                if old != dominant:
                    new_map[c] = dominant
                    reassigned += 1

    
    df = df.copy()
    df["stable_cluster"] = df["norm_concept"].map(new_map)
    print(f"  Substring merge: {reassigned:,} concepts reassigned")
    print(f"  Remaining noise: {(df['stable_cluster']==-1).sum():,}")

    return df


for year in range(2000,2027):
    df = pd.read_csv(f"concept_provenance/stable_clustered_concepts_provenance_{year}.csv") # adjust to your own path
    df = merge_substring_clusters(df)
    df.to_csv(f"concept_provenance/stable_clustered_concepts_provenance_{year}_merged.csv", index=False) # adjust to your own path