import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import pearsonr
import numpy as np

# Load the data
lastfm_edges = pd.read_csv('data/lastfm_asia_edges.csv')
print(lastfm_edges.head())

# Convert the dataframe to NetworkX graph
GA = nx.from_pandas_edgelist(lastfm_edges, 'node_1', 'node_2')
node_degree_before = str(GA).split(' ')[2]
edge_degree_before = str(GA).split(' ')[5]
print(GA)

##### 1. Preprocessing #####

# Remove nodes with degree less than 2
nodes_to_remove = [node for node, degree in dict(GA.degree()).items() if degree < 2]
GA_filtered = GA.copy()
GA_filtered.remove_nodes_from(nodes_to_remove)
node_degree_after = str(GA_filtered).split(' ')[2]
edge_degree_after = str(GA_filtered).split(' ')[5]
print(GA_filtered)

# Change in percentages
node_degree_change = int(node_degree_after) / int(node_degree_before) * 100
edge_degree_change = int(edge_degree_after) / int(edge_degree_before) * 100
print(f"Node degree change: {node_degree_change:.2f}%", f"Edge degree change: {edge_degree_change:.2f}%\n")

# while (len(nodes_to_remove)>0):
#     GA_filtered = GA_filtered.copy()
#     GA_filtered.remove_nodes_from(nodes_to_remove)
#     node_degree_after = str(GA_filtered).split(' ')[2]
#     edge_degree_after = str(GA_filtered).split(' ')[5]
#     print(GA_filtered)

#     # Change in percentages
#     node_degree_change = int(node_degree_after) / int(node_degree_before) * 100
#     edge_degree_change = int(edge_degree_after) / int(edge_degree_before) * 100
#     print(f"Node degree change: {node_degree_change:.2f}%", f"Edge degree change: {edge_degree_change:.2f}%\n")
#     nodes_to_remove = [node for node, degree in dict(GA_filtered.degree()).items() if degree < 2]



##### 2. Graph centrality properties #####

# Degree
degree = dict(GA_filtered.degree())
# Add the degree as a node attribute
nx.set_node_attributes(GA_filtered, degree, 'degree')

# Calculate the clustering coefficient
clustering_coefficient = nx.clustering(GA_filtered)
# Calculate the average clustering coefficient
avg_clustering_coefficient = np.mean(list(clustering_coefficient.values()))
print(f"Average clustering coefficient: {avg_clustering_coefficient}")
# Add the clustering coefficient as a node attribute
nx.set_node_attributes(GA_filtered, clustering_coefficient, 'Clustering Coefficient')

# Calculate the shortest path length
avg_path_length = nx.average_shortest_path_length(GA_filtered)
print(f"Average path length: {avg_path_length}")
# Add the shortest path length as a node attribute
nx.set_node_attributes(GA_filtered, avg_path_length, 'avg_path_length')

# Calculate the Average Degree
avg_degree = sum(dict(GA_filtered.degree()).values()) / len(GA_filtered)
print(f"Average degree: {avg_degree}")
# Add the average degree as a node attribute
nx.set_node_attributes(GA_filtered, avg_degree, 'avg_degree')

# Calculate the diameter
diameter = nx.diameter(GA_filtered)
print(f"Diameter: {diameter}")
# Add the diameter as a node attribute
nx.set_node_attributes(GA_filtered, diameter, 'diameter')

# Calculate the freemen centrality
def freeman_centralization(G):
    max_possible_ties = len(G.nodes()) - 1
    total_diff = sum(max_possible_ties - len(list(G.neighbors(node))) for node in G.nodes())
    total_possible_diff = len(G.nodes()) * (len(G.nodes()) - 1)
    centralization_index = total_diff / total_possible_diff if total_possible_diff != 0 else 0
    return centralization_index

freeman_centralization_index = freeman_centralization(GA_filtered)
print(f"Freeman centralization index: {freeman_centralization_index}")

# Add the Freeman centralization index as a node attribute
nx.set_node_attributes(GA_filtered, freeman_centralization_index, 'freeman_centralization_index')
# Calculate the density
density = nx.density(GA_filtered)
print(f"Density: {density}")
# Add the density as a node attribute
nx.set_node_attributes(GA_filtered, density, 'density')


##### 3. Centrality index of the top 10 leading nodes #####

print("\nTop10 list of the main users in the network with the help of centrality indices")
# Calculate the degree centrality
degree_centrality = nx.degree_centrality(GA_filtered)
# Calculate the average degree centrality
avg_degree_centrality = np.mean(list(degree_centrality.values()))
print(f"Average degree centrality: {avg_degree_centrality}")
# Print the top 10 nodes with the highest degree centrality
top_10_degree_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"Top 10 nodes with highest degree centrality: ")
for i in range(len(top_10_degree_centrality)):
    print(f"{i+1}. {top_10_degree_centrality[i]}")
# Add the degree centrality as a node attribute
nx.set_node_attributes(GA_filtered, degree_centrality, 'degree_centrality')

# Calculate the betweenness centrality
betweenness_centrality = nx.betweenness_centrality(GA_filtered)
# Calculate the average betweenness centrality
avg_betweenness_centrality = np.mean(list(betweenness_centrality.values()))
print(f"Average betweenness centrality: {avg_betweenness_centrality}")
# Print the top 10 nodes with the highest betweenness centrality
top_10_betweenness_centrality = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"Top 10 nodes with highest betweenness centrality: ")
for i in range(len(top_10_betweenness_centrality)):
    print(f"{i+1}. {top_10_betweenness_centrality[i]}")
# Add the betweenness centrality as a node attribute
nx.set_node_attributes(GA_filtered, betweenness_centrality, 'betweenness_centrality')

# Calculate the closeness centrality
closeness_centrality = nx.closeness_centrality(GA_filtered)
# Calculate the average closeness centrality
avg_closeness_centrality = np.mean(list(closeness_centrality.values()))
print(f"Average closeness centrality: {avg_closeness_centrality}")
# Print the top 10 nodes with the highest closeness centrality
top_10_closeness_centrality = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"Top 10 nodes with highest closeness centrality:")
for i in range(len(top_10_closeness_centrality)):
    print(f"{i+1}. {top_10_closeness_centrality[i]}")
# Add the closeness centrality as a node attribute
nx.set_node_attributes(GA_filtered, closeness_centrality, 'closeness_centrality')


# Calculate the eigenvector centrality
eigenvector_centrality = nx.eigenvector_centrality(GA_filtered)
# Calculate the average eigenvector centrality
avg_eigenvector_centrality = np.mean(list(eigenvector_centrality.values()))
print(f"Average eigenvector centrality: {avg_eigenvector_centrality}")
# Print the top 10 nodes with the highest eigenvector centrality
top_10_eigenvector_centrality = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"Top 10 nodes with highest eigenvector centrality:")
for i in range(len(top_10_eigenvector_centrality)):
    print(f"{i+1}. {top_10_eigenvector_centrality[i]}")
# Add the eigenvector centrality as a node attribute
nx.set_node_attributes(GA_filtered, eigenvector_centrality, 'eigenvector_centrality')


# Calculate the PageRank
pagerank = nx.pagerank(GA_filtered)
# Calculate the average PageRank
avg_pagerank = np.mean(list(pagerank.values()))
print(f"Average PageRank: {avg_pagerank}")
# Print the top 10 nodes with the highest PageRank
top_10_PageRank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"Top 10 nodes with highest PageRank:")
for i in range(len(top_10_PageRank)):
    print(f"{i+1}. {top_10_PageRank[i]}")
# Add the PageRank as a node attribute
nx.set_node_attributes(GA_filtered, pagerank, 'pagerank')



##### 4. Correlation between the various centrality indices #####
print("\nCorrelation between the various centrality indices")

measures = {
    "Degree Centrality": degree_centrality,
    "Betweenness Centrality": betweenness_centrality,
    "Closeness Centrality": closeness_centrality,
    "Eigenvector Centrality": eigenvector_centrality,
    "PageRank": pagerank,
}

correlation_coefficients = {}

for measure1 in measures:
    for measure2 in measures:
        if measure1 != measure2:
            coefficient, _ = pearsonr(list(measures[measure1].values()), list(measures[measure2].values()))
            if (measure2, measure1) not in correlation_coefficients:
                correlation_coefficients[(measure1, measure2)] = coefficient


# Print correlation coefficients
for (measure1, measure2), coefficient in correlation_coefficients.items():
    print(f"Pearson correlation between {measure1} and {measure2}: {coefficient}")
    # Add the correlation as a node attribute
    nx.set_node_attributes(GA_filtered, {node: coefficient for node in GA_filtered.nodes()}, name=f"{measure1}_{measure2}_correlation")

# Create a dataframe from the correlation coefficients
df = pd.DataFrame(index=measures.keys(), columns=measures.keys())
for (measure1, measure2), coefficient in correlation_coefficients.items():
    df.loc[measure1, measure2] = coefficient
    df.loc[measure2, measure1] = coefficient

# Plot the heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(df.astype(float), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlations Between Network Measures")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
plt.tight_layout()
plt.show()

# outliners in the network 
outlinersHigh_Betweenness_Low_Degree = []
for node in GA_filtered.nodes():
    if float(degree[node]) <  avg_degree and betweenness_centrality[node] > avg_betweenness_centrality:
        outlinersHigh_Betweenness_Low_Degree.append(node)
print(f"Outliners in the network [High Betweenness, Low Degree]: {outlinersHigh_Betweenness_Low_Degree}")
nx.set_node_attributes(GA_filtered, {node: 1 for node in outlinersHigh_Betweenness_Low_Degree}, name=f"outliners High Betweenness, Low Degree")

outliners_High_PageRank_Low_Degree = []
for node in GA_filtered.nodes():
    if float(degree[node]) <  avg_degree and pagerank[node] > avg_pagerank:
        outliners_High_PageRank_Low_Degree.append(node)
print(f"Outliners in the network [High PageRank, Low Degree]: {outliners_High_PageRank_Low_Degree}")
nx.set_node_attributes(GA_filtered, {node: 1 for node in outliners_High_PageRank_Low_Degree}, name=f"outliners High PageRank, Low Degree")

intersection_PageRank_and_Betweenness_lists = set(outliners_High_PageRank_Low_Degree).intersection(set(outlinersHigh_Betweenness_Low_Degree))
print(f"Intersection between [[High Betweenness, Low Degree] & [High PageRank, Low Degree]]: {intersection_PageRank_and_Betweenness_lists}")

outliners = []
for node in GA_filtered.nodes():
    if float(degree[node]) >  avg_degree and eigenvector_centrality[node] < avg_eigenvector_centrality:
        outliners.append(node)
print(f"Outliners in the network [Low Eigenvector Centrality, High Degree]: {outliners}")
nx.set_node_attributes(GA_filtered, {node: 1 for node in outliners}, name=f"outliners Low Eigenvector Centrality, High Degree")





##### 5. Network communities #####

# Load the data user and country and his/her favorite artists
lastfm_target = pd.read_csv('data/lastfm_asia_target.csv')
# lastfm_asia_features = json.load(open('lastfm_asia_features.json'))

# Detect communities using the Louvain method
communities = nx.algorithms.community.greedy_modularity_communities(GA_filtered)
community_list = [list(community) for community in communities]
modularity_score = nx.algorithms.community.quality.modularity(GA_filtered, community_list)
print(f"Modularity Score: {modularity_score}")

outer_list_of_countries = []
outer_community = []
outer_community_artist_preferences = {}

# For each community, check if nodes are from the same country
for i, community in enumerate(communities):
    # print(f"Community {i + 1}: {community}")
    print(f"Community {i + 1}")
    community_data = lastfm_target[lastfm_target['id'].isin(community)] # Filter the node data for nodes in the current community

    country_communities = community_data.groupby('target')['id'].apply(lambda x: list(x)).to_dict()
    list_of_countries = []
    for country in country_communities:
        list_of_countries.append((country, len((country_communities)[country])/len(community)))
    sorted_country_communities = sorted(list_of_countries, key=lambda x: x[1], reverse=True)
    print(f"Country distribution for Community {i + 1}: {sorted_country_communities}")

    lastfm_asia_features = json.load(open('lastfm_asia_features.json'))
    # Extract artist preferences for nodes in the community from the JSON data
    community_artist_preferences = {}
    for node_id in community:
        for artist in lastfm_asia_features[str(node_id)]:
            if artist in community_artist_preferences:
                community_artist_preferences[artist] += 1 / len(community)
            else:
                community_artist_preferences[artist] = 1 / len(community)

    # Print the top 10 artist preferences for the community
    top_10 = sorted(community_artist_preferences.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"Top 10 artist preferences for Community {i + 1}: {top_10}")

    # summarize the community
    print(f"Community {i + 1} summary:")
    print("-" * 25)
    print(f"{len(community)} nodes, {len(country_communities)} countries, and {len(community_artist_preferences)} artists.")
    print(f"Percentage of the community: {len(community)/len(GA_filtered.nodes())*100:.2f}% of all nodes in the network\n")

    # Draw the graph using Spring layout
    GA_community = nx.from_pandas_edgelist(community_data, 'id', 'target')
    bridges = []
    for u, v in GA_community.edges():
        GA_community.remove_edge(u, v)
        if not nx.is_connected(GA_community):
            bridges.append((u, v))
        GA_community.add_edge(u, v)
    # Visualize the graph and bridges
    pos = nx.spring_layout(GA_community)
    nx.draw(GA_community, pos, with_labels=True, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(GA_community, pos, edgelist=bridges, edge_color='red', width=2)
    plt.title('Network with Bridges Highlighted')
    plt.show()
    outer_list_of_countries.append(sorted_country_communities)
    outer_community.append(community)
    outer_community_artist_preferences[i+1] = top_10
    # Add the community as a node attribute
    nx.set_node_attributes(GA_filtered, {node: i + 1 for node in community}, name="community")

# Add the node to country as a node attribute
nx.set_node_attributes(GA_filtered, lastfm_target.set_index('id')['target'].to_dict(), name="country")





# Initialize an empty dictionary to store the inverted key-value pairs
lastfm_asia_features = json.load(open('lastfm_asia_features.json'))
inverted_dict = {}
for key, value_list in lastfm_asia_features.items():
    for value in value_list:
        inverted_dict.setdefault(value, []).append(key)
# which artist is listened to the most by the nodes in the network
sorted_keys = sorted(inverted_dict.keys(), key=lambda k: len(inverted_dict[k]), reverse=True)[:10]
print(f"Top 10 artists with the most listeners: {sorted_keys}")
for i in range(len(sorted_keys)):
    print(f"{i+1}. {sorted_keys[i]}")


# Which artist is listened to the most by the nodes in each country
for country in lastfm_target['target'].unique():
    country_data = lastfm_target[lastfm_target['target'] == country]
    lastfm_asia_features = json.load(open('lastfm_asia_features.json'))
    # Extract artist preferences for nodes in the country from the JSON data
    country_artist_preferences = {}
    for node_id in country_data['id']:
        for artist in lastfm_asia_features[str(node_id)]:
            if artist in country_artist_preferences:
                country_artist_preferences[artist] += 1 / len(country_data)
            else:
                country_artist_preferences[artist] = 1 / len(country_data)

    # Print the top 10 artist preferences for the country
    top_10 = sorted(country_artist_preferences.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"Top 10 artist preferences for country {country}: {top_10}")
    for i in range(len(top_10)):
        print(f"{i+1}. {top_10[i]}")



#### 7. Link prediction  ####

preds_jc = nx.jaccard_coefficient(GA_filtered)
pred_jc_dict = {}
for u, v, p in preds_jc:
    pred_jc_dict[(u,v)] = p
jaccard = sorted(pred_jc_dict.items(), key=lambda x:x[1], reverse=True)[:10]
print(f"The new predicted friendship Jaccard will be: {jaccard}")
# add the jaccard coefficient as a node attribute
nx.set_node_attributes(GA_filtered, pred_jc_dict, 'jaccard_coefficient')
preds_aa = nx.adamic_adar_index(GA_filtered)
pred_aa_dict = {}
for u, v, p in preds_aa:
    pred_aa_dict[(u,v)] = p
adamic = sorted(pred_aa_dict.items(), key=lambda x:x[1], reverse=True)[:10]
print(f"The new predicted friendship in Adamic Adar will be: {adamic}")
# add the adamic adar index as a node attribute
nx.set_node_attributes(GA_filtered, pred_aa_dict, 'adamic_adar_index')


#### 8. Excel file report ####

# Collect all unique IDs from all centrality measures
all_ids = set()
# top_10_clustering_coefficient = sorted(clustering_coefficient.items(), key=lambda x: x[1], reverse=True)[:10]
for centrality_measure in [top_10_degree_centrality, top_10_betweenness_centrality,
                           top_10_closeness_centrality, top_10_eigenvector_centrality, top_10_PageRank]:
    all_ids.update([node[0] for node in centrality_measure])

with pd.ExcelWriter('output.xlsx') as writer:

    # Create a Pandas DataFrame with the centrality measures
    centrality_measures = pd.DataFrame({
        "ID": list(all_ids),
        "Degree Centrality": [next((val[1] for val in top_10_degree_centrality if val[0] == id), None) for id in all_ids],
        "Betweenness Centrality": [next((val[1] for val in top_10_betweenness_centrality if val[0] == id), None) for id in all_ids],
        "Closeness Centrality": [next((val[1] for val in top_10_closeness_centrality if val[0] == id), None) for id in all_ids],
        "Eigenvector Centrality": [next((val[1] for val in top_10_eigenvector_centrality if val[0] == id), None) for id in all_ids],
        "PageRank": [next((val[1] for val in top_10_PageRank if val[0] == id), None) for id in all_ids],
    })

    # Add columns for "Country", "Community_for_nodes", and "artist_community"
    centrality_measures['Country'] = centrality_measures['ID'].apply(lambda x: lastfm_target[lastfm_target['id'] == x]['target'].values[0])
    centrality_measures['Community_for_nodes'] = centrality_measures['ID'].apply(lambda x: next((i+1 for i, community in enumerate(outer_community) if x in community), 0))
    centrality_measures['artist_community'] = centrality_measures['Community_for_nodes'].apply(lambda x: outer_community_artist_preferences[x] if x > 0 else None)

    # Rearrange columns
    centrality_measures = centrality_measures[['ID', 'Country', 'Community_for_nodes', 'artist_community',
                                               'Degree Centrality', 'Betweenness Centrality',
                                               'Closeness Centrality', 'Eigenvector Centrality', 'PageRank']]

    centrality_measures.to_excel(writer, sheet_name='Centrality Measures', index=False)

# Export the graph
nx.write_gexf(GA_filtered,'GA_filtered_update.gexf')
