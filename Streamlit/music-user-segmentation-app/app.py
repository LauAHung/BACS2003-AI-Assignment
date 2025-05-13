import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

st.title("BACS2003 Artificial Intelligence")
st.markdown("### Segmentation of Music Streaming Users")
st.markdown("""
## Group Members
| Name | Student ID |
| --- | --- |
| LAU AIK HUNG | 23WMR14555 |
| KESHANDRA A/L JAYASELAN | 23WMR14549 |
| LEONG CHUN XIANG | 23WMR15624 |
""")

st.header("Dataset")
csv_file = st.file_uploader("Upload your music_users.csv file", type=["csv"])
if csv_file:
    df = pd.read_csv(csv_file)
    st.dataframe(df)
    # Data Preparation
    df.columns = df.columns.str.strip()
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 2})
    df['FavoriteMusicType'] = df['FavoriteMusicType'].replace({
        'Rock': 1, 'Pop': 2, 'J-Pop': 3, 'K-Pop': 4, 'EDM': 5, 'Jazz': 6, 'Chill': 7, 'Oldies': 8, 'Traditional': 9
    })
    df['PaidSubscription'] = df['PaidSubscription'].replace({'Yes': 1, 'No': 2})
    df['ListeningPlatform'] = df['ListeningPlatform'].replace({
        'Spotify': 1, 'Apple Music': 2, 'QQ Music': 3, 'NetEase Cloud': 4, 'YouTube Music': 5, 'Amazon Music': 6
    })
    df['Country'] = df['Country'].replace({
        'China': 1, 'USA': 2, 'Japan': 3, 'Korea': 4, 'UK': 5, 'Germany': 6, 'France': 7, 'Brazil': 8, 'India': 9, 'Australia': 10
    })
    df['SubscriptionType'] = df['SubscriptionType'].replace({
        'Free': 1, 'Premium': 2, 'Family': 3, 'Student': 4
    })
    df['GenrePreferenceLevel'] = df['GenrePreferenceLevel'].replace({'High': 1, 'Medium': 2, 'Low': 3})
    df['DeviceType'] = df['DeviceType'].replace({'Mobile': 1, 'PC': 2, 'Tablet': 3, 'Smart Speaker': 4})
    df['ListeningTimeOfDay'] = df['ListeningTimeOfDay'].replace({'Morning': 1, 'Afternoon': 2, 'Evening': 3, 'Night': 4})
    if 'UserID' in df.columns:
        df = df.drop(['UserID'], axis=1)
    st.markdown("### Data after transformation")
    st.dataframe(df.head())

    st.markdown("### Descriptive Statistics")
    st.dataframe(df.describe())

    st.header("Data Visualization")
    corr_matrix = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax)
    st.pyplot(fig)

    st.markdown("### Select Features for Clustering")
    default_features = ['Age', 'NumberOfPlaylists', 'WeeklyListeningHours', 'FavoriteMusicType']
    features = st.multiselect("Please select features", df.columns.tolist(), default=default_features)
    if len(features) < 2:
        st.warning("Please select at least two features for clustering!")
    else:
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df[features])

        st.markdown("### Pairplot Visualization (Selected Features)")
        if st.button("Show Pairplot (All Rows, Selected Features, Jittered)"):
            # Scale the selected features
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(df[features])
            # Add jitter
            jittered_data = data_scaled + np.random.normal(0, 0.05, data_scaled.shape)
            jittered_df = pd.DataFrame(jittered_data, columns=features)
            pair_grid = sns.pairplot(jittered_df, plot_kws={'s': 15, 'alpha': 0.6})
            pair_grid.fig.suptitle("Pairplot with Jitter (All Rows, Selected Features)", y=1.02)
            st.pyplot(pair_grid.figure)

        st.sidebar.header("Clustering Algorithm")
        algorithm = st.sidebar.selectbox(
            "Please select clustering algorithm",
            ("KMeans", "MeanShift", "Agglomerative Hierarchical Clustering", "Compare All Algorithms")
        )

        if algorithm == "KMeans":
            st.markdown("### K-Means Clustering")

            # 1. Best K in Mean (show best_k by silhouette)
            silhouette_scores = []
            calinski_scores = []
            davies_scores = []
            K_range = range(2, 11)
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(scaled_df)
                silhouette_scores.append(silhouette_score(scaled_df, labels))
                calinski_scores.append(calinski_harabasz_score(scaled_df, labels))
                davies_scores.append(davies_bouldin_score(scaled_df, labels))
            best_k = K_range[np.argmax(silhouette_scores)]
            st.markdown(f"**Best k by silhouette: {best_k}**")

            # 1. Use slider to dynamically select K value
            k = st.sidebar.slider("Select KMeans Clusters", min_value=2, max_value=10, value=best_k)
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(scaled_df)
            st.write(f"Current K Value: {k}, Silhouette Score: {silhouette_score(scaled_df, labels):.3f}")

            # 2. Elbow Method
            st.markdown("### Elbow Method")
            wcss = []
            K_range_elbow = range(1, 11)
            for k in K_range_elbow:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                kmeans.fit(scaled_df)
                wcss.append(kmeans.inertia_)
            fig, ax = plt.subplots()
            ax.plot(K_range_elbow, wcss, marker='o')
            ax.set_title('Elbow Method')
            ax.set_xlabel('Number of clusters (K)')
            ax.set_ylabel('WCSS')
            st.pyplot(fig)

            # 3. Distribution plots (boxplots by cluster)
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
            df['Cluster'] = kmeans.fit_predict(scaled_df)
            st.markdown("### Cluster Distributions (Best k)")
            for col in features:
                if col in df.columns:
                    st.markdown(f"#### {col} Distribution by Cluster")
                    fig, ax = plt.subplots()
                    sns.boxplot(x='Cluster', y=col, data=df, ax=ax)
                    ax.set_title(f'{col} Distribution by Cluster')
                    st.pyplot(fig)

            # 4. PCA cluster plot
            st.markdown("### PCA Cluster Plot (Best k)")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_df)
            df['PCA1'] = pca_result[:, 0]
            df['PCA2'] = pca_result[:, 1]
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', ax=ax)
            ax.set_title('KMeans Clusters (PCA Projection)')

            # Project centroids to PCA space and plot
            centroids = kmeans.cluster_centers_  # shape: (n_clusters, n_features)
            centroids_pca = pca.transform(centroids)  # shape: (n_clusters, 2)
            ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                    marker='X', s=200, c='black', label='Centroids', edgecolor='white')
            ax.legend()
            st.pyplot(fig)

            # 2. PCA Visualization with Plotly
            st.markdown("### PCA Visualization with Plotly")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_df)
            df['PCA1'] = pca_result[:, 0]
            df['PCA2'] = pca_result[:, 1]
            fig = px.scatter(df, x='PCA1', y='PCA2', color=labels.astype(str), title="KMeans Clustering PCA Projection")

            st.plotly_chart(fig)

            # 4b. KMeans with n_clusters=2: means, boxplots, PCA
            st.markdown("### KMeans (n_clusters=2) Cluster Means")
            features_kmean = ['Age', 'WeeklyListeningHours', 'NumberOfPlaylists', 'FavoriteMusicType']
            kmeans_2 = KMeans(n_clusters=2, random_state=42)
            df['Cluster_2'] = kmeans_2.fit_predict(scaled_df)
            means_table = df.groupby('Cluster_2')[features_kmean].mean()
            st.dataframe(means_table)


            # 5. Silhouette Score graph
            st.markdown("### Silhouette Score by K")
            best_k_silhouette = K_range[np.argmax(silhouette_scores)]
            fig, ax = plt.subplots()
            ax.plot(K_range, silhouette_scores, marker='o', color='orange')
            ax.axvline(x=best_k_silhouette, color='red', linestyle='--', label=f'Best k = {best_k_silhouette}')
            ax.set_title('Silhouette Score (KMeans)')
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel('Score')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            # Show Silhouette values
            sil_str = 'Silhouette Scores (KMeans):\n' + '\n'.join([
                f"Clusters = {k}, Score = {score:.4f}" for k, score in zip(K_range, silhouette_scores)
            ])
            st.text(sil_str)

            # 6. Calinski-Harabasz Index graph
            st.markdown("### Calinski-Harabasz Index by K")
            best_k_calinski = K_range[np.argmax(calinski_scores)]
            fig, ax = plt.subplots()
            ax.plot(K_range, calinski_scores, marker='o', color='blue')
            ax.axvline(x=best_k_calinski, color='red', linestyle='--', label=f'Best k = {best_k_calinski}')
            ax.set_title('Calinski-Harabasz Index (KMeans)')
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel('Score')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            # Show Calinski values
            cal_str = 'Calinski-Harabasz Scores (KMeans):\n' + '\n'.join([
                f"Clusters = {k}, Score = {score:.4f}" for k, score in zip(K_range, calinski_scores)
            ])
            st.text(cal_str)

            # 7. Davies-Bouldin Index graph
            st.markdown("### Davies-Bouldin Index by K")
            best_k_davies = K_range[np.argmin(davies_scores)]
            fig, ax = plt.subplots()
            ax.plot(K_range, davies_scores, marker='o', color='purple')
            ax.axvline(x=best_k_davies, color='red', linestyle='--', label=f'Best k = {best_k_davies}')
            ax.set_title('Davies-Bouldin Index (KMeans)')
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel('Score')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            # Show Davies-Bouldin values
            db_str = 'Davies-Bouldin Index (KMeans):\n' + '\n'.join([
                f"Clusters = {k}, Score = {score:.4f}" for k, score in zip(K_range, davies_scores)
            ])
            st.text(db_str)


            # 8. Clustering metrics for best K
            st.markdown("### Clustering Metrics for Best K")
            s_k = silhouette_score(scaled_df, df['Cluster'])
            c_k = calinski_harabasz_score(scaled_df, df['Cluster'])
            d_k = davies_bouldin_score(scaled_df, df['Cluster'])
            st.write(f"Silhouette Score: {s_k:.4f}")
            st.write(f"Calinski-Harabasz Index: {c_k:.4f}")
            st.write(f"Davies-Bouldin Index: {d_k:.4f}")

            # 9. Cluster summary
            st.markdown("### Cluster Summary")
            summary = []
            for cluster in sorted(df['Cluster'].unique()):
                cluster_df = df[df['Cluster'] == cluster]
                avg_age = cluster_df['Age'].mean()
                avg_hours = cluster_df['WeeklyListeningHours'].mean()
                avg_playlists = cluster_df['NumberOfPlaylists'].mean()
                if 'FavoriteMusicType' in cluster_df.columns:
                    music_map = {1:'Rock',2:'Pop',3:'J-Pop',4:'K-Pop',5:'EDM',6:'Jazz',7:'Chill',8:'Oldies',9:'Traditional'}
                    most_common_music = cluster_df['FavoriteMusicType'].mode()[0]
                    most_common_music = music_map.get(most_common_music, most_common_music)
                else:
                    most_common_music = 'N/A'
                summary.append(
                    f"**Cluster {cluster}:**\n"
                    f"- Average Age: {avg_age:.1f}\n"
                    f"- Weekly Listening Hours: {avg_hours:.1f}\n"
                    f"- Average Playlists: {avg_playlists:.1f}\n"
                    f"- Most Common Favorite Music Type: {most_common_music}\n"
                )
            st.markdown('\n'.join(summary))


        elif algorithm == "MeanShift":
            st.markdown("### MeanShift Clustering")
            # 1. Parameter Sensitivity Analysis
            st.markdown("#### Parameter Sensitivity Analysis (Quantile)")
            quantiles = [0.1, 0.2, 0.3, 0.4]
            quantile_results = []
            for q in quantiles:
                bandwidth = estimate_bandwidth(scaled_df, quantile=q, n_samples=5000)
                meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                meanshift.fit(scaled_df)
                labels = meanshift.labels_
                n_clusters = len(set(labels))
                result = f"Quantile: {q} | Clusters: {n_clusters}"
                if n_clusters > 1:
                    score = silhouette_score(scaled_df, labels)
                    result += f" | Silhouette Score: {score:.4f}"
                else:
                    result += " | Silhouette Score: Not applicable (only one cluster)"
                quantile_results.append(result)
            st.text("\n".join(quantile_results))

            # 2. Estimate bandwidth for quantile=0.1
            st.markdown("#### Estimate Bandwidth (quantile=0.1)")
            bandwidth = estimate_bandwidth(scaled_df, quantile=0.1, n_samples=5000)
            st.write(f"Estimated Bandwidth: {bandwidth:.4f}")

            # 3. Apply MeanShift
            meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            meanshift.fit(scaled_df)
            labels_ms = meanshift.labels_
            cluster_centers = meanshift.cluster_centers_
            n_clusters_ = len(set(labels_ms))
            # Reverse labels if exactly 2 clusters
            if n_clusters_ == 2:
                labels_ms = np.where(labels_ms == 0, 1, 0)
            st.write(f"Number of estimated clusters: {n_clusters_}")

            # 4. Evaluate clustering performance
            if n_clusters_ > 1:
                silhouette_avg = silhouette_score(scaled_df, labels_ms)
                st.write(f"Silhouette Score for MeanShift: {silhouette_avg:.4f}")
            else:
                silhouette_avg = None
                st.write("Silhouette Score for MeanShift: Not applicable (only one cluster)")

            # Step 4: Kernel Density Estimation (KDE) for PCA Reduced Data
            # Perform PCA to reduce the data to 2 dimensions
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(scaled_df)

            # Create a DataFrame for the reduced data
            df_pca = pd.DataFrame(reduced_data, columns=['PCA 1', 'PCA 2'])
            df_pca['Cluster'] = labels_ms

            # KDE plot of the reduced 2D data
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.kdeplot(
                data=df_pca,
                x='PCA 1',
                y='PCA 2',
                hue='Cluster',
                fill=True,
                common_norm=False, 
                alpha=0.5,
                palette='tab10',
                ax=ax
            )
            ax.set_title('KDE Plot in PCA Reduced Feature Space (MeanShift)')
            st.pyplot(fig)


            # 6. PCA 2D projection with centroids
            st.markdown("#### MeanShift Clustering (PCA 2D Projection)")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(scaled_df)
            fig, ax = plt.subplots(figsize=(10, 6))
            unique_labels = np.unique(labels_ms)
            for cluster_id in unique_labels:
                cluster_points = X_pca[labels_ms == cluster_id]
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, alpha=0.9, label=f'Cluster {cluster_id}')
            if n_clusters_ > 0:
                centroids_pca = pca.transform(meanshift.cluster_centers_)
                ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, c='red', label='Centroids', edgecolor='black')
            ax.set_title(f"MeanShift Clustering (Quantile=0.1, Clusters={n_clusters_})")
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.legend()
            ax.grid(alpha=0.2)
            st.pyplot(fig)

            # Plotly version for interactive PCA 2D projection with centroids
            st.markdown("#### MeanShift Clustering (PCA 2D Projection, Plotly)")
            pca_result = pca.fit_transform(scaled_df)
            df['PCA1'] = pca_result[:, 0]
            df['PCA2'] = pca_result[:, 1]

            fig = px.scatter(
                df,
                x='PCA1',
                y='PCA2',
                color=labels_ms.astype(str),
                title="MeanShift Clustering (PCA 2D Projection, Plotly)"
            )

            fig.update_traces(marker=dict(opacity=0.6), selector=dict(mode='markers'))

            centroids = meanshift.cluster_centers_
            centroids_pca = pca.transform(centroids)
            fig.add_trace(
                go.Scatter(
                    x=centroids_pca[:, 0],
                    y=centroids_pca[:, 1],
                    mode='markers',
                    marker=dict(
                        symbol='x',
                        size=36,
                        color='yellow',
                        line=dict(width=4, color='black'),
                        opacity=1
                    ),
                    name='Centroids',
                    showlegend=True
                )
            )

            st.plotly_chart(fig)


            # Cluster Size Distribution Plot (MeanShift)
            st.markdown("#### Cluster Size Distribution (MeanShift)")
            from collections import Counter
            cluster_counts = Counter(labels_ms)
            cluster_counts_df = pd.DataFrame({
                'Cluster': list(cluster_counts.keys()),
                'Count': list(cluster_counts.values())
            })
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=cluster_counts_df, x='Cluster', y='Count', hue='Cluster', palette='viridis', dodge=False, ax=ax)
            ax.set_xlabel('Cluster Label')
            ax.set_ylabel('Number of Points')
            ax.set_title('Cluster Size Distribution (MeanShift)')
            ax.legend(title='Cluster', loc='upper right')
            plt.tight_layout()
            st.pyplot(fig)

            # 7. Bandwidth sensitivity analysis for metrics
            bandwidth_range = np.linspace(0.5, 2.0, 10)
            # Silhouette
            silhouette_scores = []
            for bandwidth in bandwidth_range:
                meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                meanshift.fit(scaled_df)
                labels = meanshift.labels_
                n_clusters = len(np.unique(labels))
                if n_clusters > 1:
                    score = silhouette_score(scaled_df, labels)
                    silhouette_scores.append(score)
                else:
                    silhouette_scores.append(0)
            best_bandwidth_silhouette = bandwidth_range[np.argmax(silhouette_scores)]
            best_silhouette = np.max(silhouette_scores)
            st.markdown("#### Silhouette Score for Different Bandwidths")
            fig, ax = plt.subplots()
            ax.plot(bandwidth_range, silhouette_scores, marker='o', color='orange')
            ax.axvline(x=best_bandwidth_silhouette, color='r', linestyle='--', label=f'Best Bandwidth (Silhouette) = {best_bandwidth_silhouette:.2f}')
            ax.set_title('Silhouette Score for MeanShift')
            ax.set_xlabel('Bandwidth')
            ax.set_ylabel('Score')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            st.info(f'Best Silhouette Score: {best_silhouette:.4f} at Bandwidth = {best_bandwidth_silhouette:.2f}')

            # Calinski-Harabasz
            calinski_scores = []
            for bandwidth in bandwidth_range:
                meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                meanshift.fit(scaled_df)
                labels = meanshift.labels_
                if len(np.unique(labels)) > 1:
                    calinski_score = calinski_harabasz_score(scaled_df, labels)
                    calinski_scores.append(calinski_score)
                else:
                    calinski_scores.append(0)
            best_bandwidth_calinski = bandwidth_range[np.argmax(calinski_scores)]
            best_calinski = np.max(calinski_scores)
            st.markdown("#### Calinski-Harabasz Scores for Different Bandwidths")
            fig, ax = plt.subplots()
            ax.plot(bandwidth_range, calinski_scores, marker='o', color='blue')
            ax.axvline(x=best_bandwidth_calinski, color='r', linestyle='--', label=f'Best Bandwidth (Calinski) = {best_bandwidth_calinski:.2f}')
            ax.set_title('Calinski-Harabasz Index for MeanShift')
            ax.set_xlabel('Bandwidth')
            ax.set_ylabel('Score')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            st.info(f'Best Calinski-Harabasz Index: {best_calinski:.4f} at Bandwidth = {best_bandwidth_calinski:.2f}')

            # Davies-Bouldin
            davies_bouldin_scores = []
            for bandwidth in bandwidth_range:
                meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                meanshift.fit(scaled_df)
                labels = meanshift.labels_
                n_clusters = len(np.unique(labels))
                if n_clusters > 1:
                    db_score = davies_bouldin_score(scaled_df, labels)
                    davies_bouldin_scores.append(db_score)
                else:
                    davies_bouldin_scores.append(np.inf)
            best_bandwidth_db = bandwidth_range[np.argmin(davies_bouldin_scores)]
            best_db = np.min(davies_bouldin_scores)
            st.markdown("#### Davies-Bouldin Scores for Different Bandwidths")
            fig, ax = plt.subplots()
            ax.plot(bandwidth_range, davies_bouldin_scores, marker='o', color='green')
            ax.axvline(x=best_bandwidth_db, color='r', linestyle='--', label=f'Best Bandwidth (DB) = {best_bandwidth_db:.2f}')
            ax.set_title('Davies-Bouldin Index for MeanShift')
            ax.set_xlabel('Bandwidth')
            ax.set_ylabel('Score')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            st.info(f'Best Davies-Bouldin Index: {best_db:.4f} at Bandwidth = {best_bandwidth_db:.2f}')

            # 8. Add cluster label to df and show means
            df["MeanShift_Cluster"] = labels_ms
            feature_columns = ['Age', 'WeeklyListeningHours', 'NumberOfPlaylists', 'FavoriteMusicType']
            st.markdown("#### Mean Values by MeanShift Cluster")
            cluster_summary = df.groupby("MeanShift_Cluster")[feature_columns].mean()
            st.dataframe(cluster_summary)

            # 9. Cluster summary with music type
            st.markdown("#### MeanShift Cluster Summary")
            music_labels = {
                1: 'Rock', 2: 'Pop', 3: 'J-Pop', 4: 'K-Pop', 5: 'EDM',
                6: 'Jazz', 7: 'Chill', 8: 'Oldies', 9: 'Traditional'
            }
            summary = df.groupby('MeanShift_Cluster')[['Age', 'WeeklyListeningHours', 'NumberOfPlaylists']].mean().round(1)
            fav_music = df.groupby('MeanShift_Cluster')['FavoriteMusicType'].agg(lambda x: x.mode().iloc[0])
            cluster_counts = df['MeanShift_Cluster'].value_counts()
            lines = []
            for cluster_id, row in summary.iterrows():
                music_code = fav_music[cluster_id]
                music_label = music_labels.get(music_code, 'Unknown')
                lines.append(f"Cluster {cluster_id}:")
                lines.append(f"- Users in Cluster: {cluster_counts[cluster_id]}")
                lines.append(f"- Average Age: {row['Age']}")
                lines.append(f"- Weekly Listening Hours: {row['WeeklyListeningHours']}")
                lines.append(f"- Average Playlists: {row['NumberOfPlaylists']}")
                lines.append(f"- Most Common Favorite Music Type: {music_label}\n")
            st.text("\n".join(lines))


        elif algorithm == "Agglomerative Hierarchical Clustering":
            st.markdown("### Agglomerative Hierarchical Clustering")
            
            # Sidebar: User chooses number of clusters and dendrogram levels
            n_clusters_user = st.sidebar.slider("Number of Clusters (Agglomerative)", min_value=2, max_value=10, value=2)
            dendro_levels = st.sidebar.slider("Dendrogram Levels to Show (p)", min_value=2, max_value=20, value=5)

            # Agglomerative clustering with user-selected n_clusters
            agg = AgglomerativeClustering(n_clusters=n_clusters_user)
            df['Agglomerative_Cluster'] = agg.fit_predict(scaled_df)

            # Dendrogram
            st.markdown("#### Dendrogram (Ward linkage, truncated)")
            from scipy.cluster.hierarchy import linkage, dendrogram
            fig, ax = plt.subplots(figsize=(12, 6))
            linked = linkage(scaled_df, method='ward')
            dendrogram(linked, truncate_mode='level', p=dendro_levels, ax=ax)
            ax.set_title('Hierarchical Clustering Dendrogram')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Distance')
            st.pyplot(fig)

            # Agglomerative Clustering (PCA Projection)
            st.markdown("#### Agglomerative Clustering (PCA Projection)")
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(scaled_df)
            df['PCA1'] = pca_result[:, 0]
            df['PCA2'] = pca_result[:, 1]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Agglomerative_Cluster', palette='Set1', alpha=0.7, ax=ax)
            ax.set_title('Agglomerative Clustering (PCA Projection)')
            ax.set_xlabel('PCA1')
            ax.set_ylabel('PCA2')
            ax.legend(title='Cluster')
            st.pyplot(fig)

            # Interactive PCA plot with Plotly
            st.markdown("#### Agglomerative Clustering (PCA Projection, Plotly)")
            fig_plotly = px.scatter(
                df,
                x='PCA1',
                y='PCA2',
                color=df['Agglomerative_Cluster'].astype(str),
                title="Agglomerative Clustering (PCA Projection, Plotly)",
                labels={'color': 'Cluster'}
            )
            st.plotly_chart(fig_plotly)

            # Silhouette Score by k (Agglomerative)
            st.markdown("#### Silhouette Score by k (Agglomerative)")
            cluster_range = range(2, 10)
            silhouette_scores = []
            for k in cluster_range:
                agg = AgglomerativeClustering(n_clusters=k)
                labels = agg.fit_predict(scaled_df)
                silhouette_scores.append(silhouette_score(scaled_df, labels))
            best_k_silhouette = cluster_range[np.argmax(silhouette_scores)]
            best_silhouette = np.max(silhouette_scores)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(cluster_range, silhouette_scores, marker='o')
            ax.axvline(x=best_k_silhouette, color='r', linestyle='--', label=f'Best k = {best_k_silhouette}')
            ax.set_title('Silhouette Score')
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel('Score')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            st.info(f'Best Silhouette Score: {best_silhouette:.4f} at k = {best_k_silhouette}')

            # Davies-Bouldin Index by k (Agglomerative)
            st.markdown("#### Davies-Bouldin Index by k (Agglomerative)")
            davies_scores = []
            for k in cluster_range:
                agg = AgglomerativeClustering(n_clusters=k)
                labels = agg.fit_predict(scaled_df)
                davies_scores.append(davies_bouldin_score(scaled_df, labels))
            best_k_davies = cluster_range[np.argmin(davies_scores)]
            best_davies = np.min(davies_scores)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(cluster_range, davies_scores, marker='o', color='red')
            ax.axvline(x=best_k_davies, color='blue', linestyle='--', label=f'Best k = {best_k_davies}')
            ax.set_title('Davies-Bouldin Index')
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel('Score')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            st.info(f'Best Davies-Bouldin Index: {best_davies:.4f} at k = {best_k_davies}')

            # Calinski-Harabasz Index by k (Agglomerative)
            st.markdown("#### Calinski-Harabasz Index by k (Agglomerative)")
            calinski_scores = []
            for k in cluster_range:
                agg = AgglomerativeClustering(n_clusters=k)
                labels = agg.fit_predict(scaled_df)
                calinski_scores.append(calinski_harabasz_score(scaled_df, labels))
            best_k_calinski = cluster_range[np.argmax(calinski_scores)]
            best_calinski = np.max(calinski_scores)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(cluster_range, calinski_scores, marker='o', color='green')
            ax.axvline(x=best_k_calinski, color='r', linestyle='--', label=f'Best k = {best_k_calinski}')
            ax.set_title('Calinski-Harabasz Index')
            ax.set_xlabel('Number of Clusters')
            ax.set_ylabel('Score')
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            st.info(f'Best Calinski-Harabasz Index: {best_calinski:.4f} at k = {best_k_calinski}')

            # 5. Violin plots
            st.markdown("#### Violin Plots by Cluster")
            numeric_features = ['Age', 'WeeklyListeningHours', 'NumberOfPlaylists', 'FavoriteMusicType']
            fig, axes = plt.subplots(1, len(numeric_features), figsize=(5 * len(numeric_features), 5))
            for i, feature in enumerate(numeric_features):
                sns.violinplot(data=df, x=feature, y='Agglomerative_Cluster', ax=axes[i], hue='Agglomerative_Cluster', palette='Set2', orient='h', legend=False)
                axes[i].set_title(f'Violinplot: {feature}')
                axes[i].set_ylabel('Cluster')
                axes[i].set_xlabel(feature)
            plt.tight_layout()
            st.pyplot(fig)

            # 6. Box plots
            st.markdown("#### Box Plots by Cluster")
            fig, axes = plt.subplots(1, len(numeric_features), figsize=(5 * len(numeric_features), 5))
            for i, var in enumerate(numeric_features):
                sns.boxplot(
                    data=df,
                    x=var,
                    y='Agglomerative_Cluster',
                    hue='Agglomerative_Cluster',
                    orient='h',
                    palette='Set2',
                    dodge=False,
                    ax=axes[i]
                )
                axes[i].set_title(f'Box Plot of {var} by Agglomerative Cluster')
                axes[i].legend([], [], frameon=False)
            plt.tight_layout()
            st.pyplot(fig)

            # 7. Cluster summary
            st.markdown("#### Agglomerative Cluster Summary")
            clusters = df['Agglomerative_Cluster'].unique()
            music_type_map = {
                1: 'Rock', 2: 'Pop', 3: 'J-Pop', 4: 'K-Pop', 5: 'EDM',
                6: 'Jazz', 7: 'Chill', 8: 'Oldies', 9: 'Traditional'
            }
            lines = []
            for cluster in sorted(clusters):
                cluster_data = df[df['Agglomerative_Cluster'] == cluster]
                avg_age = cluster_data['Age'].mean()
                avg_hours = cluster_data['WeeklyListeningHours'].mean()
                avg_playlists = cluster_data['NumberOfPlaylists'].mean()
                top_music_code = cluster_data['FavoriteMusicType'].mode()[0]
                top_music = music_type_map.get(top_music_code, "Unknown")
                lines.append(f"Cluster {cluster}:")
                lines.append(f"- Average Age: {avg_age:.1f}")
                lines.append(f"- Weekly Listening Hours: {avg_hours:.1f}")
                lines.append(f"- Average Playlists: {avg_playlists:.1f}")
                lines.append(f"- Most Common Favorite Music Type: {top_music}\n")
            st.text("\n".join(lines))


        elif algorithm == "Compare All Algorithms":
            st.header("Comparison of Clustering Algorithms")

            import time
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

            # Function to evaluate clustering
            def evaluate_clustering(X, labels):
                if len(np.unique(labels)) < 2:
                    return (0, 0, np.inf)  # Invalid clustering
                silhouette = silhouette_score(X, labels)
                calinski = calinski_harabasz_score(X, labels)
                davies = davies_bouldin_score(X, labels)
                return (silhouette, calinski, davies)

            # KMeans
            start_kmeans = time.time()
            best_score_kmeans = {'k': None, 'silhouette': 0, 'calinski': 0, 'davies': np.inf}
            for k in range(2, 10):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(scaled_df)
                s, c, d = evaluate_clustering(scaled_df, labels)
                if s > best_score_kmeans['silhouette']:
                    best_score_kmeans = {'k': k, 'silhouette': s, 'calinski': c, 'davies': d}
            time_kmeans = time.time() - start_kmeans

            # MeanShift
            start_ms = time.time()
            bandwidth_range = np.linspace(0.5, 2.0, 10)
            best_score_meanshift = {'bandwidth': None, 'silhouette': 0, 'calinski': 0, 'davies': np.inf}
            for bandwidth in bandwidth_range:
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
                ms.fit(scaled_df)
                labels = ms.labels_
                s, c, d = evaluate_clustering(scaled_df, labels)
                if s > best_score_meanshift['silhouette']:
                    best_score_meanshift = {'bandwidth': bandwidth, 'silhouette': s, 'calinski': c, 'davies': d}
            time_meanshift = time.time() - start_ms

            # Agglomerative
            start_agg = time.time()
            best_score_agg = {'k': None, 'silhouette': 0, 'calinski': 0, 'davies': np.inf}
            for k in range(2, 10):
                agg = AgglomerativeClustering(n_clusters=k)
                labels = agg.fit_predict(scaled_df)
                s, c, d = evaluate_clustering(scaled_df, labels)
                if s > best_score_agg['silhouette']:
                    best_score_agg = {'k': k, 'silhouette': s, 'calinski': c, 'davies': d}
            time_agg = time.time() - start_agg

            # --- One-Time Execution with Best Parameters ---

            one_time_times = {}

            # KMeans one-time
            start_kmeans_once = time.time()
            kmeans_once = KMeans(n_clusters=best_score_kmeans['k'], random_state=42, n_init='auto')
            kmeans_once.fit(scaled_df)
            one_time_times['KMeans'] = time.time() - start_kmeans_once

            # MeanShift one-time
            start_meanshift_once = time.time()
            meanshift_once = MeanShift(bandwidth=best_score_meanshift['bandwidth'], bin_seeding=True)
            meanshift_once.fit(scaled_df)
            one_time_times['MeanShift'] = time.time() - start_meanshift_once

            # Agglomerative one-time
            start_agg_once = time.time()
            agglo_once = AgglomerativeClustering(n_clusters=best_score_agg['k'])
            agglo_once.fit(scaled_df)
            one_time_times['Agglomerative'] = time.time() - start_agg_once

            # --- Results DataFrame ---
            results_df = pd.DataFrame({
                'Algorithm': ['KMeans', 'MeanShift', 'Agglomerative'],
                'Best k / Bandwidth': [
                    f'k={best_score_kmeans["k"]}',
                    f'bandwidth={best_score_meanshift["bandwidth"]:.2f}',
                    f'k={best_score_agg["k"]}'
                ],
                'Silhouette Score': [
                    best_score_kmeans['silhouette'],
                    best_score_meanshift['silhouette'],
                    best_score_agg['silhouette']
                ],
                'Calinski-Harabasz Index': [
                    best_score_kmeans['calinski'],
                    best_score_meanshift['calinski'],
                    best_score_agg['calinski']
                ],
                'Davies-Bouldin Index': [
                    best_score_kmeans['davies'],
                    best_score_meanshift['davies'],
                    best_score_agg['davies']
                ],
                'Tuning Time (s)': [
                    time_kmeans,
                    time_meanshift,
                    time_agg
                ],
                'One-Time Execution (s)': [
                    one_time_times['KMeans'],
                    one_time_times['MeanShift'],
                    one_time_times['Agglomerative']
                ]
            }).round(4)

            # Display final results in Streamlit
            st.dataframe(results_df)

            # Bar plots for visual comparison
            labels = ['KMeans', 'MeanShift', 'Agglomerative']
            sil_scores = [best_score_kmeans['silhouette'], best_score_meanshift['silhouette'], best_score_agg['silhouette']]
            cal_scores = [best_score_kmeans['calinski'], best_score_meanshift['calinski'], best_score_agg['calinski']]
            db_scores = [best_score_kmeans['davies'], best_score_meanshift['davies'], best_score_agg['davies']]

            x = np.arange(len(labels))
            width = 0.25

            fig, ax = plt.subplots(1, 3, figsize=(15, 4))
            ax[0].bar(x, sil_scores, color='orange', width=width)
            ax[0].set_title('Silhouette Score')
            ax[0].set_xticks(x)
            ax[0].set_xticklabels(labels)

            ax[1].bar(x, cal_scores, color='blue', width=width)
            ax[1].set_title('Calinski-Harabasz Index')
            ax[1].set_xticks(x)
            ax[1].set_xticklabels(labels)

            ax[2].bar(x, db_scores, color='green', width=width)
            ax[2].set_title('Davies-Bouldin Index (lower is better)')
            ax[2].set_xticks(x)
            ax[2].set_xticklabels(labels)

            plt.tight_layout()
            st.pyplot(fig)

            # ---------- RANK-BASED DECISION ----------
            results_df['Silhouette Rank'] = results_df['Silhouette Score'].rank(ascending=False)
            results_df['Calinski Rank'] = results_df['Calinski-Harabasz Index'].rank(ascending=False)
            results_df['Davies Rank'] = results_df['Davies-Bouldin Index'].rank(ascending=True)

            results_df['Total Rank Score'] = (
                results_df['Silhouette Rank'] +
                results_df['Calinski Rank'] +
                results_df['Davies Rank']
            )

            best_algorithm_row = results_df.loc[results_df['Total Rank Score'].idxmin()]
            best_algorithm = best_algorithm_row['Algorithm']

            # ---------- DISPLAY ----------
            st.dataframe(results_df)

            st.subheader("Most Suitable Clustering Algorithm:")
            st.success(f"The most suitable algorithm based on clustering metrics is **{best_algorithm}**.")


        st.markdown("### Download Clustering Result Data")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", csv, "clustered_data.csv", "text/csv")
else:
    st.info("Please upload the music_users.csv file to start.")