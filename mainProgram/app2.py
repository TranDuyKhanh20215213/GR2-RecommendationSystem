import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to get hybrid recommendations (assumed to be pre-defined in your model)
# Content-Based Recommendations
def content_based_recommendations(data, item_name, top_n=10):
    # Check if the item name exists in the training data
    if item_name not in data['Name'].values:
        print(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()

    # Create a TF-IDF vectorizer for item descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF vectorization to item descriptions
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(data['Tags'])

    # Calculate cosine similarity between items based on descriptions
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Find the index of the item
    item_index = data[data['Name'] == item_name].index[0]

    # Get the cosine similarity scores for the item
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort similar items by similarity score in descending order
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Get the top N most similar items (excluding the item itself)
    top_similar_items = similar_items[1:top_n + 1]

    # Get the indices of the top similar items
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Get the details of the top similar items
    recommended_items_details = data.iloc[recommended_item_indices][
        ['Name', 'ImageURL', 'Brand', 'Rating', 'ReviewCount', 'Description', 'Price']]
    # recommended_items_details = data.iloc[recommended_item_indices]

    return recommended_items_details


# Collaborative Filtering Recommendations
def predict_ratings(user_id, clean_data, user_item_matrix, cosine_similarities_df, k=10, number_of_products=10):
    # Bước 1: Lấy index của người dùng từ clean_data
    user_index = clean_data[clean_data['ID'] == user_id].index[0]

    # Bước 2: Lấy k người dùng gần nhất từ cosine similarity
    similar_users = cosine_similarities_df.iloc[user_index].sort_values(ascending=False).iloc[1:k + 1].index.tolist()
    similarity_scores = cosine_similarities_df.iloc[user_index].sort_values(ascending=False).iloc[1:k + 1].values

    # Bước 3: Lấy các sản phẩm mà user chưa đánh giá
    unrated_products = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] == 0].index

    predicted_ratings = {}

    # Bước 4: Dự đoán rating cho từng sản phẩm chưa đánh giá
    for product in unrated_products:
        numerator = 0
        denominator = 0
        for similar_user, similarity_score in zip(similar_users, similarity_scores):
            # Lấy rating của người dùng tương tự cho sản phẩm hiện tại
            similar_user_rating = user_item_matrix.loc[similar_user, product]
            if similar_user_rating > 0:  # Chỉ xem xét sản phẩm mà người dùng tương tự đã đánh giá
                numerator += similarity_score * similar_user_rating
                denominator += abs(similarity_score)

        # Nếu denominator != 0, tính rating dự đoán
        if denominator != 0:
            predicted_ratings[product] = numerator / denominator
        else:
            predicted_ratings[product] = 0  # Nếu không có rating từ người dùng tương tự, gán 0

    # Bước 5: Sắp xếp các sản phẩm theo predicted rating giảm dần và lấy số lượng sản phẩm theo number_of_products
    sorted_predicted_ratings = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)

    # Trả về số lượng sản phẩm được yêu cầu (number_of_products)
    top_recommendations = sorted_predicted_ratings[:number_of_products]

    # Trả về kết quả (prod_id, predicted_rating)
    return top_recommendations

def rating_based_recommendation(data, top_n=10):
    # Calculate average ratings, sort, and select top n items
    top_rated_items = (
        data.groupby(['Name', 'ReviewCount', 'Brand', 'ImageURL'])['Rating']
        .mean().reset_index()
        .sort_values(by=['Rating', 'ReviewCount'], ascending=[False, False])
        .head(top_n)
    )

    # Convert to integer and merge to get all columns, then select necessary columns
    top_rated_items[['Rating', 'ReviewCount']] = top_rated_items[['Rating', 'ReviewCount']].astype(int)
    return top_rated_items.merge(data, on=['Name', 'Rating', 'ReviewCount', 'Brand', 'ImageURL'], how='left')[
        ['Name', 'ImageURL', 'Brand', 'Rating', 'ReviewCount', 'Description', 'Price']
    ]

def collaborative_filtering_recommendations(target_user_id, top_n=10):
    # Load the clean dataset
    df = pd.read_csv('data/clean_data.csv')
    clean_data = df[['ID', 'ProdID', 'Rating']]

    # Create the user-item matrix
    user_item_matrix = clean_data.pivot_table(index='ID', columns='ProdID', values='Rating', fill_value=0)
    user_means = user_item_matrix.mean(axis=1)

    # Normalize ratings by subtracting user means
    normalized_user_item_matrix = user_item_matrix.subtract(user_means, axis=0)
    user_item_matrix_np = normalized_user_item_matrix.values

    # Compute cosine similarity between users
    cosine_similarities = cosine_similarity(user_item_matrix_np)

    # Convert cosine similarities into a DataFrame
    cosine_similarities_df = pd.DataFrame(cosine_similarities, index=user_item_matrix.index,
                                          columns=user_item_matrix.index)

    # Predict ratings using the predict_ratings function
    top_recommendations = predict_ratings(
        user_id=target_user_id,
        clean_data=clean_data,
        user_item_matrix=user_item_matrix,
        cosine_similarities_df=cosine_similarities_df,
        k=10,
        number_of_products=top_n
    )

    # Extract product IDs from the recommendations
    recommended_product_ids = [prod_id for prod_id, _ in top_recommendations]

    # Extract details of the recommended products from the original dataset
    recommended_item_indices = df[df['ProdID'].isin(recommended_product_ids)].index
    recommended_items_details = df.iloc[recommended_item_indices][
        ['Name', 'ImageURL', 'Brand', 'Rating', 'ReviewCount', 'Description', 'Price']
    ]

    # Return the detailed recommended items
    return recommended_items_details.head(top_n)


def hybrid_recommendations(data, target_user_id, item_name, top_n=10):
    # Get content-based recommendations
    content_based_recs = content_based_recommendations(data, item_name, top_n)

    # Get collaborative filtering recommendations (top 10)
    collaborative_recs = collaborative_filtering_recommendations(target_user_id, top_n=top_n)

    # Extract the top 5 from content-based recommendations
    content_based_top_5 = content_based_recs.head(5)

    # Extract the top 10 from collaborative filtering recommendations
    collaborative_top_10 = collaborative_recs.head(10)

    # Get product IDs from content-based recommendations
    content_based_ids = content_based_top_5['Name'].values

    # Filter out any items from collaborative recommendations that are already in the top 5 content-based
    collaborative_top_10_filtered = collaborative_top_10[~collaborative_top_10['Name'].isin(content_based_ids)]

    # Select 5 items from collaborative filtering that are not in the top 5 of content-based recommendations
    collaborative_top_5_unique = collaborative_top_10_filtered.head(5)

    # Combine the top 5 from content-based and the 5 remaining from collaborative filtering
    final_recommendations = pd.concat([content_based_top_5, collaborative_top_5_unique])

    return final_recommendations
# Load the dataset
data_file = "data/clean_data.csv"
all_products_data = pd.read_csv(data_file)

# Flags
history_value = 0
choose_value = 0
product_name = ""

# Streamlit app title
st.title("User and Product Recommendations")

# User input for User ID
user_id = st.number_input("Enter your User ID:", min_value=1, step=1)

# Initialize session state for flags
if 'history_value' not in st.session_state:
    st.session_state.history_value = 0
if 'choose_value' not in st.session_state:
    st.session_state.choose_value = 0

# Check user history when button is pressed
if st.button("Check History"):
    st.session_state.choose_value = 0  # Reset product selection when checking history
    user_history = all_products_data[(all_products_data['ID'] == user_id) & (all_products_data['Rating'] > 0)]
    if not user_history.empty:
        st.session_state.history_value = 1
        st.write(f"User {user_id} History:")
        st.write(user_history[['Name', 'Rating']])
    else:
        st.session_state.history_value = 0
        st.write(f"No history found for User {user_id}.")

# Input for product selection
product_name = st.selectbox("Select a Product:", options=all_products_data['Name'].unique())

# Button to confirm product selection
if st.button("OK"):
    st.session_state.choose_value = 1  # Update the choose_value when the user selects a product
    st.write(f"Product selected: {product_name}")

# Recommendations button
if st.button("Recommendations"):
    # Get the current history and choose values from session state
    history_value = st.session_state.history_value
    choose_value = st.session_state.choose_value

    if history_value == 0 and choose_value == 0:
        st.subheader("Top-Rated Products:")
        top_rated_items = rating_based_recommendation(all_products_data)
        for _, row in top_rated_items.iterrows():
            st.write(f"**{row['Name']}**")
            st.write(f"   Brand: {row['Brand']}")
            st.write(f"   Review Count: {row['ReviewCount']}")
            st.write(f"   Price: ${row['Price']}")
            st.write(f"   Description: {row['Description']}")
    elif history_value == 0 and choose_value == 1:
        st.subheader(f"Content-Based Recommendations for '{product_name}':")
        recommendations = content_based_recommendations(all_products_data, product_name)
        for _, row in recommendations.iterrows():
            st.write(f"**{row['Name']}**")
            st.write(f"   Brand: {row['Brand']}")
            st.write(f"   Review Count: {row['ReviewCount']}")
            st.write(f"   Price: ${row['Price']}")
            st.write(f"   Description: {row['Description']}")
    elif history_value == 1 and choose_value == 0:
        st.subheader(f"Collaborative Filtering Recommendations for User {user_id}:")
        recommendations = collaborative_filtering_recommendations(user_id)
        for _, row in recommendations.iterrows():
            st.write(f"**{row['Name']}**")
            st.write(f"   Brand: {row['Brand']}")
            st.write(f"   Review Count: {row['ReviewCount']}")
            st.write(f"   Price: ${row['Price']}")
            st.write(f"   Description: {row['Description']}")
    elif history_value == 1 and choose_value == 1:
        st.subheader(f"Hybrid Recommendations for '{product_name}':")
        recommendations = hybrid_recommendations(all_products_data, user_id , product_name, 10)
        for _, row in recommendations.iterrows():
            st.write(f"**{row['Name']}**")
            st.write(f"   Brand: {row['Brand']}")
            st.write(f"   Review Count: {row['ReviewCount']}")
            st.write(f"   Price: ${row['Price']}")
            st.write(f"   Description: {row['Description']}")
