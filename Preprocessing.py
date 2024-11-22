import pandas as pd
from sklearn.preprocessing import LabelEncoder
from textblob import TextBlob
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import json

def review_sentiment_score(review_text):
    return TextBlob(review_text).sentiment.polarity

def process_reviews(data):
    review_data = []

    for idx, row in data.iterrows():
        reviews = row['reviews']
        if not isinstance(reviews, str):
            review_data.append((idx, 0, 0, 0.0, 0, 0.0))  # If no review text, store zeros
        else:
            review_list = reviews.split("\n---------------------------------\n")
            positive_count = 0
            negative_count = 0
            positive_sum = 0.0
            negative_sum = 0.0

            for review in review_list:
                score = review_sentiment_score(review)
                if score > 0:
                    positive_count += 1
                    positive_sum += score
                elif score < 0:
                    negative_count += 1
                    negative_sum += score

            review_data.append((idx, len(review_list), positive_count, positive_sum, negative_count, negative_sum))

    # Convert the sentiment data to a DataFrame
    sentiment_df = pd.DataFrame(review_data, columns=['index', 'review_count', 'positive_count', 'positive_sum', 'negative_count', 'negative_sum'])

    # Save the sentiment data separately
    sentiment_df.to_csv("./stored_data/sentiment_data.csv", index=False)

    return sentiment_df

def load_sentiment_data():
    try:
        # Load precomputed sentiment data
        sentiment_df = pd.read_csv("./stored_data/sentiment_data.csv")
        print("Loaded precomputed sentiment data.")
    except FileNotFoundError:
        # If the sentiment data is not found, calculate it
        print("No precomputed data found. Calculating sentiment data...")
        # Load your original data here, e.g., from 'original_data.csv'
        data = pd.read_csv("train.csv")
        sentiment_df = process_reviews(data)
    
    return sentiment_df




def preprocess_data(data_type):
    # Load the original data and sentiment data separately
    if data_type=='train':
        original_data = pd.read_csv('train.csv', parse_dates=['host_since', 'first_review', 'last_review'])
    elif data_type=='test':
        original_data = pd.read_csv('test.csv', parse_dates=['host_since', 'first_review', 'last_review'])
        
    sentiment_data = load_sentiment_data()

    # Merge the sentiment data with the original data using the index or an ID column
    data = original_data.merge(sentiment_data, left_index=True, right_on='index', how='left')

    # --------- Numeric Columns -------------
    # Filling with mean
    score_cols = ['host_acceptance_rate', 'host_response_rate', 'review_scores_rating', 'review_scores_accuracy', 
                'review_scores_cleanliness', 'review_scores_checkin', 
                'review_scores_communication', 'review_scores_location', 
                'review_scores_value', 'reviews_per_month']
    for col in score_cols:
        mean_value = data[col].mean()
        data[col] = data[col].fillna(mean_value)
    #-----------Number/Boolean----------------
    # Filling with Mode
    mode_fill_cols = ['host_is_superhost', 'host_response_time', 'bedrooms', 'beds']
    for col in mode_fill_cols:
        data[col] = data[col].fillna(data[col].mode()[0])
    data['host_is_superhost'] = data['host_is_superhost'].astype(int) # Convert to integer
    # Filling with zeros
    data['has_availability'] = data['has_availability'].apply(lambda x: 1 if x is True else 0)

    #Filiing with other feature
    data['bathrooms'] = data['bathrooms'].fillna(data['bedrooms'])  # If missing, use the number of bedrooms

    # --------- Text Columns -------------
    # description → # of words 
    data['description'] = data['description'].fillna('')  # Replace NaNs with an empty string
    data['description'] = data['description'].apply(lambda x: len(x) if x != '' else 0)
    data['name'] = data['name'].fillna('')  # Replace NaNs with an empty string
    data['name'] = data['name'].apply(lambda x: len(x) if x != '' else 0)

    # category → ordered numbers
    response_time_mapping = {
        "within an hour": 1,
        "within a few hours": 2,
        "within a day": 3,
        "a few days or more": 4
    }
    data['host_response_time'] = data['host_response_time'].map(response_time_mapping)

    # category → numbers
    data['bathrooms_text'] = data['bathrooms_text'].fillna('')  # Fill NaNs with an empty string temporarily
    # Define functions to check for each type
    def check_private(text):
        return 1 if "private" in text.lower() else 0

    def check_shared(text):
        return 1 if "shared" in text.lower() else 0

    def check_no_description(text):
        return 1 if text.strip() == '' else 0
    # Apply the functions to create new columns
    data['bath_private'] = data['bathrooms_text'].apply(check_private)
    data['bath_shared'] = data['bathrooms_text'].apply(check_shared)
    data['bath_x'] = data['bathrooms_text'].apply(check_no_description)
    data.loc[data['bathrooms_text'].str.contains("half", case=False, na=False), 'bathrooms_text'] = 0.5
    data['bathrooms_text'] = data['bathrooms_text'].apply(lambda x: float(x) if isinstance(x, float) else float(x.split()[0]) if x else 0)

    # Create binary features based on keywords in `property_type`
    data['is_private'] = data['property_type'].str.contains('Private', case=False, na=False).astype(int)
    data['is_shared'] = data['property_type'].str.contains('Shared', case=False, na=False).astype(int)
    data['is_entire'] = data['property_type'].str.contains('Entire', case=False, na=False).astype(int)
    data['is_room'] = data['property_type'].str.contains('room', case=False, na=False).astype(int)
    data['is_suite'] = data['property_type'].str.contains('suite', case=False, na=False).astype(int)
    data['is_hotel'] = data['property_type'].str.contains('hotel', case=False, na=False).astype(int)
    data['is_condo'] = data['property_type'].str.contains('condo', case=False, na=False).astype(int)
    data['is_guesthouse'] = data['property_type'].str.contains('guest', case=False, na=False).astype(int)
    data['is_apartment'] = data['property_type'].str.contains('apartment', case=False, na=False).astype(int)
    data['is_vacation_home'] = data['property_type'].str.contains('vacation', case=False, na=False).astype(int)
    data = data.drop(columns=['property_type'])


    # --------- Dates Columns -------------
    # Dates: convert to days / filling with zeros
    reference_date = data[['first_review', 'last_review']].min().min()
    data['first_review'] = (data['first_review'] - reference_date).dt.days
    data['last_review'] = (data['last_review'] - reference_date).dt.days
    data['first_review'] = data['first_review'].fillna(0)
    data['last_review'] = data['last_review'].fillna(0)

    host_reference_date = data['host_since'].min()
    data['host_since'] = (data['host_since'] - host_reference_date).dt.days
    data['host_since'] = data['host_since'].fillna(0)

    # --------- List Columns -------------
    # Review: Calculate average positive and negative sentiment scores per review
    data['positive_review_avg'] = data.apply(lambda row: row['positive_sum'] / row['positive_count'] if row['positive_count'] > 0 else 0, axis=1)
    data['negative_review_avg'] = data.apply(lambda row: row['negative_sum'] / row['negative_count'] if row['negative_count'] > 0 else 0, axis=1)

    data = data.drop('reviews', axis=1) # Drop the original 'reviews' column if no longer needed

    data['amenities'] = data['amenities'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Amenities
    # # TF-IDF
    data['amenities_text'] = data['amenities'].apply(lambda x: ' '.join([item.lower() for item in x]))
    tfidf = TfidfVectorizer(max_features=100)  # Use top 100 amenities or adjust as needed
    tfidf_matrix = tfidf.fit_transform(data['amenities_text'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out(), index=data.index)
    data = pd.concat([data, tfidf_df], axis=1)
    data = data.drop(columns=['amenities_text'])

    # top_amenities_per_category = {}
    # if data_type=='train':
    #     price_categories = data['price'].unique()  # Assuming you have a 'price_category' column
    #     for category in price_categories:
    #         category_data = data[data['price'] == category]
    #         amenities_flat = [amenity for sublist in category_data['amenities'] for amenity in sublist]
    #         amenities_counter = Counter(amenities_flat)
    #         top_amenities_per_category[category] = [amenity for amenity, _ in amenities_counter.most_common(30)]
    #     top_amenities_union = set()
    #     for amenities in top_amenities_per_category.values():
    #         top_amenities_union.update(amenities)
    #     top_amenities_union = sorted(top_amenities_union)
    #     with open('top_amenities_union.json', 'w') as file:
    #         json.dump(top_amenities_union, file)
    # elif data_type=='test':
    #     with open('top_amenities_union.json', 'r') as file:
    #         top_amenities_union = json.load(file)
            
    # for amenity in top_amenities_union:
    #     data[f'has_{amenity}'] = data['amenities'].apply(lambda x: 1 if amenity in x else 0)
    data = data.drop('amenities', axis=1)

    label_encoder = LabelEncoder()
    # Label encoding for categorical features
    for col in ['neighbourhood_group_cleansed', 'host_verifications', 'room_type']:
        data[col] = label_encoder.fit_transform(data[col])

    # Calculate the average price for each neighborhood and neighborhood group
    if data_type=='train':
        neighborhood_price_avg = data.groupby('neighbourhood_cleansed')['price'].mean()
        neighborhood_group_price_avg = data.groupby('neighbourhood_group_cleansed')['price'].mean()

        neighborhood_price_avg.to_csv('./stored_data/neighborhood_price_avg.csv', index=True)
        neighborhood_group_price_avg.to_csv('./stored_data/neighborhood_group_price_avg.csv', index=True)
    
    elif data_type=='test':
        # Calculate the average price for each neighborhood and neighborhood group
        neighborhood_price_avg = pd.read_csv('./stored_data/neighborhood_price_avg.csv', index_col=0)
        neighborhood_price_avg = neighborhood_price_avg.to_dict()  # Convert to dictionary for mapping
        neighborhood_price_avg = neighborhood_price_avg['price']
        neighborhood_price_avg = {k.strip().lower(): v for k, v in neighborhood_price_avg.items()}
        data['neighbourhood_cleansed'] = data['neighbourhood_cleansed'].str.strip().str.lower()

        neighborhood_group_price_avg = pd.read_csv('./stored_data/neighborhood_group_price_avg.csv', index_col=0)
        neighborhood_group_price_avg = neighborhood_group_price_avg.to_dict()  # Convert to dictionary for mapping
        neighborhood_group_price_avg = neighborhood_group_price_avg['price']


    # Map these averages to the respective columns
    data['neighbourhood_cleansed'] = data['neighbourhood_cleansed'].map(neighborhood_price_avg)
    average_value = data['neighbourhood_cleansed'].mean()  # Replace with the appropriate default value
    data['neighbourhood_cleansed'] = data['neighbourhood_cleansed'].map(neighborhood_price_avg).fillna(average_value)
    data['neighbourhood_group_cleansed'] = data['neighbourhood_group_cleansed'].map(neighborhood_group_price_avg)


    #--------Geolocation ------------------
    # Set the number of clusters based on how granular you want the location feature to be
    n_clusters = 30
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    # Fit K-means and assign cluster IDs
    data['kmeans_location_feature'] = kmeans.fit_predict(data[['latitude', 'longitude']])
    # Apply PCA to reduce the combined latitude, longitude, and price to one component
    pca = PCA(n_components=1)
    data['pca_location_price_feature'] = pca.fit_transform(data[['latitude', 'longitude']])

    # --- Feature and Target Separation ---
    return data
        
        
        
        
    