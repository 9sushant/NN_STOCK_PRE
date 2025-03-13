import tweepy
import csv
import numpy as np
from textblob import TextBlob
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#Step 1 - Insert your API keys
consumer_key= 'ekjdv2ahlfQW1U3LDHjvgfMNq'
consumer_secret= 'Vw2X8Qs3vk6xKLpcHPX91DsO6nQo9Xo4cToZc1tMJZPe9m0tnG'
access_token='1741529285595389952-PNjbFbX3sa25VoD6WJZ9nVstVGOKGT'
access_token_secret='tNsGI5VTUiotgrnKeCKtcD3DEr6EsQVY3pOOt6r8ANGta'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

#Step 2 - Search for your company name on Twitter
#Step 2 - Search for your company name on Twitter
public_tweets = api.search_tweets('GOOGLE') # Changed from api.search to api.search_tweets


##Step 3 - Define a threshold for each sentiment to classify each
#as positive or negative. If the majority of tweets you've collected are positive
#then use your neural network to predict a future price
positive_threshold = 0.1  # Adjust as needed
negative_threshold = -0.1 # Adjust as needed
positive_tweets_count = 0
negative_tweets_count = 0
neutral_tweets_count = 0

for tweet in public_tweets:
    analysis = TextBlob(tweet.text)
    sentiment = analysis.sentiment
    polarity = sentiment.polarity
    subjectivity = sentiment.subjectivity

    print(f"Tweet: {tweet.text}")
    print(f"Sentiment: Polarity={polarity:.3f}, Subjectivity={subjectivity:.3f}")

    if polarity > positive_threshold:
        print("Sentiment: Positive")
        positive_tweets_count += 1
    elif polarity < negative_threshold:
        print("Sentiment: Negative")
        negative_tweets_count += 1
    else:
        print("Sentiment: Neutral")
        neutral_tweets_count += 1
    print("-" * 30) # Separator

print("\nSentiment Summary:")
total_tweets = positive_tweets_count + negative_tweets_count + neutral_tweets_count
print(f"Total Tweets Analyzed: {total_tweets}")
print(f"Positive Tweets: {positive_tweets_count} ({positive_tweets_count/total_tweets*100:.2f}%)")
print(f"Negative Tweets: {negative_tweets_count} ({negative_tweets_count/total_tweets*100:.2f}%)")
print(f"Neutral Tweets: {neutral_tweets_count} ({neutral_tweets_count/total_tweets*100:.2f}%)")

# Determine overall sentiment (example logic - adjust as needed)
if positive_tweets_count > negative_tweets_count:
    print("\nOverall Sentiment: Positive (Majority Positive Tweets)")
elif negative_tweets_count > positive_tweets_count:
    print("\nOverall Sentiment: Negative (Majority Negative Tweets)")
else:
    print("\nOverall Sentiment: Neutral (Balanced Sentiment)")
    
dates = []
prices = []
def get_data(filename):
	with open(filename, 'r') as csvfile:
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader) # Skip header row
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0])) # Assuming date is in YYYY-MM-DD format
			prices.append(float(row[1])) # Assuming price is in the second column
	return

#Step 5 reference your CSV file here
get_data('GOOGL.csv') # Replace with your actual CSV filename

#Step 6 In this function, build your neural network model using Keras, train it, then have it predict the price 
#on a given day. We'll later print the price out to terminal.

# ... (rest of your code above) ...

#Step 6 In this function, build your neural network model using Keras, train it, then have it predict the price
#on a given day. We'll later print the price out to terminal.
def predict_prices(dates, prices, x):
    # 1. Data Preprocessing
    dates = np.array(dates).reshape(-1, 1) # Reshape dates to be 2D array
    prices = np.array(prices)

    # Scale dates and prices (important for neural networks)
    date_scaler = StandardScaler()
    price_scaler = StandardScaler()
    scaled_dates = date_scaler.fit_transform(dates)
    scaled_prices = price_scaler.fit_transform(prices.reshape(-1, 1)).flatten() # Scale and flatten prices
    x_scaled = date_scaler.transform(np.array([[x]]))  # Scale the input date 'x'

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_dates, scaled_prices, test_size=0.2, random_state=42)

    # 2. Build Neural Network Model
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=1)) # Input layer and hidden layer
    model.add(Dense(units=32, activation='relu'))           # Another hidden layer
    model.add(Dense(units=1, activation='linear'))          # Output layer (linear activation for regression)

    # 3. Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error') # Adam optimizer, MSE loss

    # 4. Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0) # Train for 100 epochs

    # 5. Evaluate (Optional - but good practice)
    loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Neural Network Test Loss: {loss:.4f}")

    # 6. Make Prediction (using scaled input)
    predicted_price_scaled = model.predict(x_scaled)

    # 7. Inverse Transform the Prediction to original scale
    predicted_price = price_scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1)).flatten()[0]

    return predicted_price


# ... (rest of your code - get_data call, sentiment analysis if needed) ...

# Step 7 - Example usage: Predict price for a date (replace 2018 with your desired year)
predicted_price = predict_prices(dates, prices, 2018) # Predict for year 2018 (adjust as needed)
print(f"\nPredicted Stock Price for year 2018: {predicted_price:.2f}")