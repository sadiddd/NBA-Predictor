import pandas as pd

games = pd.read_csv("games.csv")

# convert necessary values to usuable ints and dates to be processed by model
games["Date"] = pd.to_datetime(games["Date"])
games["Home_Code"] = games["Home/Neutral"].astype("category").cat.codes
games["Away_Code"] = games["Visitor/Neutral"].astype("category").cat.codes
games["Hour"] = pd.to_datetime(games["Date"], format="%I:%M%p").dt.hour
games["Day"] = games["Date"].dt.day_of_week
games["Target"] = (games["Home_Code"] > games["Away_Code"]).astype(int)

from sklearn.ensemble import RandomForestClassifier # import model from sklearn

rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)
train = games[games["Date"] < "2025-01-01"]
test = games[games["Date"] > "2025-01-01"]