from data_cleaner import clean_data
import pandas as pd

def main(data_path: str) -> None:
	# CLEAN DATA
	data = pd.read_csv(data_path)
	data = clean_data(data)

	# next steps like feature engineering, blah blah

if __name__ == "__main__":
	main("data/airport.csv")