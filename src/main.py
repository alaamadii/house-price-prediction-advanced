from data_loader import load_data
from pipeline_training import save_model_artifact, train_and_evaluate_models


def main():
	df = load_data("data/train.csv")
	leaderboard, _, _, artifact = train_and_evaluate_models(df)
	save_model_artifact(artifact, "models/best_model.joblib")

	print("\nModel Performance Comparison")
	print(leaderboard[["Model", "MAE", "RMSE", "R2", "CV RMSE"]])
	print("\nBest model saved to models/best_model.joblib")


if __name__ == "__main__":
	main()
