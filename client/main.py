# client/main.py

from api_client import classify, get_features
from user_interface import UserInterface


def main():
    ui = UserInterface()
    ui.show_welcome_message()
    # Fetch available feature values from server
    features = get_features()
    ui.show_feature_options(features)
    # Start interactive classification via API
    ui.interactive_classification(classify, features)


if __name__ == "__main__":
    main()
