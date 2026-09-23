import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import dotenv

dotenv.load_dotenv()

# Read token from environment variable
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

CHANNEL_ID = "C0AEPUNEM1A"

def send_slack_message(text: str):
    client = WebClient(token=SLACK_BOT_TOKEN)

    try:
        response = client.chat_postMessage(
            channel=CHANNEL_ID,
            text=text
        )
        print("✅ Message sent successfully:", response["ts"])

    except SlackApiError as e:
        print("❌ Error sending message:", e.response["error"])


if __name__ == "__main__":
    send_slack_message("🚀 Slack notification test from Python!")
